import os
import time

import skimage.transform
import cv2
from scipy.ndimage import median_filter
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import numbers

from utils import geometry
from utils import log as log
from skimage import exposure
import config as cfg
# import torch.nn.functional as F
import torchvision.transforms.functional as F


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def crop_by_bb(img, bb, size):
    x, y, w, h = bb
    x, y, x2, y2 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
    crop =  img[y:y2, x:x2]
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    # image normalization
    if cfg.WITH_HIST_NORM:
        p2, p98 = np.percentile(crop, (2, 98))
        crop = exposure.rescale_intensity(crop, in_range=(p2, p98))
    return crop


def scale_depth(img):
    depth = img.astype(np.float32) / 2**16
    depth[depth > 0.93] = 0.93
    depth[depth < 0.80] = 0.80
    # depth[mask] = 0.93 #np.max(depth[~mask])
    depth = normalize(depth)
    return median_filter(depth, (3,3))


def scale_bump(img):
    bm = img.astype(np.float32)
    # rescale
    bm -= 65536/2

    # remove bad pixels
    max_lim = 200
    min_lim = -200
    bm[bm>max_lim] = 0
    bm[bm<min_lim] = 0

    # limit bumps to reasonable values
    max_lim = 100
    min_lim = -100
    bm[bm>max_lim] = 100
    bm[bm<min_lim] = 0

    return median_filter(bm, (3,3))


def align_bump(bm, lms_):
    lms = lms_.copy()
    img_shape = bm.shape
    cy,cx = np.array(img_shape[:2])/2

    # scale with based on eye distance
    d_eyes_inner = lms[42,0] - lms[39,0]
    d_eyes_outer = lms[45,0] - lms[36,0]
    scale = 70.0/d_eyes_inner
    # scale = 100.0/d_eyes_outer
    # scale = 1.0

    # scale with based on nose length
    d_nose = lms[30,1] - (lms[42,1] + lms[39,1]) / 2.0
    # scale_y = 70.0/d_nose
    scale_y = 1.0  # FIXME: debug

    # resize img to fit eye dist. and nose length
    lms[:,0] *= scale
    lms[:,1] *= scale_y
    bm = cv2.resize(bm, None, fx=scale, fy=scale_y)

    # translate face based on m
    eye_l, eye_r = lms[36,:].astype(int),  lms[45,:].astype(int)
    eye_mid = (eye_l + eye_r) / 2
    # nose_x, nose_y = lms[30,:]
    tx = cx-int(eye_mid[0])
    ty = cy-70-int(eye_mid[1])
    bm = np.roll(bm, tx, axis=1)
    bm = np.roll(bm, ty, axis=0)
    lms[:,0] += tx
    lms[:,1] += ty

    # return cv2.resize(bm_cropped, (dst_size,dst_size))
    return bm[:img_shape[0], :img_shape[1]], lms

import PIL
def align_face(im, lms):

    # fig, ax = plt.subplots(1,2, figsize=(12,8))
    # ax[0].imshow(im, interpolation='nearest')

    cy,cx = np.array(im.shape[:2])/2

    # move to center
    eye_l, eye_r = lms[36,:].astype(int),  lms[45,:].astype(int)
    eye_mid = (eye_l + eye_r) / 2
    # nose_x, nose_y = lms[30,:]
    tx = cx-int(eye_mid[0])
    ty = cy-int(eye_mid[1])
    im = np.roll(im, tx, axis=1)
    im = np.roll(im, ty, axis=0)
    lms[:,0] += tx
    lms[:,1] += ty
    eye_l, eye_r = lms[36,:].astype(int),  lms[45,:].astype(int)
    ex, ey = eye_r.astype(float)-250.0
    angle =  np.rad2deg(np.sin(ey/ex))

    pil_im = PIL.Image.fromarray(im*255)
    pil_im = pil_im.rotate(angle, PIL.Image.BICUBIC)
    # pil_im.show()
    #
    # ax[1].imshow(im, interpolation='nearest', cmap=plt.cm.gray)
    # for i in range(30,60):
    #     sc = ax[1].scatter(lms[i,0], lms[i,1], s=10.0)
    # plt.show()

    im = np.array(pil_im)/255.0

    return im, lms


def scale_landmarks_to_crop(lms, output_size):
    if len(lms) == 68:
        eye_l, eye_r = lms[36,:].astype(int),  lms[45,:].astype(int)  # outer landmarks
    elif len(lms) == 21:
        eye_l, eye_r = lms[6,:].astype(int),  lms[11,:].astype(int)  # outer landmarks
    elif len(lms) == 5:
        eye_l, eye_r = lms[0,:].astype(int),  lms[1,:].astype(int)  # center landmarks
    else:
        raise ValueError('Unknown landmark format!')

    if eye_r[0] < eye_l[0]:
        eye_r, eye_l = eye_l, eye_l

    w = eye_r[0] - eye_l[0]
    t = lms[:, 1].min()
    b = lms[:, 1].max()
    if t > b:
        t, b = b, t
    h = b - t
    assert(h >= 0)

    # extended for training unsupervised AAE
    min_col, max_col = int(eye_l[0]-0.35*w), int(eye_r[0]+0.35*w)
    min_row, max_row = int(t-0.25*h), int(b+0.05*h)

    # in case right eye is actually left of right eye...
    if min_col > max_col:
        min_col, max_col = max_col, min_col

    height = max_row - min_row
    width = max_col - min_col

    lms_new = lms.copy()
    lms_new[:,0] = (lms_new[:,0] - min_col) * output_size[0]/width
    lms_new[:,1] = (lms_new[:,1] - min_row) * output_size[1]/height
    return lms_new


def crop_celeba(img, output_size, cx=89, cy=121):
    crop = img[cy - 64: cy + 64, cx - 64: cx + 64]
    crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_CUBIC)
    # image normalization
    if cfg.WITH_HIST_NORM:
        p2, p98 = np.percentile(crop, (2, 98))
        crop = exposure.rescale_intensity(crop, in_range=(p2, p98))
    return crop


def mirror_padding(img):
    EXTEND_BLACK = True
    if EXTEND_BLACK:
        empty = np.zeros_like(img)
        center_row = np.hstack((empty, img, empty))
        flipped_ud = np.flipud(np.hstack((empty, empty, empty)))
        result = np.vstack((flipped_ud, center_row, flipped_ud))
    else:
        s = int(img.shape[0]*0.1)
        k = (s,s)
        # blurred = cv2.blur(cv2.blur(cv2.blur(img, k), k), k)
        blurred = cv2.blur(img, k)
        flipped_lr = np.fliplr(blurred)
        center_row = np.hstack((flipped_lr, img, flipped_lr))
        flipped_ud = np.flipud(np.hstack((flipped_lr, blurred, flipped_lr)))
        result = np.vstack((flipped_ud, center_row, flipped_ud))
    return result


def scale_landmarks_celebHQ(bm, lms_orig, output_size):

    lms = lms_orig.copy()

    if len(lms) == 68:
        eye_l, eye_r = (lms[36]+lms[39])/2,  (lms[42]+lms[45])/2  # center landmarks
        mouth_l, mouth_r = lms[48],  lms[54]  # outer landmarks
    elif len(lms) == 21:
        eye_l, eye_r = lms[6,:].astype(int),  lms[11,:].astype(int)  # outer landmarks
    elif len(lms) == 5:
        eye_l, eye_r = lms[0,:].astype(int),  lms[1,:].astype(int)  # center landmarks
    else:
        raise ValueError('Unknown landmark format!')

    if eye_r[0] < eye_l[0]:
        eye_r, eye_l = eye_l, eye_l

    eye_c = (eye_l+eye_r)/2
    w = eye_r - eye_l
    h = eye_c - (mouth_l+mouth_r)/2
    c = eye_c - 0.25*h
    s = int(1.6*np.linalg.norm(h))

    vx = w / np.linalg.norm(w)
    vy = h / np.linalg.norm(h)
    angle_x = np.arcsin(vx[1])
    angle_y = np.arcsin(vy[0])
    angle_x_deg = np.rad2deg(angle_x)
    M = cv2.getRotationMatrix2D(tuple(c), angle_x_deg, 1.0)

    lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))
    lms_crop = M.dot(lms_hom.T).T

    tl = c - s
    br = c + s

    tl = tl.astype(int)
    br = br.astype(int)

    width = br[0] - tl[0]
    height =  br[1] - tl[1]

    lms_new = lms_crop.copy()
    lms_new[:,0] = (lms_new[:,0] - tl[0]) * output_size[0]/width
    lms_new[:,1] = (lms_new[:,1] - tl[1]) * output_size[1]/height
    return lms_new


def make_square(bb):
    h = bb[3] - bb[1]
    bb1 = geometry.convertBB2to1(bb)
    bb1[2] = h/2
    return geometry.convertBB1to2(bb1)


def get_bbox_from_landmarks(landmarks, loose=False, loose_scale=cfg.LOOSE_BBOX_SCALE):
    tl = landmarks.min(axis=0)
    br = landmarks.max(axis=0)
    bb = np.concatenate((tl, br))
    bb = make_square(bb)
    if loose:
        if len(landmarks) < 68:
            bb = geometry.scaleBB(bb, 1.0, 1.0, typeBB=2)
        bb = geometry.scaleBB(bb, loose_scale, loose_scale, typeBB=2)
    return bb.astype(int)


class FaceCrop():
    def __init__(self, img, output_size=(cfg.CROP_SIZE, cfg.CROP_SIZE), bbox=None, landmarks=None,
                 img_already_cropped=False, crop_by_eye_mouth_dist=False, align_face_orientation=True,
                 crop_square=False, loose=False, loose_scale=cfg.LOOSE_BBOX_SCALE):
        assert(bbox is not None or landmarks is not None)
        self.output_size = output_size
        self.align_face_orientation = align_face_orientation
        self.M = None
        self.img_already_cropped = img_already_cropped
        self.tl = None
        self.br = None
        self.loose = loose
        self.loose_scale = loose_scale
        if bbox is not None:
            bbox = np.asarray(bbox)
            self.tl = bbox[:2].astype(int)
            self.br = bbox[2:4].astype(int)
        self.lms = landmarks
        self.img = img
        self.angle_x_deg = 0
        if landmarks is not None:
            self.calculate_crop_parameters(img, landmarks, img_already_cropped)

    def __get_eye_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_eye_l, id_eye_r = [36, 39],  [42, 45]
        elif lms.shape[0] == 37:
            id_eye_l, id_eye_r = [13, 16],  [19, 22]
        elif lms.shape[0] == 21:
            id_eye_l, id_eye_r = [6, 8],  [9, 11]
        elif lms.shape[0] == 4:
            id_eye_l, id_eye_r = [0],  [1]
        elif lms.shape[0] == 5:
            # id_eye_l, id_eye_r = [1],  [2]
            id_eye_l, id_eye_r = [0],  [1]
        else:
            raise ValueError("Invalid landmark format!")
        eye_l, eye_r = lms[id_eye_l].mean(axis=0),  lms[id_eye_r].mean(axis=0)  # center landmarks
        if eye_r[0] < eye_l[0]:
            eye_r, eye_l = eye_l, eye_r
        return eye_l, eye_r

    def __get_mouth_coordinates(self, lms):
        if lms.shape[0] == 68:
            id_mouth_l, id_mouth_r = 48,  54
        elif lms.shape[0] == 37:
            id_mouth_l, id_mouth_r = 25,  31
        elif lms.shape[0] == 21:
            id_mouth_l, id_mouth_r = 17,  19
        elif lms.shape[0] == 4:
            id_mouth_l, id_mouth_r = 2,  3
        elif lms.shape[0] == 5:
            id_mouth_l, id_mouth_r = 3, 4
        else:
            raise ValueError("Invalid landmark format!")
        return lms[id_mouth_l],  lms[id_mouth_r]  # outer landmarks

    def get_face_center(self, lms, return_scale=False):
        eye_l, eye_r = self.__get_eye_coordinates(lms)
        mouth_l, mouth_r = self.__get_mouth_coordinates(lms)
        eye_c = (eye_l+eye_r)/2
        vec_nose = eye_c - (mouth_l+mouth_r)/2
        c = eye_c - 0.25*vec_nose

        if return_scale:
            s = int(1.60*np.linalg.norm(vec_nose))
            return c, s
        else:
            return c

    def __calc_rotation_matrix(self, center, eye_l, eye_r):
        vec_eye = eye_r - eye_l
        w = np.linalg.norm(vec_eye)
        vx = vec_eye / np.linalg.norm(w)
        angle_x = np.arcsin(vx[1])
        self.angle_x_deg = np.rad2deg(angle_x)
        return cv2.getRotationMatrix2D(tuple(center), self.angle_x_deg, 1.0)

    def __rotate_image(self, img, M):
        return cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)

    def __rotate_landmarks(self, lms, M):
        _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
        return M.dot(_lms_hom.T).T  # apply transformation


    def calculate_crop_parameters(self, img, lms_orig, img_already_cropped):
        self.__img_shape = img.shape
        lms = lms_orig.copy()

        self.face_center, self.face_scale = self.get_face_center(lms, return_scale=True)

        nonzeros= lms[:,0] > 0

        if self.align_face_orientation:
            eye_l, eye_r = self.__get_eye_coordinates(lms)
            self.M = self.__calc_rotation_matrix(self.face_center, eye_l, eye_r)
            lms = self.__rotate_landmarks(lms, self.M)

        # EYES_ALWAYS_CENTERED = cfg.INPUT_SIZE == 128
        EYES_ALWAYS_CENTERED = False
        if not EYES_ALWAYS_CENTERED:
            cx = (lms[nonzeros,0].min()+lms[nonzeros,0].max())/2
            self.face_center[0] = cx

        crop_by_eye_mouth_dist = False
        if len(lms) == 5:
            crop_by_eye_mouth_dist = True

        if crop_by_eye_mouth_dist:
            self.tl = (self.face_center - self.face_scale).astype(int)
            self.br = (self.face_center + self.face_scale).astype(int)
            # pass
        elif self.loose:
            bb = get_bbox_from_landmarks(lms, loose=True, loose_scale=self.loose_scale)
            self.tl = bb[:2]
            self.br = bb[2:]
        else:
            # calc height
            t = lms[nonzeros, 1].min()
            b = lms[nonzeros, 1].max()
            if t > b:
                t, b = b, t

            h = b - t
            assert(h >= 0)

            if len(lms) != 68 and len(lms) != 21:
                h *= 1.5
                t = t - h/2
                b = b + h/2

            # enlarge a little
            min_row, max_row = int(t - cfg.CROP_MOVE_TOP_FACTOR * h), int(b + cfg.CROP_MOVE_BOTTOM_FACTOR * h)

            # calc width
            s = (max_row - min_row)/2
            min_col, max_col = self.face_center[0] - s, self.face_center[0] + s

            # in case right eye is actually left of right eye...
            if min_col > max_col:
                min_col, max_col = max_col, min_col

            self.tl = np.array((min_col, min_row))
            self.br = np.array((max_col, max_row))

        # extend area by crop border margins
        scale_factor = cfg.CROP_SIZE / cfg.INPUT_SIZE
        bbox = np.concatenate((self.tl, self.br))
        bbox_crop = geometry.scaleBB(bbox, scale_factor, scale_factor, typeBB=2)
        self.tl = bbox_crop[0:2].astype(int)
        self.br = bbox_crop[2:4].astype(int)


    def apply_to_image(self, img=None, with_hist_norm=cfg.WITH_HIST_NORM):
        if img is None:
            img = self.img

        if self.img_already_cropped:
            h, w = img.shape[:2]
            if (w,h) != self.output_size:
                img = cv2.resize(img, self.output_size, interpolation=cv2.INTER_CUBIC)
            return img

        img_padded = mirror_padding(img)

        h,w = img.shape[:2]
        tl_padded = self.tl + (w,h)
        br_padded = self.br + (w,h)

        # extend image in case mirror padded image is still too smal
        dilate = -np.minimum(tl_padded, 0)
        padding = [
            (dilate[1], dilate[1]),
            (dilate[0], dilate[0]),
             (0,0)
        ]
        try:
            img_padded = np.pad(img_padded, padding, 'constant')
        except TypeError:
            plt.imshow(img)
            plt.show()
        tl_padded += dilate
        br_padded += dilate

        if self.align_face_orientation and self.lms is not None:
            # rotate image
            face_center = self.get_face_center(self.lms, return_scale=False)
            M  = cv2.getRotationMatrix2D(tuple(face_center+(w,h)), self.angle_x_deg, 1.0)
            img_padded = self.__rotate_image(img_padded, M)

        crop = img_padded[tl_padded[1]: br_padded[1], tl_padded[0]: br_padded[0]]

        try:
            resized_crop = cv2.resize(crop, self.output_size, interpolation=cv2.INTER_CUBIC)
        except cv2.error:
            print('img size', img.shape)
            print(self.tl)
            print(self.br)
            print('dilate: ', dilate)
            print('padding: ', padding)
            print('img pad size', img_padded.shape)
            print(tl_padded)
            print(br_padded)
            plt.imshow(img_padded)
            plt.show()
            raise

        # image normalization
        if with_hist_norm:
            p2, p98 = np.percentile(crop, (2, 98))
            resized_crop = exposure.rescale_intensity(resized_crop, in_range=(p2, p98))

        # resized_crop = resized_crop.astype(np.float32)
        np.clip(resized_crop, 0, 255)
        return resized_crop


    def center_on_face(self, img=None, landmarks=None):
        raise(NotImplemented)
        if img is None:
            img = self.img

        tl_padded = np.array(img.shape[:2][::-1])
        br_padded = tl_padded * 2
        img_center = tl_padded * 1.5

        face_center = self.get_face_center(landmarks)
        offset = (self.face_center - img_center).astype(int)

        img_new = img
        landmarks_new = landmarks

        if landmarks is not None:
            if self.align_face_orientation and self.lms is not None:
                landmarks_new = self.__rotate_landmarks(landmarks_new+tl_padded, self.M) - tl_padded
            landmarks_new = landmarks_new - offset

        if img is not None:
            img_padded = mirror_padding(img)
            if self.align_face_orientation and self.lms is not None:
                img_padded = self.__rotate_image(img_padded, self.M)
            tl_padded += offset
            br_padded += offset
            img_new = img_padded[tl_padded[1]: br_padded[1], tl_padded[0]: br_padded[0]]

        return img_new, landmarks_new


    def apply_to_landmarks(self, lms_orig, pose=None):
        if lms_orig is None:
            return lms_orig, pose

        if pose is not None:
            pose_new = np.array(pose).copy()
        else:
            pose_new = None

        lms = lms_orig.copy()

        if self.align_face_orientation and self.lms is not None:
            # rotate landmarks
            # if not self.img_already_cropped:
            #     lms[:,0] += self.img.shape[1]
            #     lms[:,1] += self.img.shape[0]
            face_center = self.get_face_center(self.lms, return_scale=False)
            M = cv2.getRotationMatrix2D(tuple(face_center), self.angle_x_deg, 1.0)
            lms = self.__rotate_landmarks(lms, M).astype(np.float32)

            # if not self.img_already_cropped:
            #     lms[:,0] -= self.img.shape[1]
            #     lms[:,1] -= self.img.shape[0]

            if pose_new is not None:
                pose_new[2] = 0.0

        # tl = (self.face_center - self.face_scale).astype(int)
        # br = (self.face_center + self.face_scale).astype(int)

        tl = self.tl
        br = self.br
        crop_width = br[0] - tl[0]
        crop_height = br[1] - tl[1]

        lms_new = lms.copy()
        lms_new[:, 0] = (lms_new[:, 0] - tl[0]) * self.output_size[0] / crop_width
        lms_new[:, 1] = (lms_new[:, 1] - tl[1]) * self.output_size[1] / crop_height

        return lms_new, pose_new

    def apply_to_landmarks_inv(self, lms):
        tl = self.tl
        br = self.br
        crop_width = br[0] - tl[0]
        crop_height = br[1] - tl[1]

        lms_new = lms.copy()
        lms_new[:, 0] = lms_new[:, 0] * (crop_width / self.output_size[0]) + tl[0]
        lms_new[:, 1] = lms_new[:, 1] * (crop_height / self.output_size[1]) + tl[1]

        M = cv2.getRotationMatrix2D(tuple(self.face_center), -self.angle_x_deg, 1.0)
        return self.__rotate_landmarks(lms_new, M)



def crop_face(img, lms_orig, pose, output_size, img_already_cropped=False, crop_by_eye_mouth_dist=False,
              align_face_orientation=True, crop_square=False):

    if pose is not None:
        pose_new = np.array(pose).copy()
    else:
        pose_new = None

    lms = lms_orig.copy()
    if not img_already_cropped:
        lms[:,0] += img.shape[1]
        lms[:,1] += img.shape[0]
        img = mirror_padding(img)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(img)
    # scs = []
    # legendNames = []
    # for i in range(30, 37):
    #     sc = ax.scatter(lms[i, 0], lms[i, 1])
    #     scs.append(sc)
    #     legendNames.append("{}".format(i))
    # plt.legend(scs, legendNames, scatterpoints=1, loc='best', ncol=4, fontsize=7)
    # plt.show()

    if lms.shape[0] == 68:
        # eye_l, eye_r = (lms[36]+lms[39])/2,  (lms[42]+lms[45])/2  # center landmarks
        # mouth_l, mouth_r = lms[48],  lms[54]  # outer landmarks
        id_eye_l, id_eye_r = [36, 39],  [42, 45]
        id_mouth_l, id_mouth_r = 48,  54
        # crop_by_eye_mouth_dist = True
    elif lms.shape[0] == 37:
        id_eye_l, id_eye_r = [13, 16],  [19, 22]
        id_mouth_l, id_mouth_r = 25,  31
        crop_by_eye_mouth_dist = True
    else:
        raise ValueError("Invalid landmark format!")

    eye_l, eye_r = lms[id_eye_l].mean(axis=0),  lms[id_eye_r].mean(axis=0)  # center landmarks
    mouth_l, mouth_r = lms[id_mouth_l],  lms[id_mouth_r]  # outer landmarks

    if eye_r[0] < eye_l[0]:
        eye_r, eye_l = eye_l, eye_l

    eye_c = (eye_l+eye_r)/2
    vec_eye = eye_r - eye_l
    w = np.linalg.norm(vec_eye)
    vec_nose = eye_c - (mouth_l+mouth_r)/2
    # c = eye_c - 0.50*vec_nose
    # s = int(1.40*np.linalg.norm(vec_nose))
    c = eye_c - 0.25*vec_nose
    s = int(1.60*np.linalg.norm(vec_nose))

    if align_face_orientation:
        # calculate rotation matrix M
        vx = vec_eye / np.linalg.norm(w)
        angle_x = np.arcsin(vx[1])
        angle_x_deg = np.rad2deg(angle_x)
        M = cv2.getRotationMatrix2D(tuple(c), angle_x_deg, 1.0)

        # rotate image
        if not img_already_cropped:
            img = cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


        # rotate landmarks
        _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
        lms = M.dot(_lms_hom.T).T  # apply transformation
        # eye_l, eye_r = (lms[36]+lms[39])/2,  (lms[42]+lms[45])/2  # center landmarks
        # mouth_l, mouth_r = lms[48],  lms[54]  # outer landmarks
        eye_l, eye_r = lms[id_eye_l].mean(axis=0), lms[id_eye_r].mean(axis=0)  # center landmarks
        mouth_l, mouth_r = lms[id_mouth_l], lms[id_mouth_r]  # outer landmarks

        if pose_new is not None:
            pose_new[2] = 0.0

    if crop_by_eye_mouth_dist:
        tl = c - s
        br = c + s
    else:
        t = lms[:, 1].min()
        b = lms[:, 1].max()
        if t > b:
            t, b = b, t
        h = b - t
        assert (h >= 0)

        # extended for training unsupervised AAE
        min_row, max_row = int(t - 0.25 * h), int(b + 0.05 * h)

        if crop_square:
            s = (max_row - min_row)/2
            min_col, max_col = c[0] - s, c[0] + s
        else:
            w = max(w, h * 0.5)
            min_col, max_col = int(eye_l[0] - 0.35 * w), int(eye_r[0] + 0.35 * w)

        # in case right eye is actually left of right eye...
        if min_col > max_col:
            min_col, max_col = max_col, min_col

        tl = np.array((min_col, min_row))
        br = np.array((max_col, max_row))

    tl = tl.astype(int)
    br = br.astype(int)

    crop_width = br[0] - tl[0]
    crop_height = br[1] - tl[1]
    lms_new = lms.copy()
    lms_new[:,0] = (lms_new[:,0] - tl[0]) * output_size[0]/crop_width
    lms_new[:,1] = (lms_new[:,1] - tl[1]) * output_size[1]/crop_height

    if img_already_cropped:
        return img, lms_new.astype(np.float32), pose_new

    crop = img[tl[1]:br[1], tl[0]:br[0]]
    resized_crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_CUBIC)

    # image normalization
    if cfg.WITH_HIST_NORM:
        p2, p98 = np.percentile(crop, (2, 98))
        resized_crop = exposure.rescale_intensity(resized_crop, in_range=(p2, p98))

    # resized_crop = resized_crop.astype(np.float32)
    np.clip(resized_crop, 0, 255)

    show_crop = False
    if show_crop:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        im = ax.imshow(img)
        im = ax2.imshow(resized_crop)

        min_col, min_row = tl
        max_col, max_row = br
        ax.plot([min_col, max_col], [min_row, min_row])
        ax.plot([min_col, max_col], [max_row, max_row])
        ax.plot([min_col, min_col], [min_row, max_row])
        ax.plot([max_col, max_col], [min_row, max_row])

        for i in range(0, 68):
            ax.scatter(lms[i, 0], lms[i, 1])
            ax.scatter(c[0], c[1], marker='+', c='k')
            ax2.scatter(lms_new[i, 0], lms_new[i, 1])
        plt.show()

    return resized_crop, lms_new.astype(np.float32), pose_new


def crop_celebHQ(bm, lms_orig, output_size):

    lms = lms_orig.copy()
    lms[:,0] += bm.shape[1]
    lms[:,1] += bm.shape[0]

    bm = mirror_padding(bm)

    if len(lms) == 68:
        eye_l, eye_r = (lms[36]+lms[39])/2,  (lms[42]+lms[45])/2  # center landmarks
        mouth_l, mouth_r = lms[48],  lms[54]  # outer landmarks
    elif len(lms) == 21:
        eye_l, eye_r = lms[6,:].astype(int),  lms[11,:].astype(int)  # outer landmarks
    elif len(lms) == 5:
        eye_l, eye_r = lms[0,:].astype(int),  lms[1,:].astype(int)  # center landmarks
    else:
        raise ValueError('Unknown landmark format!')

    if eye_r[0] < eye_l[0]:
        eye_r, eye_l = eye_l, eye_l

    eye_c = (eye_l+eye_r)/2
    w = eye_r - eye_l
    h = eye_c - (mouth_l+mouth_r)/2
    c = eye_c - 0.25*h
    # s = int(max(4*np.linalg.norm(w), 3.6*np.linalg.norm(h)))
    # s = int(max(1.8*np.linalg.norm(w), 1.6*np.linalg.norm(h)))
    s = int(1.6*np.linalg.norm(h))
    # angle = math.acos((w/np.linalg.norm(w))[0])

    vx = w / np.linalg.norm(w)
    vy = h / np.linalg.norm(h)
    angle_x = np.arcsin(vx[1])
    angle_y = np.arcsin(vy[0])
    angle_x_deg = np.rad2deg(angle_x)
    M = cv2.getRotationMatrix2D(tuple(c), angle_x_deg, 1.0)
    bm_rot = cv2.warpAffine(bm, M, bm.shape[:2], flags=cv2.INTER_CUBIC)

    lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))
    lms_crop = M.dot(lms_hom.T).T

    tl = c - s
    br = c + s

    tl = tl.astype(int)
    br = br.astype(int)

    crop = bm_rot[tl[1]:br[1], tl[0]:br[0]]

    lms_new = lms_crop.copy()
    lms_new[:,0] = (lms_new[:,0] - tl[0]) * output_size[0]/crop.shape[1]
    lms_new[:,1] = (lms_new[:,1] - tl[1]) * output_size[1]/crop.shape[0]

    resized_crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_CUBIC)

    show_crop = False
    if show_crop:
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        im = ax.imshow(bm)
        im = ax2.imshow(bm_rot)
        im = ax3.imshow(resized_crop)

        min_col, min_row = tl
        max_col, max_row = br
        ax2.plot([min_col, max_col], [min_row, min_row])
        ax2.plot([min_col, max_col], [max_row, max_row])
        ax2.plot([min_col, min_col], [min_row, max_row])
        ax2.plot([max_col, max_col], [min_row, max_row])

        for i in range(0, 68):
            ax.scatter(lms[i, 0], lms[i, 1])
            ax2.scatter(lms_crop[i, 0], lms_crop[i, 1])
            ax3.scatter(lms_new[i, 0], lms_new[i, 1])
        plt.show()

    # image normalization
    if cfg.WITH_HIST_NORM:
        p2, p98 = np.percentile(crop, (2, 98))
        resized_crop = exposure.rescale_intensity(resized_crop, in_range=(p2, p98))

    # resized_crop = resized_crop.astype(np.float32)
    np.clip(resized_crop, 0, 255)

    return resized_crop, lms_new.astype(np.float32)


def crop_bump(bm, lms, output_size=(60,80)):
    if len(lms) == 68:
        eye_l, eye_r = lms[36,:].astype(int),  lms[45,:].astype(int)  # outer landmarks
    elif len(lms) == 21:
        eye_l, eye_r = lms[6,:].astype(int),  lms[11,:].astype(int)  # outer landmarks
    elif len(lms) == 5:
        eye_l, eye_r = lms[0,:].astype(int),  lms[1,:].astype(int)  # center landmarks
    else:
        raise ValueError('Unknown landmark format!')

    if eye_r[0] < eye_l[0]:
       eye_r, eye_l = eye_l, eye_l

    w = eye_r[0] - eye_l[0]
    t = lms[:, 1].min()
    b = lms[:, 1].max()
    if t > b:
        t, b = b, t
    h = b - t
    w = max(w, h*0.5)
    assert(h >= 0)

    # extended for training unsupervised AAE
    min_col, max_col = int(eye_l[0]-0.35*w), int(eye_r[0]+0.35*w)
    min_row, max_row = int(t-0.25*h), int(b+0.05*h)

    # in case right eye is actually left of right eye...
    if min_col > max_col:
        min_col, max_col = max_col, min_col

    height = max_row - min_row
    width = max_col - min_col
    bm_cropped = np.zeros((height, width, 3), dtype=bm.dtype)

    min_row_src = max(0, min_row)
    min_col_src = max(0, min_col)
    max_row_src = min(bm.shape[0], max_row)
    max_col_src = min(bm.shape[1], max_col)

    top_dst = min_row_src - min_row
    bottom_dst = top_dst + (max_row_src-min_row_src)
    left_dst = min_col_src - min_col
    right_dst = left_dst + (max_col_src-min_col_src)
    try:
        bm_cropped[top_dst:bottom_dst, left_dst:right_dst] = bm[min_row_src:max_row_src, min_col_src:max_col_src]
    except Exception as e:
        print(min_row, max_row)
        print(min_row_src, max_row_src)
        print(top_dst, bottom_dst)
        print(bm.shape[0])
        raise(e)

    # fill back border with first/last row/column
    if top_dst > 0:
        bm_cropped[:top_dst] = bm_cropped[top_dst]
    if bottom_dst < bm_cropped.shape[0]:
        bm_cropped[bottom_dst:] = bm_cropped[bottom_dst-1]
    for x in range(left_dst):
        bm_cropped[:,x] = bm_cropped[:, left_dst]
    for x in range(right_dst, bm_cropped.shape[1]):
        bm_cropped[:, x] = bm_cropped[:, right_dst-1]

    show_crop = False
    if show_crop:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(bm, interpolation='nearest', cmap=plt.cm.viridis)
        ax.plot([min_col, max_col], [min_row, min_row])
        ax.plot([min_col, max_col], [max_row, max_row])
        ax.plot([min_col, min_col], [min_row, max_row])
        ax.plot([max_col, max_col], [min_row, max_row])
        legendNames = []
        scs = []
        for i in range(17,27):
        # for i in range(len(lms)):
            sc = ax.scatter(lms[i,0], lms[i,1])
            scs.append(sc)
            legendNames.append("{}".format(i))
        plt.legend(scs, legendNames, scatterpoints=1, loc='best', ncol=4, fontsize=7)
        fig.colorbar(im, ax=ax)
        plt.show()

    lms_new = lms.copy()
    lms_new[:,0] = (lms_new[:,0] - min_col) * output_size[0]/bm_cropped.shape[1]
    lms_new[:,1] = (lms_new[:,1] - min_row) * output_size[1]/bm_cropped.shape[0]

    bm_cropped = cv2.resize(bm_cropped, output_size, interpolation=cv2.INTER_CUBIC)

    # image normalization
    if cfg.WITH_HIST_NORM:
        # if cfg.GREY:
        #     bm_cropped = np.minimum(bm_cropped, 1.0)
        # else:
        # bm_cropped = np.minimum(bm_cropped, 255)
        # bm_cropped = exposure.equalize_adapthist(bm_cropped)
        # bm_cropped = exposure.equalize_hist(bm_cropped)
        p2, p98 = np.percentile(bm_cropped, (2, 98))
        bm_cropped = exposure.rescale_intensity(bm_cropped, in_range=(p2, p98))

        # if not cfg.GREY:
        # bm_cropped = (bm_cropped*255).astype(np.uint8)

    return bm_cropped, lms_new


class CenterCrop(object):
    """Like tf.CenterCrop, but works works on numpy arrays instead of PIL images."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __crop_image(self, img):
        t = int((img.shape[0] - self.size[0]) / 2)
        l = int((img.shape[1] - self.size[1]) / 2)
        b = t + self.size[0]
        r = l + self.size[1]
        return img[t:b, l:r]

    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
            if landmarks is not None:
                landmarks[...,0] -= int((img.shape[0] - self.size[0]) / 2)
                landmarks[...,1] -= int((img.shape[1] - self.size[1]) / 2)
                landmarks[landmarks < 0] = 0
            return {'image': self.__crop_image(img), 'landmarks': landmarks, 'pose': pose}
        else:
            return self.__crop_image(sample)


    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)


class RandomRotation(object):
    """Rotate the image by angle.

    Like tf.RandomRotation, but works works on numpy arrays instead of PIL images.

    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])
        return angle


    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        angle = self.get_params(self.degrees)
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = calc_rotation_matrix(center, angle)
        img_rotated = rotate_image(image, M)
        if landmarks is not None:
            landmarks = rotate_landmarks(landmarks, M).astype(np.float32)
            pose_rotated = pose
            pose_rotated[2] -= np.deg2rad(angle).astype(np.float32)
        return {'image': img_rotated, 'landmarks': landmarks, 'pose': pose}

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ')'
        return format_string


def calc_rotation_matrix(center, degrees):
    return cv2.getRotationMatrix2D(tuple(center), degrees, 1.0)


def rotate_image(img, M):
    return cv2.warpAffine(img, M, img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


def rotate_landmarks(lms, M):
    _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
    return M.dot(_lms_hom.T).T  # apply transformation


def transform_image(img, M):
    # r = skimage.transform.AffineTransform(rotation=np.deg2rad(45))
    # im_centered = skimage.transform.warp(img, t._inv_matrix).astype(np.float32)
    # im_trans = skimage.transform.warp(im_centered, M._inv_matrix, order=3).astype(np.float32)
    # tmp = skimage.transform.AffineTransform(matrix=M)

    # return  skimage.transform.warp(img, M._inv_matrix, order=3).astype(np.float32)  # Very slow!
    return cv2.warpAffine(img, M.params[:2], img.shape[:2][::-1], flags=cv2.INTER_CUBIC)


def transform_landmarks(lms, M):
    _lms_hom = np.hstack((lms, np.ones((lms.shape[0], 1))))  # make landmarks homogeneous
    # t = skimage.transform.AffineTransform(translation=-np.array(img.shape[:2][::-1])/2)
    # m = t._inv_matrix.dot(M.params.dot(t.params))
    # return M.params.dot(_lms_hom.T).T[:,:2]
    return M(lms)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (numbers.Number, tuple))
        self.output_size = output_size

    # @staticmethod
    # def crop(image, scale):
    #     s = random.uniform(*scale)
    #     print(s)
    #     h, w = image.shape[:2]
    #     cy, cx = h//2, w//2
    #     h2, w2 = int(cy*s), int(cx*s)
    #     return image[cy-h2:cy+h2, cx-h2:cx+h2]

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        # if random.random() < self.p:
        #     image = self.crop(image, self.scale)

        h, w = image.shape[:2]

        if isinstance(self.output_size, numbers.Number):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # img = F.resize(image, (new_h, new_w))
        img = cv2.resize(image, dsize=(new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if landmarks is not None:
            landmarks = landmarks * [new_w / w, new_h / h]
            landmarks = landmarks.astype(np.float32)

        return {'image': img, 'landmarks': landmarks, 'pose': pose}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        if landmarks is not None:
            landmarks = landmarks - [left, top]
            landmarks = landmarks.astype(np.float32)

        return {'image': image, 'landmarks': landmarks, 'pose': pose}


class RandomResizedCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, p=1.0, scale=(1.0, 1.0), keep_aspect=True):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        self.scale = scale
        self.p = p
        self.keep_aspect = keep_aspect

    def __call__(self, sample):
        image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

        h, w = image.shape[:2]
        s_x = random.uniform(*self.scale)
        if self.keep_aspect:
            s_y = s_x
        else:
            s_y = random.uniform(*self.scale)
        new_w, new_h = int(self.output_size[0] * s_x), int(self.output_size[1] * s_y)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        landmarks = landmarks - [left, top]

        image = cv2.resize(image, dsize=self.output_size)
        landmarks /= [s_x, s_y]

        return {'image': image, 'landmarks': landmarks.astype(np.float32), 'pose': pose.astype(np.float32)}


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    lm_left_to_right_68 = {
        # outline
        0:16,
        1:15,
        2:14,
        3:13,
        4:12,
        5:11,
        6:10,
        7:9,
        8:8,

        #eyebrows
        17:26,
        18:25,
        19:24,
        20:23,
        21:22,

        #nose
        27:27,
        28:28,
        29:29,
        30:30,

        31:35,
        32:34,
        33:33,

        #eyes
        36:45,
        37:44,
        38:43,
        39:42,
        40:47,
        41:46,

        #mouth outer
        48:54,
        49:53,
        50:52,
        51:51,
        57:57,
        58:56,
        59:55,

        #mouth inner
        60:64,
        61:63,
        62:62,
        66:66,
        67:65,
    }

    # AFLW
    lm_left_to_right_21 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:16,
        13:15,
        14:14,
        17:19,
        18:18,
        20:20
    }

    # AFLW without ears
    lm_left_to_right_19 = {
        0:5,
        1:4,
        2:3,
        6:11,
        7:10,
        8:9,

        12:14,
        13:13,
        15:17,
        16:16,
        18:18
    }

    lm_left_to_right_5 = {
        0:1,
        2:2,
        3:4,
    }

    def __init__(self, p=0.5):

        def build_landmark_flip_map(left_to_right):
            map = left_to_right
            right_to_left = {v:k for k,v in map.items()}
            map.update(right_to_left)
            return map

        self.p = p

        self.lm_flip_map_68 = build_landmark_flip_map(self.lm_left_to_right_68)
        self.lm_flip_map_21 = build_landmark_flip_map(self.lm_left_to_right_21)
        self.lm_flip_map_19 = build_landmark_flip_map(self.lm_left_to_right_19)
        self.lm_flip_map_5 = build_landmark_flip_map(self.lm_left_to_right_5)


    def __call__(self, sample):
        if random.random() < self.p:
            if isinstance(sample, dict):
                img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
                # flip image
                flipped_img = np.fliplr(img).copy()
                # flip landmarks
                non_zeros = landmarks[:,0] > 0
                landmarks[non_zeros, 0] *= -1
                landmarks[non_zeros, 0] += img.shape[1]
                landmarks_new = landmarks.copy()
                if len(landmarks) == 21:
                    lm_flip_map = self.lm_flip_map_21
                if len(landmarks) == 19:
                    lm_flip_map = self.lm_flip_map_19
                elif len(landmarks) == 68:
                    lm_flip_map = self.lm_flip_map_68
                elif len(landmarks) == 5:
                    lm_flip_map = self.lm_flip_map_5
                else:
                    raise ValueError('Invalid landmark format.')
                for i in range(len(landmarks)):
                    landmarks_new[i] = landmarks[lm_flip_map[i]]
                # flip pose
                if pose is not None:
                    pose[1] *= -1
                return {'image': flipped_img, 'landmarks': landmarks_new, 'pose': pose}

            return np.fliplr(sample).copy()
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees=0, translate=None, scale=None, shear=None, resample=False, fillcolor=0, keep_aspect=True):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.keep_aspect = keep_aspect

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size, keep_aspect):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])

        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (-np.round(random.uniform(-max_dx, max_dx)),
                            -np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale_x = random.uniform(scale_ranges[0], scale_ranges[1])
            if keep_aspect:
                scale_y = scale_x
            else:
                scale_y = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale_x, scale_y = 1.0, 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        M =  skimage.transform.AffineTransform(
            translation=translations,
            shear=np.deg2rad(shear),
            scale=(scale_x, scale_y),
            rotation=np.deg2rad(angle)
        )
        t = skimage.transform.AffineTransform(translation=-np.array(img_size[::-1])/2)
        return skimage.transform.AffineTransform(matrix=t._inv_matrix.dot(M.params.dot(t.params)))


    def __call__(self, sample):
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
        else:
            img = sample

        M = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.shape[:2], self.keep_aspect)
        img_new = transform_image(img, M)

        if isinstance(sample, dict):
            if landmarks is None:
                landmarks_new = None
            else:
                landmarks_new = transform_landmarks(landmarks, M).astype(np.float32)
            return {'image': img_new, 'landmarks': landmarks_new, 'pose': pose}
        else:
            return img_new


    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)



class RandomLowQuality(object):
    """Reduce image quality by as encoding as low quality jpg.

    Args:
        p (float): probability of the image being recoded. Default value is 0.2
        qmin (float): min jpg quality
        qmax (float): max jpg quality
    """

    def __init__(self, p=0.5, qmin=8, qmax=25):
        self.p = p
        self.qmin = qmin
        self.qmax = qmax

    def _encode(self, img, q):
        return cv2.imencode('.jpg', img, params=[int(cv2.IMWRITE_JPEG_QUALITY), q])

    def _recode(self, img, q):
        return cv2.imdecode(self._encode(img, q)[1], flags=cv2.IMREAD_COLOR)

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be recoded .

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return self._recode(img, random.randint(self.qmin, self.qmax))
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(sample, dict):
            sample['image'] = F.normalize(sample['image'], self.mean, self.std)
        else:
            sample = F.normalize(sample, self.mean, self.std)
        return sample


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        if isinstance(sample, dict):
            image, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            # image = image.transpose((2, 0, 1))
            return {'image': F.to_tensor(image),
                    'landmarks': landmarks,
                    'pose': pose}
        else:
            return F.to_tensor(sample)
            # return torch.from_numpy(sample)


def convert_video_to_image_seqence(vidfile, out_dir, imseq_pattern='%06d', img_ext='png'):
    """ Use FFMPEG to convert a video to a sequence of images. """
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    dst = os.path.join(out_dir, '{}.{}'.format(imseq_pattern, img_ext))
    ffmpegStr = 'ffmpeg -i "{}" -qscale:v 1 "{}" >/dev/null 2>&1'.format(vidfile, dst)
    log.info(ffmpegStr)
    os.system(ffmpegStr)


from utils.io import makedirs
def convert_video_to_wav(vidfile, wav_file):
    """ Use FFMPEG to convert a video to an audio wav file. """
    try:
        makedirs(wav_file)
    except OSError:
        pass
    ffmpegStr = 'ffmpeg -i "{}" "{}" >/dev/null 2>&1'.format(vidfile, wav_file)
    log.info(ffmpegStr)
    os.system(ffmpegStr)

def run_opensmile(wav_file, out_file):
    """ Extract audio features from wav file with OpenSMILE. Creates WEKA arff file."""
    try:
        makedirs(out_file)
    except OSError:
        pass
    OPEN_SMILE_BIN = '/home/browatbn/dev/libs/opensmile-2.3.0/SMILExtract'
    OPEN_SMILE_CONFIG = '/home/browatbn/dev/libs/opensmile-2.3.0/config/emobase2010.conf'
    cmd = '{} -noconsoleoutput 1 -C {} -I "{}" -O "{}" '.format(OPEN_SMILE_BIN,
                                                                         OPEN_SMILE_CONFIG,
                                                                         wav_file,
                                                                         out_file)
    log.info(cmd)
    os.system(cmd)

def cropImgLM(img, tlx, tly, brx, bry, lms, img2, rescale):
    l = float(tlx)
    t = float(tly)
    ww = float(brx - l)
    hh = float(bry - t)
    # Approximate LM tight BB
    h = img.shape[0]
    w = img.shape[1]
    if img2 is not None:
        cv2.rectangle(img2, (int(l), int(t)), (int(brx), int(bry)), (0, 255, 255), 2)
    cx = l + ww / 2
    cy = t + hh / 2
    tsize = max(ww, hh) / 2
    l = cx - tsize
    t = cy - tsize

    # Approximate expanded bounding box
    bl = int(round(cx - rescale[0] * tsize))
    bt = int(round(cy - rescale[1] * tsize))
    br = int(round(cx + rescale[2] * tsize))
    bb = int(round(cy + rescale[3] * tsize))
    nw = int(br - bl)
    nh = int(bb - bt)
    imcrop = np.zeros((nh, nw, 3), dtype="uint8")
    lms0 = lms.copy()
    lms0[:, 0] = lms0[:, 0] - bl
    lms0[:, 1] = lms0[:, 1] - bt

    ll = 0
    if bl < 0:
        ll = -bl
        bl = 0
    rr = nw
    if br > w:
        rr = w + nw - br
        br = w
    tt = 0
    if bt < 0:
        tt = -bt
        bt = 0
    bbb = nh
    if bb > h:
        bbb = h + nh - bb
        bb = h
    imcrop[tt:bbb, ll:rr, :] = img[bt:bb, bl:br, :]
    return imcrop, lms0


rescaleTest = [1.5, 1.5, 1.5, 1.5]
rescaleCASIA = [1.9255, 2.2591, 1.9423, 1.6087]
rescaleBB = [1.785974, 1.951171, 1.835600, 1.670403]


def cropByLM(img, lms, img2=None):
    lms_x = lms[:, 0]
    lms_y = lms[:, 1]
    img_new, lms_new = cropImgLM(img,min(lms_x),min(lms_y),max(lms_x),max(lms_y), lms, img2, rescaleTest)
    return img_new, lms_new


def run_open_face(frame_dir, out_dir, is_sequence):
    """ Extract facial features using Open Face """
    OPEN_FACE_BIN = '/home/browatbn/dev/libs/OpenFace2/build/bin/'
    OPEN_FACE_BIN += 'FeatureExtraction' if is_sequence else 'FaceLandmarkImg'
    open_face_str = '{} -fdir "{}" -out_dir "{}" -2Dfp -pose >/dev/null 2>&1'.format(OPEN_FACE_BIN, frame_dir, out_dir)
    log.info(open_face_str)
    os.system(open_face_str)


def draw_landmarks(img_, landmarks):
    img = img_.copy()
    for lm in landmarks:
        lm_x, lm_y = lm[0], lm[1]
        cv2.circle(img, (int(lm_x), int(lm_y)), 1, (0, 0, 255), -1)
    return img


def get_face_mask(landmarks, size):
    mask = np.zeros(size[:2], dtype=np.uint8)

    hull = np.vstack((landmarks[:17],
                      # np.flipud(landmarks[17:27])
                      landmarks[26:27]-(0,10),
                      landmarks[24:25]-(0,15),
                      landmarks[23:24]-(0,20),
                      landmarks[20:21]-(0,20),
                      landmarks[19:20]-(0,15),
                      landmarks[17:18]-(0,10)
                      ))

    # cut off top part of forehead
    # hull[17, 1] -= 10
    # hull[18, 1] -= 10
    # hull[19, 1] -= 10
    # t = hull[19,1] - 5
    # hull[20, 1] -= 20
    # hull[21, 1] -= 20
    #
    # hull[24, 1] -= 10
    # hull[25, 1] -= 10
    # hull[26, 1] -= 10
    # t = hull[24,1] - 5
    # hull[23, 1] -= 20
    # hull[22, 1] -= 20

    # limit face width
    # hull[:8, 0] = hull[:8, 0].clip(min=landmarks[17,0])
    # hull[9:17, 0] = hull[9:17, 0].clip(max=landmarks[26,0])
    cv2.fillConvexPoly(mask, hull.astype(int), 1)

    # disp = draw_landmarks(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), hull)
    # cv2.imshow('hull', disp)
    # cv2.waitKey()
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.blur(mask, (5,5))
    return mask


def calc_face_bumps(img1, img2, lms1, lms2, H=None):
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if H is None:
        H = cv2.findHomography(lms2, lms1)[0]

    gray2_warp = cv2.warpPerspective(gray2, H, gray2.shape)
    # img2_warp = cv2.warpPerspective(img2, H, img2.shape[:2])

    nonzeros = np.ones_like(gray2, dtype=np.uint8)
    nonzeros = cv2.warpPerspective(nonzeros, H, gray2.shape)
    nonzeros = cv2.erode(nonzeros.astype(np.uint8), np.ones((3,3), np.uint8), iterations=2)

    # nonzeros = cv2.erode((gray2_warp > 0).astype(np.uint8), np.ones((5,5), np.uint8), iterations=2)

    # set non face pixels to zero
    mask = get_face_mask(lms1, gray2.shape)

    if not isinstance(img1, np.ndarray):
        diff = torch.abs(img1.mean(dim=0) - torch.tensor(gray2_warp).cuda())
        diff *= torch.tensor(mask, dtype=torch.float32).cuda()
        diff *= torch.tensor(nonzeros, dtype=torch.float32).cuda()
        diff += torch.tensor(mask) * 0.1
    else:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if gray1.dtype == np.uint8:
            gray1, gray2_warp = gray1.astype(np.float32)/255, gray2_warp.astype(np.float32)/255
        diff_gray = np.abs(gray1-gray2_warp)
        diff_gray *= nonzeros
        diff_gray *= mask
        diff_gray += mask * 0.05
        diff_gray += 0.05

        if False:
            if img1.dtype == np.uint8:
                img1, img2_warp = img1.astype(np.float32)/255, img2_warp.astype(np.float32)/255

            diff = np.abs(img1-img2_warp)
            # diff *= nonzeros
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            diff *= mask
            diff += mask * 0.1
            diff += 0.05

        # show faces with landmarks
        # img1_lms = draw_landmarks(cv2.cvtColor(gray1, cv2.COLOR_GRAY2RGB), lms1)
        # img2_lms = draw_landmarks(cv2.cvtColor(gray2_warp, cv2.COLOR_GRAY2RGB), lms1)
        # cv2.imshow('img1 lms', cv2.cvtColor(img1_lms, cv2.COLOR_BGR2RGB))
        # cv2.imshow('img2 lms', cv2.cvtColor(img2_lms, cv2.COLOR_BGR2RGB))
        # cv2.imshow('gray1', gray1)
        # cv2.imshow('gray2', gray2)
        # cv2.imshow('gray2_warp', gray2_warp)
        # cv2.waitKey()

    # clear top,left,right borders to remove artefacts from warping
    # leave bottom, we need the mouth area
    # margin = cfg.CROP_BORDER
    # diff[:margin,:] = 0
    # diff[-margin:,:] = 0
    # diff[:,:margin] = 0
    # diff[:,-margin:] = 0

    def post_process(diff):
        diff = cv2.blur(diff, (3,3))
        max_diff = 0.25
        diff = diff.clip(max=max_diff)
        return diff**2 / max_diff**2

    if False:
        diff = diff.mean(axis=2)
        diff = post_process(diff)
        plt.imshow(diff)
        plt.figure()

    diff_gray = post_process(diff_gray)

    # plt.imshow(diff_gray)
    # plt.show()
    return diff_gray



class RandomOcclusion(object):
    def __init__(self):
        pass

    def __add_occlusions(self, img):
        bkg_size = cfg.CROP_BORDER//2
        min_occ_size = 30

        cx = random.randint(bkg_size, bkg_size+cfg.INPUT_SIZE)
        cy = random.randint(bkg_size, bkg_size+cfg.INPUT_SIZE)

        w_half = min(img.shape[1]-cx-1, random.randint(min_occ_size, cfg.INPUT_SIZE/2)) // 2
        h_half = min(img.shape[0]-cy-1, random.randint(min_occ_size, cfg.INPUT_SIZE/2)) // 2
        w_half = min(cx, w_half)
        h_half = min(cy, h_half)

        # l = max(0, w_half+1)
        l = 0
        t = random.randint(h_half+1, cfg.INPUT_SIZE)

        r = l+2*w_half
        b = min(img.shape[0]-1, t+2*h_half)

        cutout = img[t:b, l:r]
        dst_shape = (2*h_half, 2*w_half)

        if cutout.shape[:2] != dst_shape:
            try:
                cutout = cv2.resize(cutout, dsize=dst_shape[::-1], interpolation=cv2.INTER_CUBIC)
            except:
                print('resize error', img.shape, dst_shape, cutout.shape[:2], cy, cx, h_half, w_half)

        try:
            cutout = cv2.blur(cutout, ksize=(5,5))
            img[cy-h_half:cy+h_half, cx-w_half:cx+w_half] = cutout
        except:
            print(img.shape, dst_shape, cutout.shape[:2], cy, cx, h_half, w_half)
        # plt.imshow(img)
        # plt.show()
        return img

    def __call__(self, sample):
        # res_dict = {}
        if isinstance(sample, dict):
            img, landmarks, pose = sample['image'], sample['landmarks'], sample['pose']
            return {'image': self.__add_occlusions(img), 'landmarks': landmarks, 'pose': pose}
            # img = sample['image']
            # res_dict.update(sample)
        else:
            return self.__add_occlusions(sample)
            # img = sample
        # res_dict['image_mod'] = self.__add_occlusions(img)
        # return res_dict


    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)

