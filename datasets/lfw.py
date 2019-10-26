import os
import cv2
import numpy as np
from skimage import io

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils

from constants import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class LFWImages(td.Dataset):

    def __init__(self, root_dir=cfg.LFW_ROOT, train=True, start=None,
                 max_samples=None, deterministic=True, use_cache=True):

        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        self.use_cache = use_cache
        self.root_dir = root_dir
        self.cropped_img_dir = os.path.join(root_dir, 'crops_tight')
        self.fullsize_img_dir = os.path.join(root_dir, 'images')
        self.feature_dir = os.path.join(root_dir, 'features')

        import glob

        ann = []
        person_dirs = sorted(glob.glob(os.path.join(cfg.LFW_ROOT, 'images', '*')))
        for id, person_dir in enumerate(person_dirs):
            name = os.path.split(person_dir)[1]
            for img_file in sorted(glob.glob(os.path.join(person_dir, '*.jpg'))):
                # create fnames of format 'Aaron_Eckhart/Aaron_Eckhart_0001'
                fname = os.path.join(name, os.path.splitext(os.path.split(img_file)[1])[0])
                ann.append({'fname': fname, 'id': id, 'name': name})

        self.annotations = pd.DataFrame(ann)

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.annotations = self.annotations[st:nd]

        self.transform = ds_utils.build_transform(deterministic=True, color=True)

    @property
    def labels(self):
        return self.annotations.id.values

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        filename = sample.fname
        id = sample.id
        of_conf, landmarks, pose = ds_utils.read_openface_detection(os.path.join(self.feature_dir, filename))
        # if of_conf < 0.8:
        #     return self.__getitem__((idx+1) % len(self))

        try:
            # crop, landmarks, pose = ds_utils.get_face(filename+'.jpg', self.fullsize_img_dir, self.cropped_img_dir,
            #                                           landmarks, pose, use_cache=False)
            crop, landmarks, pose, cropper = self.face_extractor.get_face(filename + '.jpg', self.fullsize_img_dir,
                                                                          self.cropped_img_dir, landmarks=landmarks,
                                                                          pose=pose, use_cache=self.use_cache,
                                                                          detect_face=False, crop_type='tight',
                                                                          aligned=True)
        except:
            print(filename)
            return self.__getitem__((idx+1) % len(self))

        # vis.show_landmarks(crop, landmarks, pose=pose, title='lms', wait=10, color=(0,0,255))

        transformed_crop = self.transform(crop)

        landmarks[..., 0] -= int((crop.shape[0] - transformed_crop.shape[1]) / 2)
        landmarks[..., 1] -= int((crop.shape[1] - transformed_crop.shape[2]) / 2)

        item = {
            'image': transformed_crop,
            'id': id,
            'fnames': filename,
            'pose': pose,
            'landmarks': landmarks,
        }
        return item



class LFW(td.Dataset):

    def __init__(self, root_dir=cfg.LFW_ROOT, train=True, start=None,
                 max_samples=None, deterministic=True, use_cache=True, view=2):

        assert(view in [1,2])

        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        self.mode = TRAIN if train else VAL

        self.use_cache = use_cache
        self.root_dir = root_dir
        self.cropped_img_dir = os.path.join(root_dir, 'crops_tight')
        self.fullsize_img_dir = os.path.join(root_dir, 'images')
        self.feature_dir = os.path.join(root_dir, 'features')

        self.default_bbox = [65, 80, 65+100, 80+100]

        pairs_file = 'pairsDevTest.txt' if view == 1 else 'pairs.txt'
        path_annotations = os.path.join(self.root_dir, pairs_file)
        # self.annotations = pd.read_csv(path_annotations)
        self.pairs = []
        with open(path_annotations) as txt_file:
            # num_pairs = int(txt_file.readline()) * 2
            # print(num_pairs)
            for line in txt_file:
                items = line.split()
                if len(items) == 3:
                    pair = (items[0], int(items[1]), items[0], int(items[2]))
                elif len(items) == 4:
                    pair = (items[0], int(items[1]), items[2], int(items[3]))
                else:
                    # print("Invalid line: {}".format(line))
                    continue
                self.pairs.append(pair)

        # assert(num_pairs == len(self.pairs))

        from sklearn.utils import shuffle
        self.pairs = shuffle(self.pairs, random_state=0)

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.pairs = self.pairs[st:nd]

        self.transform = ds_utils.build_transform(deterministic, color=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]  # name1, img1, name2, img2

        filepattern = '{}/{}_{:04d}'

        fname1 = filepattern.format(pair[0], pair[0], pair[1])
        fname2 = filepattern.format(pair[2], pair[2], pair[3])

        of_conf1, landmarks, pose = ds_utils.read_openface_detection(os.path.join(self.feature_dir, fname1),
                                                                     expected_face_center=[125,125])

        # if of_conf < 0.025:
        #     landmarks = None
        #     pose = None
        bb = self.default_bbox


        crop, landmarks, pose, cropper = self.face_extractor.get_face(fname1 + '.jpg', self.fullsize_img_dir,
                                                                      self.cropped_img_dir, landmarks=landmarks,
                                                                      pose=pose, use_cache=self.use_cache,
                                                                      bb=bb,
                                                                      detect_face=False, crop_type='tight',
                                                                      aligned=True)

        transformed_crop1 = self.transform(crop)

        of_conf2, landmarks, pose = ds_utils.read_openface_detection(os.path.join(self.feature_dir, fname2),
                                                                     expected_face_center=[125,125])
        # if of_conf < 0.025:
        #     landmarks = None
        #     pose = None
        crop, landmarks, pose, cropper = self.face_extractor.get_face(fname2 + '.jpg', self.fullsize_img_dir,
                                                                      self.cropped_img_dir, landmarks=landmarks,
                                                                      pose=pose, use_cache=self.use_cache,
                                                                      bb=bb,
                                                                      detect_face=False, crop_type='tight',
                                                                      aligned=True)
            # import matplotlib.pyplot as plt
            # plt.imshow(crop)
            # plt.show()
        transformed_crop2 = self.transform(crop)

        return transformed_crop1, transformed_crop2, pair[0], pair[2], pair[0]==pair[2], float(of_conf1), float(of_conf2)


    def get_face(self, filename, bb, landmarks=None, size=(cfg.CROP_SIZE, cfg.CROP_SIZE)):
        # Load image from dataset
        img_path = os.path.join(self.fullsize_img_dir, filename + '.jpg')
        img = io.imread(img_path)
        if img is None:
            raise IOError("\tError: Could not load image {}!".format(img_path))

        #
        # Crop face using landmarks or bounding box
        #

        def crop_by_bb(img, bb):
            x, y, w, h = bb
            x, y, x2, y2 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
            return img[y:y2, x:x2]

        def crop_by_lm(img, landmarks):
            return face_processing.crop_bump(img, landmarks, output_size=size)

        # print(filename)

        # load landmarks extracted with OpenFace2
        pose = np.zeros(3, dtype=np.float32)
        lmFilepath = os.path.join(self.feature_dir, filename + '.csv')
        try:
            features = pd.read_csv(lmFilepath, skipinitialspace=True)
            features.sort_values('confidence', ascending=False)
            if features.confidence[0] > 0.0:
                landmarks_x = features.as_matrix(columns=['x_{}'.format(i) for i in range(68)])[0]
                landmarks_y = features.as_matrix(columns=['y_{}'.format(i) for i in range(68)])[0]
                landmarks = np.vstack((landmarks_x, landmarks_y)).T
                pitch = features.pose_Rx.values[0]
                yaw = features.pose_Ry.values[0]
                roll = features.pose_Rz.values[0]
                pose = np.array((pitch, yaw, roll), dtype=np.float32)
        except IOError:
            # raise IOError("\tError: Could not load landmarks from file {}!".format(lmFilepath))
            print("\tError: Could not load landmarks from file {}!".format(lmFilepath))

        if landmarks is not None:
            crop, landmarks = crop_by_lm(img, landmarks)
        else:
            crop, landmarks = crop_by_bb(img, bb), np.zeros((68, 2), dtype=np.float32)

        return cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC), pose, landmarks


def extract_features(st=None, nd=None):
    """ Extract facial features (landmarks, pose,...) from images """
    import glob
    from utils import visionLogging as log
    person_dirs = sorted(glob.glob(os.path.join(cfg.LFW_ROOT, 'images', '*')))[st:nd]
    for cnt, person_dir in enumerate(person_dirs):
        log.info("{}[{},{}]".format(st+cnt, st, nd))
        out_dir = person_dir.replace('images', 'features')
        face_processing.run_open_face(person_dir, out_dir, is_sequence=False)


def extract_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--st', default=0, type=int)
    parser.add_argument('--nd', default=1, type=int)
    args = parser.parse_args()

    extract_features(st=args.st, nd=args.nd)



if __name__ == '__main__':
    # extract_main()
    # exit()

    import torch
    from utils import vis, face_processing
    from utils.nn import Batch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ds = LFWImages(train=True, deterministic=True, max_samples=40, use_cache=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True, num_workers=1)

    data = next(iter(dl))
    inputs = Batch(data).images

    ds_utils.denormalize(inputs)
    # imgs = vis.add_id_to_images(inputs.numpy(), data[1].numpy())
    # imgs = vis.add_pose_to_images(inputs.numpy(), pose.numpy())
    # imgs = vis.add_landmarks_to_images(inputs.numpy(), landmarks.numpy())
    imgs = inputs.detach().cpu().numpy()
    vis.vis_square(imgs, fx=1.0, fy=1.0, normalize=False)