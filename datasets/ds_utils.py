import os

import numpy as np
import pandas as pd

from torchvision import transforms as tf
from utils.face_processing import RandomLowQuality, RandomHorizontalFlip
import config as cfg
from utils.io import makedirs
import cv2
from skimage import io
from utils import face_processing as fp, face_processing

# To avoid exceptions when loading truncated image files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def save_to_dataframe(preds, annots):
    from datasets.affectnet import CLASS_NAMES as affectnet_classes
    from datasets.emotiw import EmotiW
    def load_rotations_into_dataframe(df, rotation_dir):
        rots = []
        for fname in df['filename']:
            imname = fname.split('/')[1].split('.')[0]
            rots.append(np.loadtxt(os.path.join(rotation_dir, imname + '_rot.txt')).transpose())
        rots = np.vstack(rots)
        rots = np.array(map(np.rad2deg, rots))
        df['pitch'] = rots[:,0]
        df['yaw'] = rots[:,1]
        df['roll'] = rots[:,2]
        return df
    if preds.shape[1] == 8:
        columns=affectnet_classes
    elif preds.shape[1] == 7:
        columns=EmotiW.classes[:7]
    elif preds.shape[1] == 2:
        columns=['valence', 'arousal']
    elif preds.shape[1] == 8+2:
        columns=['valence', 'arousal']+affectnet_classes
    elif preds.shape[1] > 8+2:
        columns=None
    else:
        raise ValueError

    if preds.shape[1] == 1:
        preds = np.repeat(preds, repeats=2, axis=1)

    df = pd.DataFrame(preds, columns=columns)
    try:
        df.insert(0, 'filename', annots['filename'].tolist())
    except KeyError:
        pass

    if preds.shape[1] > 2:
        df['class'] = np.argmax(preds[:, -8:], axis=1)
    try:
        df['gt_class'] =  annots['class'].tolist()
    except KeyError:
        df['gt_class'] =  annots['emotion_plus'].tolist()
    try:
        df['gt_valence'] =  annots['valence'].tolist()
        df['gt_arousal'] =  annots['arousal'].tolist()
    except KeyError:
        pass
    # rotation_dir = os.path.join(dataset.feature_root, '3D')
    # df = load_rotations_into_dataframe(df, rotation_dir)
    return df


def save_results(filepath, preds, dataset):
    df = save_to_dataframe(preds, dataset)
    df.to_csv(filepath, index=False)


def denormalize(tensor):
    # assert(len(tensor.shape[1] == 3)
    if tensor.shape[1] == 3:
        tensor[:, 0] += 0.518
        tensor[:, 1] += 0.418
        tensor[:, 2] += 0.361
    elif tensor.shape[-1] == 3:
        tensor[..., 0] += 0.518
        tensor[..., 1] += 0.418
        tensor[..., 2] += 0.361

def denormalized(tensor):
    # assert(len(tensor.shape[1] == 3)
    if isinstance(tensor, np.ndarray):
        t = tensor.copy()
    else:
        t = tensor.clone()
    denormalize(t)
    return t


def read_openface_detection(lmFilepath, numpy_lmFilepath=None, from_sequence=False, use_cache=True,
                            return_num_faces=False, expected_face_center=None):
    num_faces_in_image = 0
    try:
        if numpy_lmFilepath is not None:
            npfile = numpy_lmFilepath + '.npz'
        else:
            npfile = lmFilepath + '.npz'
        if os.path.isfile(npfile) and use_cache:
            try:
                data = np.load(npfile)
                of_conf, landmarks, pose = [data[arr] for arr in data.files]
                if of_conf > 0:
                    num_faces_in_image = 1
            except:
                print('Could not open file {}'.format(npfile))
                raise
        else:
            if from_sequence:
                lmFilepath = lmFilepath.replace('features', 'features_sequence')
                lmDir, fname = os.path.split(lmFilepath)
                clip_name = os.path.split(lmDir)[1]
                lmFilepath = os.path.join(lmDir, clip_name)
                features = pd.read_csv(lmFilepath + '.csv', skipinitialspace=True)
                frame_num = int(os.path.splitext(fname)[0])
                features = features[features.frame == frame_num]
            else:
                features = pd.read_csv(lmFilepath + '.csv', skipinitialspace=True)
            features.sort_values('confidence', ascending=False, inplace=True)
            selected_face_id = 0
            num_faces_in_image = len(features)
            if num_faces_in_image > 1 and expected_face_center is not None:
                max_face_size = 0
                min_distance = 1000
                for fid in range(len(features)):
                    face = features.iloc[fid]
                    # if face.confidence < 0.2:
                    #     continue
                    landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
                    landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])

                    landmarks = np.vstack((landmarks_x, landmarks_y)).T
                    face_center = landmarks.mean(axis=0)
                    distance = ((face_center - expected_face_center)**2).sum()**0.5
                    if distance < min_distance:
                        min_distance = distance
                        selected_face_id = fid

                # print("Warning: {} faces in image {}!".format(len(features), lmFilepath))
                # cv2.imshow('read_openface_detection', cv2.imread(lmFilepath.replace('features', 'crops/tight')+'.jpg'))
                # cv2.waitKey()
                    # width = landmarks_x.max() - landmarks_x.min()
                    # height = landmarks_y.max() - landmarks_y.min()
                    # face_size = np.sqrt(height**2 + width**2)
                    # if face_size > max_face_size:
                    #     max_face_size = face_size
                    #     selected_face_id = fid

            # if num_faces_in_image > 1:
            #     min_dist = 125
            #     for fid in range(len(features)):
            #         face = features.iloc[fid]
            #         landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
            #         landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])
            #         landmarks = np.vstack((landmarks_x, landmarks_y)).T
            #         face_center = landmarks.mean(axis=0)
            #         image_center = [125,125]
            #         dist_image_center = ((face_center - image_center)**2).sum()**0.5
            #         if dist_image_center < min_dist:
            #             min_dist = dist_image_center
            #             selected_face_id = fid

            try:
                face = features.iloc[selected_face_id]
            except KeyError:
                face = features
            of_conf = face.confidence
            landmarks_x = face.as_matrix(columns=['x_{}'.format(i) for i in range(68)])
            landmarks_y = face.as_matrix(columns=['y_{}'.format(i) for i in range(68)])
            landmarks = np.vstack((landmarks_x, landmarks_y)).T
            pitch = face.pose_Rx
            yaw = face.pose_Ry
            roll = face.pose_Rz
            pose = np.array((pitch, yaw, roll), dtype=np.float32)
            if numpy_lmFilepath is not None:
                makedirs(npfile)
            np.savez(npfile, of_conf, landmarks, pose)
    except IOError as e:
        # raise IOError("\tError: Could not load landmarks from file {}!".format(lmFilepath))
        # pass
        # print(e)
        of_conf = 0
        landmarks = np.zeros((68,2), dtype=np.float32)
        pose = np.zeros(3, dtype=np.float32)

    result = [of_conf, landmarks.astype(np.float32), pose]
    if return_num_faces:
        result += [num_faces_in_image]
    return result


def read_300W_detection(lmFilepath):
    lms = []
    with open(lmFilepath) as f:
        for line in f:
            try:
                x,y = [float(e) for e in line.split()]
                lms.append((x, y))
            except:
                pass
    assert(len(lms) == 68)
    landmarks = np.vstack(lms)
    return landmarks


def build_transform(deterministic, color, daug=0):

    transforms = []

    if not deterministic:
        transforms = [
            RandomLowQuality(),
            RandomHorizontalFlip(),
            tf.ToPILImage(),
            tf.ColorJitter(brightness=0.2, contrast=0.2),
            # tf.RandomRotation(10, resample=PIL.Image.BICUBIC),
            tf.RandomResizedCrop(cfg.CROP_SIZE, scale=(0.95, 1.0)),
            tf.RandomCrop(cfg.INPUT_SIZE)
        ]
        if color:
            transforms += [tf.RandomGrayscale(0.1)]
        transforms = [fp.RandomHorizontalFlip(0.5)]
        if daug == 1:
            transforms += [fp.RandomAffine(3, translate=[0.025,0.025], scale=[0.975, 1.025], shear=0, keep_aspect=False)]
        elif daug == 2:
            transforms += [fp.RandomAffine(3, translate=[0.035,0.035], scale=[0.970, 1.030], shear=2, keep_aspect=False)]
        elif daug == 3:
            transforms += [fp.RandomAffine(20, translate=[0.035,0.035], scale=[0.970, 1.030], shear=5, keep_aspect=False)]
        elif daug == 4: # for roation invariance
            # transforms += [fp.RandomAffine(degrees=45, translate=[0.030,0.030], scale=[0.97, 1.03], shear=0, keep_aspect=False)]
            # transforms += [fp.RandomRotation(degrees=30)]
            transforms += [fp.RandomAffine(45, translate=[0.035,0.035], scale=[0.940, 1.030], shear=5, keep_aspect=False)]
        elif daug == 5: # for AFLW
            transforms += [fp.RandomAffine(60, translate=[0.035,0.035], scale=[0.940, 1.030], shear=5, keep_aspect=False)]
        elif daug == 6: # for LM CNN
            transforms += [fp.RandomAffine(0, translate=[0.035,0.035], scale=[0.940, 1.030], shear=0, keep_aspect=False)]
        elif daug == 7: # for CFPW profiles (shift left/right)
            transforms += [fp.RandomAffine(10, translate=[0.05,0.035], scale=[0.940, 1.000], shear=0, keep_aspect=False)]

    # transforms = [fp.CenterCrop(cfg.INPUT_SIZE)]
    # transforms += [fp.ToTensor() ]
    # transforms += [ fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1])  # VGGFace(2) ]
    return tf.Compose(transforms)


def build_coarse_lmdet_transform(deterministic):
    if deterministic:
        transforms = [
            # fp.Rescale(cfg.INPUT_SIZE*1.2),
            # fp.RandomAffine(35, translate=[0.2,0.2]),
            # fp.RandomAffine(shear=20),
            # fp.RandomResizedCrop(cfg.INPUT_SIZE, p=1.0, scale=(0.4,1.0), keep_aspect=False),
            # fp.RandomAffine(0, shear=0.5),
            # fp.RandomAffine(40, translate=[0.15,0.15], scale=[0.70, 2.25], shear=15, keep_aspect=False),
            # fp.RandomAffine(0, translate=[0.,0.], scale=[1.20, 1.20], shear=0, keep_aspect=True),
            fp.CenterCrop(cfg.INPUT_SIZE)
        ]
    else:
        transforms = [
            fp.RandomHorizontalFlip(0.5),
            fp.RandomAffine(40, translate=[0.15,0.15], scale=[0.70, 2.25], shear=15, keep_aspect=False),
            # # fp.Rescale(cfg.INPUT_SIZE*1.1),
            # fp.RandomRotation(35),
            # fp.RandomResizedCrop(cfg.INPUT_SIZE, p=1.0, scale=(0.65,1.0)),
            fp.CenterCrop(cfg.INPUT_SIZE)
        ]

    transforms += [fp.ToTensor(),
                   fp.Normalize([0.518, 0.418, 0.361], [1, 1, 1]),  # VGGFace(2)
                   ]
    return tf.Compose(transforms)


def get_face(filename, fullsize_img_dir, cropped_img_dir, landmarks, pose=None, bb=None, size=(cfg.CROP_SIZE, cfg.CROP_SIZE),
             use_cache=True, cropper=None):
    filename_noext = os.path.splitext(filename)[0]
    crop_filepath = os.path.join(cropped_img_dir, filename_noext + '.jpg')
    is_cached_crop = False
    if use_cache and os.path.isfile(crop_filepath):
        try:
            img = io.imread(crop_filepath)
        except:
            raise IOError("\tError: Could not cropped image {}!".format(crop_filepath))
        if img.shape[:2] != size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
        is_cached_crop = True
    else:
        # Load image from dataset
        img_path = os.path.join(fullsize_img_dir, filename)
        try:
            img = io.imread(img_path)
        except:
            raise IOError("\tError: Could not load image {}!".format(img_path))
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        assert(img.shape[2] == 3)

    if (landmarks is None or not landmarks.any()) and not 'crops_celeba' in cropped_img_dir:
        # if 'crops_celeba' in cropped_img_dir and not is_cached_crop:
        #     crop = face_processing.crop_celeba(img, size)
        # else:
            assert(bb is not None)
            # Fall back to bounding box if no landmarks found
            # print('falling back to bounding box')
            crop = face_processing.crop_by_bb(img, face_processing.scale_bb(bb, f=1.075), size=size)
    else:

        if 'crops_celeba' in cropped_img_dir:
            if is_cached_crop:
                crop = img
            else:
                crop = face_processing.crop_celeba(img, size)
        else:
            # try:

                # cropper.calculate_crop_parameters(img, landmarks, img_already_cropped=is_cached_crop)
                # crop = cropper.apply_crop_to_image(img)
                # landmarks, pose = cropper.apply_crop_to_landmarks(landmarks, pose)

                crop, landmarks, pose = face_processing.crop_face(img,
                                                                  landmarks,
                                                                  img_already_cropped=is_cached_crop,
                                                                  pose=pose,
                                                                  output_size=size,
                                                                  crop_by_eye_mouth_dist=cfg.CROP_BY_EYE_MOUTH_DIST,
                                                                  align_face_orientation=cfg.CROP_ALIGN_ROTATION,
                                                                  crop_square=cfg.CROP_SQUARE)
            # except:
            #     print(filename)
            #     print(landmarks)
            #     crop = img

    if use_cache and not is_cached_crop:
        makedirs(crop_filepath)
        io.imsave(crop_filepath, crop)

    return crop, landmarks, pose