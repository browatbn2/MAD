import os
import numpy as np

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils
from utils import vis, face_processing

from constants import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CelebA(td.Dataset):

    def __init__(self, root_dir=cfg.CELEBA_ROOT, train=True, color=True, start=None,
                 max_samples=None, deterministic=None, crop_type='tight', **kwargs):

        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        self.mode = TRAIN if train else TEST

        self.crop_type = crop_type
        self.root_dir = root_dir
        root_dir_local = cfg.CELEBA_ROOT_LOCAL
        assert(crop_type in ['tight', 'loose', 'fullsize'])
        self.cropped_img_dir = os.path.join(root_dir_local, 'crops')
        self.fullsize_img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.feature_dir = os.path.join(root_dir_local, 'features')
        self.color = color
        annotation_filename = 'list_landmarks_align_celeba.txt'

        path_annotations_mod = os.path.join(root_dir_local, annotation_filename + '.mod.pkl')
        if os.path.isfile(path_annotations_mod):
            self.annotations = pd.read_pickle(path_annotations_mod)
        else:
            print('Reading original TXT file...')
            self.annotations = pd.read_csv(os.path.join(self.root_dir, 'Anno', annotation_filename), delim_whitespace=True)
            print('done.')

            # store OpenFace features in annotation dataframe
            poses = []
            confs = []
            landmarks = []
            for cnt, filename in enumerate(self.annotations.fname):
                if cnt % 1000 == 0:
                    print(cnt)
                filename_noext = os.path.splitext(filename)[0]
                conf, lms, pose = ds_utils.read_openface_detection(os.path.join(self.feature_dir, filename_noext))
                poses.append(pose)
                confs.append(conf)
                landmarks.append(lms)
            self.annotations['pose'] = poses
            self.annotations['conf'] = confs
            self.annotations['landmarks_of'] = landmarks

            # add identities to annotations
            self.identities =  pd.read_csv(os.path.join(self.root_dir, 'Anno', 'identity_CelebA.txt'),
                                           delim_whitespace=True, header=None, names=['fname', 'id'])
            self.annotations = pd.merge(self.annotations, self.identities, on='fname', copy=False)

            # save annations as pickle file
            self.annotations.to_pickle(path_annotations_mod)

        # select training or test set (currently not using validation set)
        SPLIT = {TRAIN: (0, 162772),
                 VAL: (162772, 182639),
                 TEST: (182639, 202601)}
        self.annotations = self.annotations[(self.annotations.index >= SPLIT[self.mode][0]) & (self.annotations.index < SPLIT[self.mode][1])]

        self.annotations = self.annotations.sort_values(by='id')

        print("Num. faces: {}".format(len(self.annotations)))
        if 'crops_celeba' in self.cropped_img_dir:
            min_of_conf = 0.0
        else:
            min_of_conf = 0.5
        print("Removing faces with conf < {}".format(min_of_conf))
        self.annotations = self.annotations[self.annotations.conf >= min_of_conf]
        print("Remaining num. faces: {}".format(len(self.annotations)))

        # max_rot_deg = 1
        # print('Limiting rotation to +-{} degrees...'.format(max_rot_deg))
        # poses = np.abs(np.stack(self.annotations.pose.values))
        # self.annotations = self.annotations[(poses[:, 0] > np.deg2rad(max_rot_deg)) |
        #                                     (poses[:, 1] > np.deg2rad(max_rot_deg)) |
        #                                     (poses[:, 2] > np.deg2rad(max_rot_deg))]
        # print(len(self.annotations))

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.annotations = self.annotations[st:nd]
        self._annotations = self.annotations[st:nd].copy()

        if deterministic is None:
            deterministic = self.mode != TRAIN
        self.transform = ds_utils.build_transform(deterministic, self.color)

    @property
    def labels(self):
        return self.annotations.id.values

    def _stats_repr(self):
        fmt_str =  "    Number of identities: {}\n".format(self.annotations.id.nunique())
        return fmt_str

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.mode)
        fmt_str += '    Root Location: {}\n'.format(self.root_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        fmt_str += self._stats_repr()
        return fmt_str

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        filename = sample.fname
        landmarks = sample.landmarks_of
        pose = sample.pose
        id = sample.id

        # crop, landmarks, pose = ds_utils.get_face(filename, self.fullsize_img_dir, self.cropped_img_dir,
        #                                           landmarks, pose, use_cache=True)
        crop, landmarks, pose, cropper = self.face_extractor.get_face(filename, self.fullsize_img_dir,
                                                                      self.cropped_img_dir, landmarks=landmarks,
                                                                      pose=pose, use_cache=True,
                                                                      detect_face=False, crop_type=self.crop_type,
                                                                      aligned=True)

        cropped_sample = {'image': crop, 'landmarks': landmarks, 'pose': pose}
        item = self.transform(cropped_sample)

        # face_mask = face_processing.get_face_mask(item['landmarks'], crop.shape)
        # transformed_face_mask = face_processing.CenterCrop(cfg.INPUT_SIZE)(face_mask)

        em_val_ar = np.array([[-1,0,0]], dtype=np.float32)

        item.update({
            'id': id,
            'fnames': filename,
            # 'face_mask': transformed_face_mask,
            'expression': em_val_ar
        })
        return item

    def get_face(self, filename, size=(cfg.CROP_SIZE, cfg.CROP_SIZE), use_cache=True):
        print(filename)
        sample = self._annotations.loc[self._annotations.fname == filename].iloc[0]
        landmarks = sample.landmarks_of.astype(np.float32)
        pose = sample.pose

        # if OpenFace didn't detect a face, fall back to AffectNet landmarks
        # if sample.conf < 0.1:
        #     landmarks = self.parse_landmarks(sample.facial_landmarks)

        crop, landmarks, pose = ds_utils.get_face(filename, self.fullsize_img_dir, self.cropped_img_dir,
                                                  landmarks, pose, use_cache=use_cache, size=size)
        return crop, landmarks, pose


def extract_features():
    """ Extract facial features (landmarks, pose,...) from images """
    person_dir = os.path.join(cfg.CELEBA_ROOT, 'img_align_celeba')
    out_dir = os.path.join(cfg.CELEBA_ROOT_LOCAL, 'features')
    face_processing.run_open_face(person_dir, out_dir, is_sequence=False)


if __name__ == '__main__':
    # extract_features()
    import torch
    from utils.nn import Batch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    ds = CelebA(train=True, start=0)
    print(ds)
    dl = td.DataLoader(ds, batch_size=10, shuffle=True, num_workers=0)

    for data in dl:
        # data = next(iter(dl))
        batch = Batch(data, gpu=False)
        # inputs = batch.images.clone()
        inputs = batch.images.clone()
        ds_utils.denormalize(inputs)
        imgs = vis.add_landmarks_to_images(inputs.numpy(), batch.landmarks.numpy())
        imgs = vis.add_pose_to_images(imgs, batch.poses.numpy())
        imgs = vis.add_id_to_images(imgs, batch.ids.numpy())
        # imgs = vis.add_emotion_to_images(imgs, batch.emotions.numpy())
        vis.vis_square(imgs, nCols=10, fx=1.0, fy=1.0, normalize=False)