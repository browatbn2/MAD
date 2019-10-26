import os
import time
import cv2
import numpy as np
from skimage import io

import torch.utils.data as td
import pandas as pd

import config as cfg
from datasets import ds_utils
import utils.io
from utils import log

import cv2 as cv

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    ang = 1
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*30, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def encode_landmarks(landmarks):
    return ';'.join(landmarks.ravel().astype(str))

class VoxCeleb(td.Dataset):

    def __init__(self, root_dir=cfg.VOXCELEB_ROOT, train=True, start=None,
                 max_samples=None, deterministic=True, with_bumps=False,
                 min_of_conf=0.3, min_face_height=100, use_cache=True, **kwargs):

        from utils.face_extractor import FaceExtractor
        self.face_extractor = FaceExtractor()

        self.use_cache = use_cache
        self.root_dir = root_dir
        self.cropped_img_dir = os.path.join(cfg.VOXCELEB_ROOT_LOCAL, 'crops')
        self.fullsize_img_dir = os.path.join(root_dir, 'frames/unzippedIntervalFaces/data')
        self.feature_dir = os.path.join(root_dir, 'features/unzippedIntervalFaces/data')
        self.npfeature_dir = os.path.join(cfg.VOXCELEB_ROOT_LOCAL, 'features/unzippedIntervalFaces/data')
        self.train = train
        self.with_bumps = with_bumps

        annotation_filename = 'dev' if train else 'test'
        path_annotations_mod = os.path.join(root_dir, annotation_filename + '.mod.pkl')
        if os.path.isfile(path_annotations_mod) and False:
            self.annotations = pd.read_pickle(path_annotations_mod)
        else:
            print('Reading CSV file...')
            self.annotations = pd.read_csv(os.path.join(root_dir, annotation_filename+'.csv'))
            print('done.')

            # self.annotations['of_conf'] = -1
            # self.annotations['landmarks'] = ''
            # self.annotations['pose'] = ''
            # of_confs, poses, landmarks = [], [], []
            #
            #
            #
            # # for cnt, filename in enumerate(self.annotations.fname):
            # for cnt, idx in enumerate(self.annotations.index):
            #     filename = self.annotations.iloc[idx].fname
            #     filename_noext = os.path.splitext(filename)[0]
            #     of_conf, lms, pose = ds_utils.read_openface_detection(os.path.join(self.feature_dir, filename_noext))
            #     str_landmarks = encode_landmarks(lms)
            #     of_confs.append(of_conf)
            #     # poses.append(pose)
            #     landmarks.append(lms)
            #     self.annotations.loc[idx, 'of_conf'] = of_conf
            #     self.annotations.loc[idx, 'landmarks'] = str_landmarks
            #     self.annotations.loc[idx, 'pose'] = encode_landmarks(pose)
            #     if (cnt+1) % 100 == 0:
            #         print(cnt+1)
            #     if (cnt+1) % 1000 == 0:
            #         print('saving annotations...')
            #         self.annotations.to_pickle(path_annotations_mod)
            # # self.annotations.to_csv(path_annotations_mod, index=False)
            # self.annotations.to_pickle(path_annotations_mod)

        path_annotations_mod = os.path.join(root_dir, annotation_filename + '.lms.pkl')
        lm_annots = pd.read_pickle(os.path.join(root_dir, path_annotations_mod))

        t = time.time()
        self.annotations = pd.merge(self.annotations, lm_annots, on='fname', how='inner')
        print("Time merge: {:.2f}".format(time.time()-t))

        t = time.time()
        self.annotations['vid'] = self.annotations.fname.map(lambda x: x.split('/')[2])
        self.annotations['id'] = self.annotations.uid.map(lambda x: int(x[2:]))
        print("Time vid/id labels: {:.2f}".format(time.time()-t))

        print("Num. faces: {}".format(len(self.annotations)))
        print("Num. ids  : {}".format(self.annotations.id.nunique()))

        # drop bad face detections
        print("Removing faces with conf < {}".format(min_of_conf))
        self.annotations = self.annotations[self.annotations.of_conf >= min_of_conf]
        print("Num. faces: {}".format(len(self.annotations)))

        # drop small faces
        print("Removing faces with height < {}px".format(min_face_height))
        self.annotations = self.annotations[self.annotations.face_height >= min_face_height]
        print("Num. faces: {}".format(len(self.annotations)))


        fr = 0
        prev_vid = -1
        frame_nums = []
        for n, id in enumerate(self.annotations.vid.values):
            fr += 1
            if id != prev_vid:
                prev_vid = id
                fr = 0
            frame_nums.append(fr)
        self.annotations['FRAME'] = frame_nums

        self.max_frames_per_video = 200
        self.frame_interval = 3
        print('Limiting videos in VoxCeleb to {} frames...'.format(self.max_frames_per_video))
        self.annotations = self.annotations[self.annotations.FRAME % self.frame_interval == 0]
        self.annotations = self.annotations[self.annotations.FRAME < self.max_frames_per_video * self.frame_interval]
        print("Num. faces: {}".format(len(self.annotations)))

        # limit number of samples
        st,nd = 0, None
        if start is not None:
            st = start
        if max_samples is not None:
            nd = st+max_samples
        self.annotations = self.annotations[st:nd]

        self.transform = ds_utils.build_transform(deterministic=True, color=True)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Train: {}\n'.format(self.train)
        fmt_str += '    Root Location: {}\n'.format(self.root_dir)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        fmt_str += self._stats_repr()
        return fmt_str

    def _stats_repr(self):
        fmt_str =  "     Number of identities: {}\n".format(self.annotations.id.nunique())
        fmt_str += "     Number of videos:     {}\n".format(self.annotations.vid.nunique())
        fmt_str += "     Frame inverval:       {}\n".format(self.frame_interval)
        fmt_str += "     Max frames per vid:   {}\n".format(self.max_frames_per_video)
        return fmt_str

    @property
    def labels(self):
        return self.annotations.id.values

    def get_landmarks(self, sample):
        landmarks = np.array([sample.landmarks_x, sample.landmarks_y], dtype=np.float32).T
        # return face_processing.scale_landmarks_to_crop(landmarks, output_size=(cfg.CROP_SIZE, cfg.CROP_SIZE))
        return landmarks

    @property
    def vids(self):
        return self.annotations.vid.values

    def show_landmarks(self, img, landmarks, title='landmarks'):
        for lm in landmarks:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 2, (0, 0, 255), -1)
        cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations.iloc[idx]
        filename, id, vid  = sample.fname, sample.id, sample.vid

        pose = np.array((sample.pose_pitch, sample.pose_yaw, sample.pose_roll), dtype=np.float32)
        landmarks = self.get_landmarks(sample)

        t = time.time()
        crop, landmarks, pose, cropper = self.face_extractor.get_face(filename + '.jpg', self.fullsize_img_dir,
                                                                      self.cropped_img_dir, landmarks=landmarks,
                                                                      pose=pose, use_cache=self.use_cache,
                                                                      detect_face=False, crop_type='tight',
                                                                      aligned=True)
        # self.show_landmarks(crop, landmarks, 'imgs')
        # crop, pose, landmarks,of_conf_seq = self.get_face(fname, landmarks=landmarks, use_cache=True, from_sequence=True)
        # landmarks = face_processing.scale_landmarks_to_crop(landmarks, output_size=(cfg.CROP_SIZE, cfg.CROP_SIZE))
        # self.show_landmarks(crop, landmarks, 'sequence')
        # print(of_conf, of_conf_seq)
        # cv2.waitKey()

        cropped_sample = {'image': crop, 'landmarks': landmarks, 'pose': pose}
        item = self.transform(cropped_sample)

        # face_mask = face_processing.get_face_mask(landmarks, crop.shape)
        # transformed_face_mask = face_processing.CenterCrop(cfg.INPUT_SIZE)(face_mask)

        item.update({
            'id': id,
            'fnames': filename,
            # 'face_mask': transformed_face_mask,
            'expression': np.array([[-1,0,0]], np.float32),
            'vid': vid
        })

        if self.with_bumps:
            H = np.eye(3,3)
            step = 1
            next_id = max(0, min(len(self)-1, idx+step))
            if self.annotations.iloc[next_id].vid != vid:
                next_id = max(0, min(len(self)-1, idx-step))
            if self.annotations.iloc[max(0, idx-step)].vid != vid:
                # fallback to single image
                crop_next = crop
                landmarks_next = landmarks
            else:
                sample_next = self.annotations.iloc[next_id]
                pose_next = np.array((sample_next.pose_pitch, sample_next.pose_yaw, sample_next.pose_roll), dtype=np.float32)
                of_conf = sample_next.of_conf
                crop_next, landmarks_next = self.get_face(sample_next.fname,
                                                          landmarks=self.get_landmarks(sample_next),
                                                          use_cache=True)

                pose_diff = np.abs(pose - pose_next)
                if np.any(pose_diff > np.deg2rad(7)) or of_conf < 0.9:
                    # print(np.rad2deg(pose_diff))
                    crop_next = crop
                    landmarks_next = landmarks
                else:
                    # calculate homograpy to project next images onto current image
                    H = cv2.findHomography(landmarks_next, landmarks)[0]

            bumps = face_processing.calc_face_bumps(crop, crop_next, landmarks, landmarks_next, H)
            transformed_bumps = face_processing.CenterCrop(cfg.INPUT_SIZE)(bumps)
            # transformed_bumps = self.transform(bumps)

            # bumps = face_processing.calc_face_bumps(crop, crop_next, landmarks, landmarks_next)
            # transformed_crop_next = self.transform(crop_next)
            # return transformed_crop, id, pose, landmarks, emotion, vid, fname, transformed_crop_next, landmarks_next, H, transformed_bumps
            item.update({'bumps': transformed_bumps})

        return item

    def get_face(self, filename, landmarks=None, size=(cfg.CROP_SIZE, cfg.CROP_SIZE), use_cache=True, from_sequence=False):
        # landmarks = np.zeros((68, 2), dtype=np.float32)
        # pose = np.zeros(3, dtype=np.float32)
        crop_filepath = os.path.join(self.cropped_img_dir, filename + '.jpg')

        if use_cache and os.path.isfile(crop_filepath):
            try:
                crop = io.imread(crop_filepath)
            except OSError:
                os.remove(crop_filepath)
                return self.get_face(filename, landmarks, size, use_cache, from_sequence)
            if crop.shape[:2] != size:
                crop =  cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
            if landmarks is None:
                of_conf, landmarks, _ = ds_utils.read_openface_detection(
                    os.path.join(self.feature_dir, filename),
                    numpy_lmFilepath=os.path.join(self.npfeature_dir, filename)
                )
            landmarks = face_processing.scale_landmarks_to_crop(landmarks, output_size=size)
        else:
            # Load image from dataset
            img_path = os.path.join(self.fullsize_img_dir, filename + '.jpg')
            img = io.imread(img_path)
            if img is None:
                raise IOError("\tError: Could not load image {}!".format(img_path))

            # load landmarks extracted with OpenFace2
            if landmarks is None:
                of_conf, landmarks, _ = ds_utils.read_openface_detection(
                    os.path.join(self.feature_dir, filename),
                    numpy_lmFilepath=os.path.join(self.npfeature_dir, filename),
                    from_sequence=from_sequence
                )
                if of_conf <= 0.0:
                    log.warning("No landmarks for image {}".format(filename))

            # crop, landmarks = face_processing.crop_bump(img, landmarks, output_size=size)
            crop, landmarks = face_processing.crop_celebHQ(img, landmarks, output_size=size)

            if use_cache:
                utils.io.makedirs(crop_filepath)
                io.imsave(crop_filepath, crop)

        return crop, landmarks



def get_name_uid_map(split):
    import glob
    def create_map(in_dir):
        ytid2pid = {}
        person_dirs = sorted(glob.glob(os.path.join(cfg.VOXCELEB_ROOT, in_dir, '*')))
        for cnt, p_dir in enumerate(person_dirs):
            pid = p_dir.split('/')[-1]
            if 'frames' in in_dir:
                vid_dirs = sorted(glob.glob(os.path.join(p_dir, '1.6', '*')))
            else:
                vid_dirs = sorted(glob.glob(os.path.join(p_dir, '*')))
            for cnt_vids, vid_dir in enumerate(vid_dirs[:1]):
                yt_id = vid_dir.split('/')[-1]
                ytid2pid[yt_id] = pid
        return ytid2pid

    ytid2uid = create_map(split+'_txt')
    ytid2name = create_map('frames/unzippedIntervalFaces/data')

    # name2uid = {}
    map = {}
    for ytid, uid in ytid2uid.items():
        name = ytid2name[ytid]
        # print(uid, name, ytid)
        map[name] = uid
        map[uid] = name
        # name2uid[name] = uid
        # uid2name[uid] = name

    return map


def create_annotations(split, num_ids):
    import glob
    WRITE_CROPS = False
    annotations = {}
    id_map = get_name_uid_map(split)
    # person_dirs = sorted(glob.glob(os.path.join(cfg.VOXCELEB_ROOT, 'frames/unzippedIntervalFaces/data', '*')))
    person_dirs = sorted(glob.glob(os.path.join(cfg.VOXCELEB_ROOT, split+'_txt', '*')))
    for cnt, p_dir in enumerate(person_dirs[:num_ids]):
        uid = p_dir.split('/')[-1]
        name_dir = p_dir.replace(split+'_txt', 'frames/unzippedIntervalFaces/data')
        print(cnt, name_dir, uid)
        try:
            name_dir = name_dir.replace(uid, id_map[uid])
        except:
            pass
        vid_dirs = sorted(glob.glob(os.path.join(name_dir, '1.6', '*')))
        for cnt_vids, vid_dir in enumerate(vid_dirs):
            track_dirs = sorted(glob.glob(os.path.join(vid_dir, '*')))
            for cnt_vids, img_dir in enumerate(track_dirs):
                imagefiles = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
                for cnt_img, img_path in enumerate(imagefiles):
                    # lm_path = img_path.replace('frames', 'features').replace('jpg', 'csv')
                    # of_conf, landmarks, pose = ds_utils.read_openface_detection(lm_path)
                    # of_conf = 0.1

                    # face_height = landmarks[:, 1].max() - landmarks[:, 1].min()

                    # if of_conf > 0.0:
                    #     if WRITE_CROPS:
                    #         img_orig = io.imread(img_path)
                    #
                    #         crop, landmarks_crop = face_processing.crop_bump(img_orig, landmarks,
                    #                                                          output_size=(cfg.INPUT_SIZE + cfg.CROP_BORDER * 2,
                    #                                                                       cfg.INPUT_SIZE + cfg.CROP_BORDER * 2))
                    #
                    #         crop_filepath = img_path.replace('OriginalImg', 'crops')
                    #         utils.io.makedirs(crop_filepath)
                    #         io.imsave(crop_filepath, crop)
                    # else:
                    #     print(img_path)

                    fname = os.path.splitext('/'.join(img_path.split('/')[-5:]))[0]
                    annotations[fname] = {'uid': uid}#, 'conf': of_conf, 'face_height': face_height}

    print("Saving DataFrame...")
    df = pd.DataFrame.from_dict(annotations, orient='index')
    df.index.name = 'fname'
    df.to_csv(os.path.join(cfg.VOXCELEB_ROOT, split+'.csv'))




def read_openface_csvs():
    class VoxCelebLandmarks(td.Dataset):

        def __init__(self, root_dir=cfg.VOXCELEB_ROOT, train=True, start=None,
                 max_samples=None):

            self.root_dir = root_dir
            self.cropped_img_dir = os.path.join(cfg.VOXCELEB_ROOT_LOCAL, 'crops/unzippedIntervalFaces/data')
            self.fullsize_img_dir = os.path.join(root_dir, 'frames/unzippedIntervalFaces/data')
            self.feature_dir = os.path.join(root_dir, 'features/unzippedIntervalFaces/data')
            self.npfeature_dir = os.path.join(cfg.VOXCELEB_ROOT_LOCAL, 'features/unzippedIntervalFaces/data')

            annotation_filename = 'dev' if train else 'test'
            self.annotations = pd.read_csv(os.path.join(root_dir, annotation_filename + '.csv'))

            # limit number of samples
            st, nd = 0, None
            if start is not None:
                st = start
            if max_samples is not None:
                nd = st + max_samples
            self.annotations = self.annotations[st:nd]

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            fname = self.annotations.iloc[idx].fname
            of_conf, landmarks, pose = ds_utils.read_openface_detection(
                os.path.join(self.feature_dir, fname),
                numpy_lmFilepath=os.path.join(self.npfeature_dir, fname),
                from_sequence=False
            )
            return {'fn': fname,
                    'cnf': of_conf,
                    'lmx': landmarks[:,0],
                    'lmy': landmarks[:,1],
                    'h': int(landmarks[:,1].max() - landmarks[:,1].min()),
                    'w': int(landmarks[:,0].max() - landmarks[:,0].min()),
                    'p': pose[0],
                    'y': pose[1],
                    'r': pose[2]
                    }

    train = True
    data = []
    ds = VoxCelebLandmarks(train=train, max_samples=8000000)
    dl = td.DataLoader(ds, batch_size=200, shuffle=False, num_workers=12, collate_fn=lambda b: b)
    for batch in dl:
        data.extend(batch)
        log.info(len(data))

    print("Saving DataFrame...")
    df = pd.DataFrame.from_dict(data)
    df = df.set_index('fn')
    annotation_filename = 'dev' if train else 'test'
    df.to_pickle(os.path.join(cfg.VOXCELEB_ROOT, annotation_filename + '.lms.pkl'))


def extract_features(split='dev', st=None, nd=None, is_sequence=True):
    """ Extract facial features (landmarks, pose,...) from images """
    import glob
    from utils import visionLogging as log
    # person_dirs = sorted(glob.glob(os.path.join(cfg.VOXCELEB_ROOT, 'frames/unzippedIntervalFaces/data', '*')))
    id_map = get_name_uid_map(split)
    person_dirs = sorted(glob.glob(os.path.join(cfg.VOXCELEB_ROOT, split+'_txt', '*')))
    for cnt, p_dir in enumerate(person_dirs[st:nd]):
        uid = p_dir.split('/')[-1]
        name_dir = p_dir.replace(split + '_txt', 'frames/unzippedIntervalFaces/data')
        try:
            person_name = id_map[uid]
            name_dir = name_dir.replace(uid, person_name)
            log.info("{}[{}-{}] {}".format(st + cnt, st, nd, person_name))
        except:
            print(name_dir, uid)
        vid_dirs = sorted(glob.glob(os.path.join(name_dir, '1.6', '*')))
        for cnt_vids, vid_dir in enumerate(vid_dirs):
            track_dirs = sorted(glob.glob(os.path.join(vid_dir, '*')))
            for cnt_vids, img_dir in enumerate(track_dirs):
                if is_sequence:
                    out_dir = img_dir.replace('frames', 'features_sequence')
                else:
                    out_dir = img_dir.replace('frames', 'features')
                face_processing.run_open_face(img_dir, out_dir, is_sequence=is_sequence)


def extract_main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--st', default=None, type=int)
    parser.add_argument('--nd', default=None, type=int)
    args = parser.parse_args()

    extract_features(st=args.st, nd=args.nd)

if __name__ == '__main__':
    # extract_main()
    # create_annotations(split='dev', num_ids=500)
    # extract_crops()

    # read_openface_csvs()
    # exit()

    from utils import vis, face_processing

    ds = VoxCeleb(train=True, max_samples=50000, use_cache=True)
    print(ds)
    dl = td.DataLoader(ds, batch_size=40, shuffle=False, num_workers=0)
    from utils.nn import Batch

    for data in dl:
        batch = Batch(data)
        print(batch.ids)
        ds_utils.denormalize(batch.images)
        # vis.show_images_in_batch(batch.images.detach().cpu())
        vis.vis_square(batch.images.detach().cpu(), fx=0.7, fy=0.7, normalize=False)
        # print(item)
