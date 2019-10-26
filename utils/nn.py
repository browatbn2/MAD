import json
import os

import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist
from torch import nn as nn
from torch.utils import data as td

import config as cfg


def tensor_dist_mat(v):
    # input vector is row vector like
    # 1,
    # 2,
    # 3
    n, dims = v.shape

    # repeat input vector
    # 1,2,3,1,2,3,1,2,3
    col_vec = v.repeat(n, 1)

    # convert input vector to matrix by repeating rows
    # 1,2,3
    # 1,2,3
    # 1,2,3
    row_mat = v.view(n, 1, -1).repeat(1, n, 1)
    # reshape matrix to vector
    # 1,1,1,2,2,2,3,3,3
    row_vec = row_mat.view(n*n, dims)

    # calc pairwise distance between stacked vectors
    pdists = nn.PairwiseDistance(p=2)(row_vec, col_vec)

    # reshape to get distance matrix
    return pdists.view(n, n)


def expanded_pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    if y is not None:
         differences = x.unsqueeze(1) - y.unsqueeze(0)
    else:
        differences = x.unsqueeze(1) - x.unsqueeze(0)
    distances = torch.sum(differences * differences, -1)
    return distances


def categ_dist_mat(c):
    n = len(c)
    c = np.atleast_2d(c)
    row_mat = np.tile(c, reps=(n, 1))
    col_mat = np.tile(c.transpose(), reps=(1, n))
    return col_mat != row_mat
    # bin_cl = (col_mat != row_mat).astype(np.float32)
    # return bin_cl


def expr_dist_mat(e, v, a):
    l2_va = squareform(pdist(np.stack((v, a), axis=1), metric='euclidean'))
    n = len(e)
    e = np.atleast_2d(e)
    row_mat = np.tile(e, reps=(n, 1))
    col_mat = np.tile(e.transpose(), reps=(1, n))
    bin_cl = (col_mat != row_mat).astype(np.float32)
    dist_mat = l2_va*1.0 + bin_cl * 0.25
    # dist_mat = l2_va*1.0 #+ bin_cl * 0.25
    # dist_mat = bin_cl
    return dist_mat.astype(np.float32)


def shuffle_within_class(outputs, labels):
    num = len(labels)
    target_dist_mat = categ_dist_mat(labels)
    dists_idx = np.argsort(target_dist_mat, axis=1)
    pivot_posneg = (1.0 - target_dist_mat).sum(axis=1)
    inds = np.arange(num)
    pos_rnd = []
    for i in inds:
        try:
            val = np.random.randint(1, pivot_posneg[i])
        except ValueError:
            # only one sample in class
            val = 0
        pos_rnd.append(val)
    new_ids = dists_idx[inds, pos_rnd]
    new_outputs = outputs[new_ids, :]
    return new_outputs


def make_triplets(outputs, labels_, debug=False):
    if isinstance(labels_, list):
        ids = to_numpy(labels_[0])
        labels = labels_[1]
    else:
        labels = labels_

    labels = to_numpy(labels)

    num = len(labels)
    assert(num > 1)
    inds = np.arange(num)

    embedding_dist_mat = tensor_dist_mat(outputs).detach().cpu().numpy()

    is_expression_labels = len(labels.shape) > 1 and labels.shape[1] == 3

    # FIXME
    # if is_expression_labels:
    #     labels = labels[:, 0]
    #     is_expression_labels = False

    if is_expression_labels:
        target_dist_mat = expr_dist_mat(labels[:, 0], labels[:, 1], labels[:, 2])
    else:
        target_dist_mat = categ_dist_mat(labels)

    pos_rnd = []
    neg_rnd = []
    if is_expression_labels:
        dists_idx = np.argsort(target_dist_mat, axis=1)
        pos_rnd = 1
        # neg_rnd = np.random.randint(num//2, num, size=num)

        neg_rnd = np.random.randint(10, num, size=num)
        # neg_rnd = num//2

        # pivot_posneg = (target_dist_mat < 0.20).sum(axis=1)
        # pos_rnd = (np.random.uniform(1/num, pivot_posneg/num, size=num) * num).astype(int)
        # neg_rnd = (np.random.uniform(pivot_posneg/num, size=num) * num).astype(int)

        pos_id = dists_idx[inds, pos_rnd]
        neg_id = dists_idx[inds, neg_rnd]
    else:
        # select positive sample (any of same class)
        pivot_posneg = (1.0 - target_dist_mat).sum(axis=1).astype(int)
        dists_idx = np.argsort(target_dist_mat+embedding_dist_mat*0.5, axis=1)
        for i in inds:
            try:
                val = np.random.randint(1, pivot_posneg[i])
                # val = max(0, pivot_posneg[i]-2) # take most difficult positive sample
            except ValueError:
                # only one sample in class
                val = 0
            pos_rnd.append(val)
        pos_id = dists_idx[inds, pos_rnd]

        if isinstance(labels_, list):
            # print(pivot_posneg, np.mean(pivot_posneg))
            target_dist_mat = 1 - (categ_dist_mat(ids).astype(np.float32) + (1 - target_dist_mat.astype(np.float32)))
            dists_idx = np.argsort(target_dist_mat+embedding_dist_mat*0.1, axis=1)
            # pivot_posneg_labels = pivot_posneg.copy()
            pivot_posneg = (1.0 - target_dist_mat).sum(axis=1)
            # print(pivot_posneg)
            # print('---------')

        # select negative samples (look for hard negatives)
        for i in inds:
            try:
                if cfg.HARD_TRIPLETS_FOR_IDENTITY:
                    hardest_pos = int(pivot_posneg[i])
                    easiest_pos = hardest_pos + 5
                    neg_rnd.append(min(num-1, np.random.randint(hardest_pos, easiest_pos)))
                else:
                    neg_rnd.append(np.random.randint(pivot_posneg[i], pivot_posneg[i]+1+(num-pivot_posneg[i]-1)/2))
            except:
                # all samples of same class
                # print(i, ' all same class')
                neg_rnd.append(num-1)
        neg_id = dists_idx[inds, neg_rnd]

    if debug:
        try:
            for i in range(10):
                pid, nid = pos_id[i], neg_id[i]
                print("({:5.2f}, {:5.2f}): {:5.2f} ({:5.2f}, {:5.2f}), {:5.2f} ({:5.2f}, {:5.2f})".format(
                    labels[i, 1].item(), labels[i, 2].item(),
                    target_dist_mat[i, pid].item(),
                    labels[pid, 1].item(), labels[pid, 2].item(),
                    target_dist_mat[i, nid].item(),
                    labels[nid, 1].item(), labels[nid, 2].item()
                ))
        except:
            pass

    return pos_id, neg_id


def calc_triplet_loss(outputs, c, return_acc=False, images=None, feature_name=None, wnd_title=None):

    margin = 0.2
    eps = 1e-8

    debug = False
    is_expressions = (not isinstance(c, list)) and len(c.shape) > 1 and c.shape[1] == 3

    pos_id, neg_id = make_triplets(outputs, c, debug=debug)

    X,P,N = outputs[:,:], outputs[pos_id,:], outputs[neg_id,:]
    dpos = torch.sqrt(torch.sum((X - P)**2, dim=1) + eps)
    dneg = torch.sqrt(torch.sum((X - N)**2, dim=1) + eps)
    loss = torch.mean(torch.clamp(dpos-dneg+margin, min=0.0, max=margin*2.0))
    # show triplets
    if images is not None:
        from utils import vis
        from datasets import ds_utils
        if debug and is_expressions:
            for i in range(10):
                print(c[:,0][i].item(), c[pos_id,0][i].item(), c[neg_id,0][i].item())
        # ids, vids = c[0], c[1]
        # print(vids[:5])
        # print(vids[pos_id][:5])
        # print(vids[neg_id][:5])
        nimgs = 20
        losses = to_numpy(torch.clamp(dpos-dneg+margin, min=0.0, max=margin*2.0))
        # print("Acc: ", 1 - sum(dpos[:nimgs] >= dneg[:nimgs]).item()/float(len(dpos[:nimgs])))
        # print("L  : ", losses.mean())
        images_ref = ds_utils.denormalized(images[:nimgs].clone())
        images_pos = ds_utils.denormalized(images[pos_id][:nimgs].clone())
        images_neg = ds_utils.denormalized(images[neg_id][:nimgs].clone())
        colors = [(0, 1, 0) if c else (1, 0, 0) for c in dpos < dneg]
        f = 0.75
        images_ref = vis.add_error_to_images(vis.add_cirle_to_images(images_ref, colors),
                                             losses, size=1.0, vmin=0, vmax=0.5, thickness=2, format_string='{:.2f}')
        images_pos = vis.add_error_to_images(images_pos, to_numpy(dpos), size=1.0, vmin=0.5, vmax=1.0,
                                             thickness=2, format_string='{:.2f}')
        images_neg = vis.add_error_to_images(images_neg, to_numpy(dneg), size=1.0, vmin=0.5, vmax=1.0,
                                             thickness=2, format_string='{:.2f}')
        if is_expressions:
            emotions = to_numpy(c[:, 0]).astype(int)
            images_ref = vis.add_emotion_to_images(images_ref, emotions)
            images_pos = vis.add_emotion_to_images(images_pos, emotions[pos_id])
            images_neg = vis.add_emotion_to_images(images_neg, emotions[neg_id])
        elif feature_name == 'id':
            ids = to_numpy(c).astype(int)
            images_ref = vis.add_id_to_images(images_ref, ids, loc='tr')
            images_pos = vis.add_id_to_images(images_pos, ids[pos_id], loc='tr')
            images_neg = vis.add_id_to_images(images_neg, ids[neg_id], loc='tr')

        img_ref = vis.make_grid(images_ref, nCols=nimgs, padsize=1, fx=f, fy=f)
        img_pos = vis.make_grid(images_pos, nCols=nimgs, padsize=1, fx=f, fy=f)
        img_neg = vis.make_grid(images_neg, nCols=nimgs, padsize=1, fx=f, fy=f)
        title = 'triplets'
        if feature_name is not None:
            title += " " + feature_name
        if wnd_title is not None:
            title += " " + wnd_title
        vis.vis_square([img_ref, img_pos, img_neg], nCols=1, padsize=1, normalize=False, wait=10, title=title)

        # plt.plot(to_numpy((X[:nimgs]-P[:nimgs]).abs()), 'b')
        # plt.plot(to_numpy((X[:nimgs]-N[:nimgs]).abs()), 'r')
        # plt.show()
    if return_acc:
        return loss, sum(dpos >= dneg).item()/float(len(dpos))
    else:
        return loss


def to_numpy(ft):
    if isinstance(ft, np.ndarray):
        return ft
    try:
        return ft.detach().cpu().numpy()
    except AttributeError:
        return None


def to_image(m):
    img = to_numpy(m)
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    return img



class Batch:

    def __init__(self, data, n=None, gpu=True, eval=False):
        self.images = data['image']
        self.eval = eval

        try:
            self.ids = data['id']
            try:
                if self.ids.min() < 0 or self.ids.max() == 0:
                    self.ids = None
            except AttributeError:
                self.ids = np.array(self.ids)
        except KeyError:
            self.ids = None

        try:
            self.images_mod = data['image_mod']
        except KeyError:
            self.images_mod = None

        try:
            self.face_heights = data['face_heights']
        except KeyError:
            self.face_heights = None

        try:
            self.poses = data['pose']
        except KeyError:
            self.poses = None

        try:
            self.landmarks = data['landmarks']
        except KeyError:
            self.landmarks = None

        try:
            self.emotions = torch.squeeze(data['expression'])[:,0].long()
            self.valence = torch.squeeze(data['expression'])[:,1].float()
            self.arousal = torch.squeeze(data['expression'])[:,2].float()
            self.expression = torch.squeeze(data['expression'])
            if self.emotions.min() < 0:
                raise ValueError
        except (KeyError, IndexError, ValueError):
            self.emotions = None
            self.valence = None
            self.arousal = None
            self.expression = None

        try:
            self.clips = np.array(data['vid'])
        except KeyError:
            self.clips = None

        try:
            self.fnames = data['fnames']
        except:
            self.fnames = None

        try:
            self.bumps = data['bumps']
        except:
            self.bumps = None

        try:
            self.face_masks = data['face_mask']
        except:
            self.face_masks = None

        if self.face_masks is not None:
            self.face_weights = self.face_masks.float()
            if not self.eval:
                self.face_weights += 1.0
            self.face_weights /= self.face_weights.max()
            # plt.imshow(self.face_weights[0,0])
            # plt.show()

            if cfg.WITH_FACE_MASK:
                mask = self.face_masks.unsqueeze(1).expand_as(self.images).float()
                mask /= mask.max()
                self.images *= mask

        self.lm_heatmaps = None

        try:
            # self.face_weights = data['face_weights']
            self.lm_heatmaps = data['lm_heatmaps']
            if len(self.lm_heatmaps.shape) == 3:
                self.lm_heatmaps = self.lm_heatmaps.unsqueeze(1)
        except KeyError:
            self.face_weights = 1.0

        for k, v in self.__dict__.items():
            if v is not None:
                try:
                    self.__dict__[k] = v[:n]
                except TypeError:
                    pass

        if gpu:
            for k, v in self.__dict__.items():
                if v is not None:
                    try:
                        self.__dict__[k] = v.cuda()
                    except AttributeError:
                        pass

    def __len__(self):
        return len(self.images)



class MacroBatch(td.Dataset):
    def __init__(self, batch, ds):
        self.ds = ds
        self.inds = np.concatenate(batch)

    def __getitem__(self, idx):
        return self.ds[self.inds[idx]]

    def __len__(self):
        return len(self.inds)


class GroupedIter():
    def __init__(self, values, max_group_size, vids=None, num_vids_per_id=None):
        self.values = np.array(values)
        self.vids = np.array(vids) if vids is not None else None
        self.max_group_size = max_group_size
        self.num_vids_per_id = num_vids_per_id
        self.unique = np.unique(self.values)
        self.nunique = len(self.unique)
        self.num2idx = {k: v for k,v in enumerate(self.unique)}

    def __getitem__(self, num):
        id = self.num2idx[num]
        vals = np.where(self.values == id)[0]

        if self.vids is not None:
            vids_of_id = self.vids[vals]
            unique_vids_of_id = np.unique(vids_of_id)
            # print("num vids {}: {}".format(id, len(unique_vids_of_id)))
            selected_vid_ids = np.random.permutation(unique_vids_of_id)[:self.num_vids_per_id]

            keep_vals = []
            for vid in selected_vid_ids:
                inds = np.where(vids_of_id == vid)[0]
                # print(id, vid, len(inds))
                keep_vals.append(vals[inds])
            vals = np.concatenate(keep_vals)

        # print(len(vals), self.max_group_size)
        np.random.shuffle(vals)
        return vals[:self.max_group_size]

    def __len__(self):
        return self.nunique


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def read_model(in_dir, model_name, model):
    filepath_mdl = os.path.join(in_dir, model_name+'.mdl')
    snapshot = torch.load(filepath_mdl)
    try:
        model.load_state_dict(snapshot['state_dict'], strict=False)
    except RuntimeError as e:
        print(e)


def read_meta(in_dir):
    with open(os.path.join(in_dir, 'meta.json'), 'r') as outfile:
        data = json.load(outfile)
    return data


