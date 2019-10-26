import os
import cv2
import numpy as np

import torch
import torch.utils.data as td

import utils.nn
from networks import saae
import config as cfg
from datasets import lfw
from utils import vis
from datasets import ds_utils
import metrics.roc


def show_pairs(images, features, pairs):
    dists = np.sqrt(np.sum((features[0] - features[1])**2, axis=1))
    ds_utils.denormalize(images[0])
    ds_utils.denormalize(images[1])
    images[1] = vis.add_error_to_images(images[1], dists, size=2.0, thickness=2, vmin=0, vmax=1)
    images[1] = vis.add_id_to_images(images[1], pairs.numpy(), size=1.2, thickness=2, color=(1, 0, 1))
    thresh = 0.4
    corrects = (dists < thresh) == pairs.cpu().numpy()
    colors = [(0,1,0) if c else (1,0,0) for c in corrects]
    images[1] = vis.add_cirle_to_images(images[1], colors)
    images[0] = vis._to_disp_images(images[0])
    img_rows = [vis.make_grid(imgs, fx=0.75, fy=0.75, nCols=len(dists), normalize=False) for imgs in images]
    vis.vis_square(img_rows, nCols=1, normalize=False)


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0.0, 2, 0.01)
    tpr, fpr, accuracy = metrics.roc.calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                                                   nrof_folds=nrof_folds, distance_metric=distance_metric,
                                                   subtract_mean=subtract_mean)
    # thresholds = np.arange(0, 2, 0.001)
    # val, val_std, far = metrics.roc.calculate_val(thresholds, embeddings1, embeddings2, actual_issame, 1e-3,
    #                                               nrof_folds=nrof_folds, distance_metric=distance_metric,
    #                                               subtract_mean=subtract_mean)
    return tpr, fpr, accuracy#, val, val_std, far


def eval_lfw(net, n=6000, feat_type=1, only_good_images=False, show=False):

    print("Evaluating LFW...")

    MIN_CONF = 0.6
    batch_size = 40 if show else 100
    dataset = lfw.LFW(train=False, max_samples=n, deterministic=True, use_cache=True)
    dataloader = td.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    embeddings1 = []
    embeddings2 = []
    embeddings1_h = []
    embeddings2_h = []
    actual_issame = []
    names = []
    confs1 = []
    confs2 = []

    for iter, data in enumerate(dataloader):
        images1 = data[0].cuda()
        images2 = data[1].cuda()
        conf1 = data[-2].numpy()
        conf2 = data[-1].numpy()

        names.append(np.vstack((data[2], data[3])).T)
        confs1.append(conf1)
        confs2.append(conf2)

        with torch.no_grad():
            net(images1, Y=None)
            embeddings1.append(net.f_vec(feat_type))
            embeddings1_h.append(net.res_vec(feat_type))

            net(images2, Y=None)
            embeddings2.append(net.f_vec(feat_type))
            embeddings2_h.append(net.res_vec(feat_type))

        actual_issame.append(data[4])

        if show:
            # keep_ids = (conf1 > MIN_CONF) & (conf2 > MIN_CONF)
            dists = np.sqrt(np.sum((embeddings1[-1] - embeddings2[-1]) ** 2, axis=1))
            pairs = data[4].numpy()
            incorrects = (dists > 0.8) == pairs
            # keep_ids = np.where(((conf1 > MIN_CONF) & (conf2 > MIN_CONF) & pairs & incorrects))[0]
            # keep_ids = np.where(~((conf1 > MIN_CONF) & (conf2 > MIN_CONF)) & pairs & incorrects)[0]
            keep_ids = np.where(conf1 > MIN_CONF)[0]
            # print(keep_ids)
            if len(keep_ids) > 0:
                # print(names[-1][keep_ids])
                img = saae.vis_reconstruction(net, torch.cat((images1[keep_ids], images2[keep_ids])), fx=1.0, fy=1.0, ncols=batch_size)
                cv2.imshow('reconst', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                # print(net.levels['pose'].z.detach().cpu().numpy()[:,-3])
                show_pairs([images1[keep_ids], images2[keep_ids]], [embeddings1[-1][keep_ids], embeddings2[-1][keep_ids]], actual_issame[-1][keep_ids])

    embeddings1 = np.vstack(embeddings1)
    embeddings2 = np.vstack(embeddings2)
    embeddings1_h = np.vstack(embeddings1_h)
    embeddings2_h = np.vstack(embeddings2_h)
    confs1 = np.concatenate(confs1)
    confs2 = np.concatenate(confs2)
    actual_issame = np.concatenate(actual_issame).astype(int)

    # print(embeddings1[:5])
    # print("")
    # print(embeddings2[:5])

    # only evalutate pairs with high OF confidences
    if only_good_images:
        keep_ids = (confs1 > MIN_CONF) & (confs2 > MIN_CONF)
        # keep_ids = ~keep_ids
        print("Num good pairs: {}".format(np.count_nonzero(keep_ids)))
        embeddings1 = embeddings1[keep_ids]
        embeddings2 = embeddings2[keep_ids]
        embeddings1_h = embeddings1_h[keep_ids]
        embeddings2_h = embeddings2_h[keep_ids]
        actual_issame = actual_issame[keep_ids]

    tpr, fpr, accuracy = evaluate(embeddings1, embeddings2, actual_issame)
    print('Accuracy  F: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))

    tpr, fpr, accuracy = evaluate(embeddings1_h, embeddings2_h, actual_issame)
    print('Accuracy ~F: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))



if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


    import argparse
    bool_str = lambda x: (str(x).lower() in ['true', '1'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default=None, type=str, help='model used to generate samples')
    parser.add_argument('-f', default=1, type=int, help='feature to evaluate ')
    parser.add_argument('--eval-notf', default=False, action='store_true', help='eval feature complement')
    parser.add_argument('--show', default=False, action='store_true', help='show pairs')
    parser.add_argument('-n', default=6000, type=int, help='number of pairs')
    parser.add_argument('--easy', default=False, action='store_true', help='only images with high face det confs')
    args = parser.parse_args()

    modelfile = cfg.CURRENT_MODEL
    if args.modelname is not None:
        modelfile = os.path.join(cfg.SNAPSHOT_DIR, args.modelname)

    net = saae.SAAE()
    print("Loading model {}...".format(modelfile))
    utils.nn.read_model(modelfile, 'saae', net)
    # net.eval()

    eval_lfw(net, n=args.n, feat_type=args.f, show=args.show, only_good_images=args.easy)

