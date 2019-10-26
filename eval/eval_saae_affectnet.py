import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data as td

import utils.nn
from networks import saae
import config as cfg
from datasets import affectnet
from utils import nn, vis
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import sklearn.metrics
import utils.log as log


def calc_acc(outputs, labels):
    assert(outputs.shape[1] == 8)
    assert(len(outputs) == len(labels))
    preds = np.argmax(outputs, 1)
    corrects = np.sum(preds == labels)
    acc = corrects/float(len(outputs))
    return acc

def evaluate(clprobs, labels):
    # Calculate evaluation metrics
    accuracy = calc_acc(clprobs, labels)

    # Compute ROC curve and ROC area for each class
    y = label_binarize(labels, classes=range(8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 8
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], clprobs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), clprobs.ravel())
    roc_auc_micro = auc(fpr["micro"], tpr["micro"])

    # Create confusion matrix
    conf_matrix = sklearn.metrics.confusion_matrix(labels, clprobs.argmax(axis=1))
    return accuracy, roc_auc, roc_auc_micro, conf_matrix


    # if False:
    #     # Calculate correlation between recon error and emotion predictions
    #     recon_errors = np.concatenate([stats['l1_recon_errors'] for stats in self.epoch_stats])
    #     cycle_errors = np.concatenate([stats['l1_dis_cycle'] for stats in self.epoch_stats])
    #
    #     pred_prob = clprobs[range(len(labels)), labels]
    #     pred_label = clprobs.argmax(axis=1)
    #
    #     print(recon_errors[labels == pred_label].mean(), recon_errors[labels != pred_label].mean())
    #     print(cycle_errors[labels == pred_label].mean(), cycle_errors[labels != pred_label].mean())
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     colors = [affectnet.AffectNet.colors[l] for l in labels]
    #
    #     colors = ['g' if labels[l] == pred_label[l] else 'r' for l in range(len(labels))]
    #     ax.scatter(recon_errors, pred_prob, c=colors)
    #     plt.show()


def eval_affectnet(net, n=2000, feat_type=3, eval_notf=True, only_good_images=True, show=False):

    print("Evaluating AffectNet...")

    batch_size = 20 if show else 100
    dataset = affectnet.AffectNet(train=False, max_samples=n, deterministic=True, use_cache=True)
    dataloader = td.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    print(dataset)

    labels = []
    clprobs = []

    for iter, data in enumerate(dataloader):
        batch = nn.Batch(data)

        with torch.no_grad():
            X_recon = net(batch.images, Y=None)[:, :3]
        if show:
            nimgs = 25
            f = 1.0
            img = saae.draw_results(batch.images, X_recon, net.z_vecs(),
                                    emotions=batch.emotions, emotions_pred=net.emotions_pred(),
                                    fx=f, fy=f, ncols=10)
            cv2.imshow('reconst', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()

        clprobs.append(nn.to_numpy(net.emotion_probs))
        labels.append(nn.to_numpy(batch.emotions))

        if (iter % 10) == 0:
            print(iter)


    clprobs = np.vstack(clprobs)
    labels = np.concatenate(labels).astype(int)

    accuracy, auc, auc_micro, conf_matrix = evaluate(clprobs, labels)
    print('Accuracy  F: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    log.info("\nAUCs: {} ({})".format(auc, np.mean(list(auc.values()))))
    log.info("\nAUC micro: {} ".format(auc_micro))

    print(conf_matrix)
    vis.plot_confusion_matrix(conf_matrix, classes=affectnet.AffectNet.classes, normalize=True)
    plt.show()



if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    import argparse
    bool_str = lambda x: (str(x).lower() in ['true', '1'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default=None, type=str, help='model used to generate samples')
    parser.add_argument('-f', default=3, type=int, help='feature to evaluate ')
    parser.add_argument('--eval-notf', default=False, action='store_true', help='eval feature complement')
    parser.add_argument('--show', default=False, action='store_true', help='show pairs')
    parser.add_argument('-n', default=4000, type=int, help='number of images')
    parser.add_argument('--easy', default=False, action='store_true', help='only images with high face det confs')
    args = parser.parse_args()

    modelfile = cfg.CURRENT_MODEL
    if args.modelname is not None:
        modelfile = os.path.join(cfg.SNAPSHOT_DIR, args.modelname)

    net = saae.SAAE()
    print("Loading model {}...".format(modelfile))
    utils.nn.read_model(modelfile, 'saae', net)
    net.eval()

    eval_affectnet(net, n=args.n, feat_type=args.f, eval_notf=args.eval_notf, show=args.show, only_good_images=args.easy)

