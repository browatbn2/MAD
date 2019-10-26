import itertools
import os

import cv2
from matplotlib import pyplot as plt
from matplotlib.offsetbox import DrawingArea, AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
import numpy as np
import sklearn.metrics
import seaborn as sns
from utils.nn import to_numpy
from datasets.ds_utils import denormalized


def show_expression_confusion_matrix(results, modelname=''):
    fig, ax = plt.subplots(figsize=(6,6))
    class_preds = results['class'].values
    labels = results['gt_class'].values
    correct = labels == class_preds
    acc = float(sum(correct)) / len(labels)
    print("{} Accuracy: {:.2}".format(modelname, acc))
    # Plot normalized confusion matrix
    cnf_matrix = sklearn.metrics.confusion_matrix(labels, class_preds)
    plot_confusion_matrix(cnf_matrix, classes=affectnet.AffectNet.classes, normalize=True, ax=ax,
                          title="'{}.mdl' ACC={:.3}".format(modelname, acc))
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        vmax = 1.0
    else:
        print('Confusion matrix, without normalization')
        vmax = None

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    font_labels = {'fontsize': 'small', 'fontweight':'normal'}

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmax=vmax, aspect='auto')
    # ax.set_title(title, fontdict={'fontsize': 10, 'fontweight':'bold'})
    ax.set_title(title, fontdict={'fontsize': 'small', 'fontweight':'bold'})
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, fontdict=font_labels)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontdict=font_labels)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontdict=font_labels)

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return im


def plot_tsne(X, Y, gt_Y, train_X, train_Y, filenames_val):
    fig = plt.figure()
    cmap = get_cmap(len(expressionNames) + 1)

    ax2 = fig.add_subplot(111)
    scs = []
    legendNames = []
    for id in np.unique(train_Y):
        idx = np.where(train_Y == id)[0]
        sc = ax2.scatter(train_X[idx, 1], train_X[idx, 0], color=cmap(id), alpha=0.4, s=75.0, linewidths=0)
        scs.append(sc)

    for id in np.unique(Y):
        print(len(np.where((Y == id))[0]))
        idx_correct = np.where((Y == id) & (Y == gt_Y))[0]
        idx_incorrect = np.where((Y == id) & (Y != gt_Y))[0]
        sc = ax2.scatter(X[idx_correct, 1], X[idx_correct, 0], color=cmap(id), marker='s', edgecolor='k')
        scs.append(sc)
        ax2.scatter(X[idx_incorrect, 1], X[idx_incorrect, 0], color=cmap(id), marker='v', edgecolor='k')
        expr = expressionNames[id]
        legendNames.append("{} ({})".format(expr, str(len(idx_correct)+len(idx_incorrect))))
    plt.legend(scs, legendNames, scatterpoints=1, loc='lower left', ncol=4, fontsize=7)

    if True:
        for nImg, img_file in enumerate(filenames_val):
            is_correct = Y[nImg] == gt_Y[nImg]
            img_size = 36
            img_path = os.path.join(affectNetRoot, img_file)
            print("{}: {}".format(nImg, img_path))
            arr_img = cv2.imread(img_path)
            arr_img = cv2.cvtColor(arr_img, cv2.COLOR_BGR2RGB)
            arr_img = arr_img.astype(np.float)/255.0
            arr_img = cv2.resize(arr_img, (img_size,img_size), interpolation=cv2.INTER_CUBIC)
            cv2.circle(arr_img, (img_size-6, 6), 3, (0,255,0) if is_correct else (255,0,0), -1)

            border_size = 2
            da = DrawingArea(img_size+2*border_size, img_size+2*border_size, 0, 0)
            p = Rectangle((0, 0), img_size+2*border_size, img_size+2*border_size, color=cmap(Y[nImg]))
            da.add_artist(p)
            border = AnnotationBbox(da, X[nImg][::-1],
                                #xybox=(120., -80.),
                                xybox=(0., 0.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.0,
                                arrowprops=dict(arrowstyle="->",
                                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                )
            ax2.add_artist(border)

            im = OffsetImage(arr_img, interpolation='gaussian')
            ab = AnnotationBbox(im, X[nImg][::-1],
                                #xybox=(120., -80.),
                                xybox=(0., 0.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.0,
                                arrowprops=dict(arrowstyle="->",
                                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                )
            ax2.add_artist(ab)

def plot_tsne_regression(X, Y, C, data, C2=None, title=None, dataset=None, filenames=None, normalize_y=True, img_size=40):
    if normalize_y and Y is not None:
        Y -= Y.min()
        Y /= Y.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # sns.scatterplot(X[:, 1], X[:, 0], style=[AffectNet.markers[i] for i in C2 % len(AffectNet.markers)])
    # print(C2)
    # print(np.unique(C2))
    # if C2 is not None:
    #     C2 = C2%2

    # sns.scatterplot(X[:, 1], X[:, 0],  hue=C, style=C2,
                    # palette=sns.color_palette("Set1", n_colors=len(np.unique(C))))  # 'husl'
                    # palette=sns.color_palette("hls", n_colors=len(np.unique(C))))  # 'husl'

    sns.scatterplot('x', 'y', hue='Expression', data=data, palette=affectnet.AffectNet.colors)

    if False:
        plt.figure()
        for id in np.unique(C):
            idx = np.where(C == id)[0]
            if Y is not None:
                sc = ax.scatter(X[idx, 1], X[idx, 0], c=Y[idx], alpha=0.9, linewidths=0,
                                # marker=AffectNet.markers[id % len(AffectNet.markers)],
                                vmin=-1.0, vmax=1.0, cmap=plt.cm.RdYlGn)
            else:
                # sc = ax.scatter(X[idx, 1], X[idx, 0], alpha=0.9, s=75.0,
                #                 marker=[AffectNet.markers[i] for i in C2[idx] % len(AffectNet.markers)],
                #                 linewidths=0)#, color=AffectNet.colors[id])
                if C2 is not None:
                    # sns.scatterplot(X[idx, 1], X[idx, 0], style=[AffectNet.markers[i] for i in C2[idx] % len(AffectNet.markers)], label=id)
                    sns.scatterplot(X[idx, 1], X[idx, 0])
                else:
                    # sns.scatterplot(X[idx, 1], X[idx, 0], style=[AffectNet.markers[id % len(AffectNet.markers)] for i in idx])
                    ax.scatter(X[idx, 1], X[idx, 0], marker=AffectNet.markers[id % len(AffectNet.markers)])
                # print(id, np.unique(C2[idx]))
            # legendNames.append("{} ({})".format(AffectNet.classes[id], str(len(idx))))
            # scs.append(sc)

    # for id in np.unique(Y):
    #     print len(np.where((Y == id))[0])
    #     idx_correct = np.where((Y == id) & (Y == gt_Y))[0]
    #     idx_incorrect = np.where((Y == id) & (Y != gt_Y))[0]
    #     sc = ax.scatter(X[idx_correct, 1], X[idx_correct, 0], color=cmap(id), marker='s', edgecolor='k')
    #     scs.append(sc)
    #     ax.scatter(X[idx_incorrect, 1], X[idx_incorrect, 0], color=cmap(id), marker='v', edgecolor='k')
    #     expr = expressionNames[id]
    #     legendNames.append("{} ({})".format(expr, str(len(idx_correct)+len(idx_incorrect))))
    # plt.legend(scs, legendNames, scatterpoints=1, loc='lower left', ncol=4, fontsize=7)

    if dataset is not None and img_size > 0:
        add_faces_to_scatterplot(ax, X, filenames, dataset, img_size, C)



def add_faces_to_scatterplot(ax, X, filenames, dataset, img_size, C=None, border_size=3):
    nx, ny = (16, 10)
    x = np.linspace(X[:, 0].min(), X[:, 0].max(), nx)
    y = np.linspace(X[:, 1].min(), X[:, 1].max(), ny)
    xx,yy = np.meshgrid(x, y)
    coords =  np.hstack([xx.reshape(-1,1), yy.reshape(-1,1)])
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors().fit(X)

    max_dist = np.abs(X[:, 0].min() - X[:, 0].max()) / nx

    faces_to_draw = []
    for coord in coords:
        dists, nbs = nn.kneighbors(np.atleast_2d(coord))
        if dists[0][0] < max_dist/2:
            faces_to_draw.append(nbs[0][0])

    # for nImg, img_file in enumerate(filenames):
        # if (nImg % every_n_img) != 0:
        #     continue
    for nImg, img_file in enumerate(filenames):
        if nImg not in faces_to_draw:
            continue

        arr_img = dataset.get_face(img_file, size=(img_size,img_size))[0]

        if C is not None:
            print(nImg, affectnet.AffectNet.classes[C[nImg]])
            da = DrawingArea(img_size+2*border_size, img_size+2*border_size, 0, 0)
            p = Rectangle((0, 0), img_size + 2 * border_size, img_size + 2 * border_size, color=AffectNet.colors[C[nImg]])
            da.add_artist(p)
            border = AnnotationBbox(da, X[nImg][::],
                                #xybox=(120., -80.),
                                xybox=(0., 0.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0.0,
                                arrowprops=dict(arrowstyle="->",
                                                connectionstyle="angle,angleA=0,angleB=90,rad=3")
                                )
            ax.add_artist(border)

        im = OffsetImage(arr_img, interpolation='gaussian')
        ab = AnnotationBbox(im, X[nImg][::],
                            #xybox=(120., -80.),
                            xybox=(0., 0.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.0,
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="angle,angleA=0,angleB=90,rad=3"),
                            )
        ax.add_artist(ab)



def plot_depthbuffers(features, labels):
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    for exprId in range(8):
        idx = np.where(labels == exprId)[0]
        ax.plot(np.mean(features[idx,:], axis=0), color=AffectNet.colors[exprId])
    plt.legend(AffectNet.classes)


def show_imgs(bumps):
    cmap=plt.cm.viridis
    fig, axes = plt.subplots(nrows=4, ncols=5)
    for bm_,ax in zip(bumps, axes.flat):
        im = ax.imshow(bm_, interpolation='nearest', cmap=cmap)
    fig.colorbar(im, ax=axes.ravel().tolist())


def show_bumps(bumps):
    cmap=plt.cm.viridis
    fig, axes = plt.subplots(nrows=4, ncols=5)
    for bm_,ax in zip(bumps, axes.flat):
        im = ax.imshow(bm_, interpolation='nearest', cmap=cmap, vmin=-0.0, vmax=1.0)
    fig.colorbar(im, ax=axes.ravel().tolist())

    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    for bm in bumps:
        im_mean = ax.imshow(bm, interpolation='nearest', cmap=cmap, alpha=1.0/float(len(bumps)))
    fig2.colorbar(im_mean)


def show_expression_means(bumps, labels):
    arr_bumps = np.array(bumps)
    cmap=plt.cm.viridis
    fig, axes = plt.subplots(nrows=2, ncols=4)
    for eId,ax in zip(np.unique(labels), axes.flat):
        exprName = AffectNet.classes[eId]
        idx = np.where(labels == eId)[0]
        im_mean = ax.imshow(np.mean(arr_bumps[idx,:], axis=0), interpolation='nearest', cmap=cmap)
        ax.set_title(exprName)
    fig.colorbar(im_mean, ax=axes.ravel().tolist())


def color_map(data, vmin=None, vmax=None, cmap=plt.cm.viridis):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    val = np.maximum(vmin, np.minimum(vmax, data))
    norm = (val-vmin)/(vmax-vmin)
    cm = cmap(norm)
    if isinstance(cm, tuple):
        return cm[:3]
    if len(cm.shape) > 2:
        cm = cm[:,:,:3]
    return cm


# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def make_grid(data, padsize=2, padval=1, nCols=10, dsize=None, fx=None, fy=None, normalize=False):
    # if not isinstance(data, np.ndarray):
    data = np.array(data)
    if data.shape[0] == 0:
        return
    if data.shape[1] == 3:
        data = data.transpose((0,2,3,1))
    data = data.astype(np.float32)
    if normalize:
        data -= data.min()
        data /= data.max()
    else:
        data[data < 0] = 0
    #     data[data > 1] = 1

    # force the number of filters to be square
    # n = int(np.ceil(np.sqrt(data.shape[0])))
    c = nCols
    r = int(np.ceil(data.shape[0]/float(c)))

    padding = ((0, r*c - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((r, c) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((r * data.shape[1], c * data.shape[3]) + data.shape[4:])

    if dsize is not None or fx is not None or fy is not None:
        data = cv2.resize(data, dsize=dsize, fx=fx, fy=fy, interpolation=cv2.INTER_LANCZOS4)

    return data


def vis_square(data, padsize=1, padval=0, wait=0, nCols=10, title='results', dsize=None, fx=None, fy=None, normalize=False):
    img = make_grid(data, padsize=padsize, padval=padval, nCols=nCols, dsize=dsize, fx=fx, fy=fy, normalize=normalize)
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)


def to_disp_image(img, denorm=False):
    if not isinstance(img, np.ndarray):
        img = img.detach().cpu().numpy()
    img = img.astype(np.float32).copy()
    if img.shape[0] == 3:
        img = img.transpose((1, 2, 0)).copy()
    if denorm:
        img = denormalized(img)
    if img.max() > 2.00:
        raise ValueError("Image data in wrong value range (min/max={:.2f}/{:.2f}).".format(img.min(), img.max()))
    return img

def _to_disp_images(images, denorm=False):
    return [to_disp_image(i, denorm) for i in images]


def add_frames_to_images(images, labels, label_colors, gt_labels=None):
    import collections
    if not isinstance(labels, (collections.Sequence, np.ndarray)):
        labels = [labels] * len(images)
    new_images = _to_disp_images(images)
    for idx, (disp, label) in enumerate(zip(new_images, labels)):
        frame_width = 3
        bgr = label_colors[label]
        cv2.rectangle(disp,
                      (frame_width // 2, frame_width // 2),
                      (disp.shape[1] - frame_width // 2, disp.shape[0] - frame_width // 2),
                      color=bgr,
                      thickness=frame_width)

        if gt_labels is not None:
            radius = 8
            color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
            cv2.circle(disp, (disp.shape[1] - 2*radius, 2*radius), radius, color, -1)
    return new_images


def add_cirle_to_images(images, intensities, cmap=plt.cm.viridis, radius=10):
    new_images = _to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, intensities)):
        # color = (0, 1, 0) if gt_labels[idx] == label else (1, 0, 0)
        # color = plt_colors.to_rgb(val)
        if isinstance(val, float):
            color = cmap(val).ravel()
        else:
            color = val
        cv2.circle(disp, (2*radius, 2*radius), radius, color, -1)
        # new_images.append(disp)
    return new_images


def get_pos_in_image(loc, text_size, image_shape):
    bottom_offset = int(6*text_size)
    right_offset = int(95*text_size)
    line_height = int(35*text_size)
    mid_offset = right_offset
    top_offset = line_height + int(0.05*line_height)
    if loc == 'tl':
        pos = (2, top_offset)
    elif loc == 'tr':
        pos = (image_shape[1]-right_offset, top_offset)
    elif loc == 'tr+1':
        pos = (image_shape[1]-right_offset, top_offset + line_height)
    elif loc == 'tr+2':
        pos = (image_shape[1]-right_offset, top_offset + line_height*2)
    elif loc == 'bl':
        pos = (2, image_shape[0]-bottom_offset)
    elif loc == 'bl-1':
        pos = (2, image_shape[0]-bottom_offset-line_height)
    elif loc == 'bl-2':
        pos = (2, image_shape[0]-bottom_offset-2*line_height)
    elif loc == 'bm':
        pos = (mid_offset, image_shape[0]-bottom_offset)
    elif loc == 'bm-1':
        pos = (mid_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset)
    elif loc == 'br-1':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-line_height)
    elif loc == 'br-2':
        pos = (image_shape[1]-right_offset, image_shape[0]-bottom_offset-2*line_height)
    else:
        raise ValueError("Unknown location {}".format(loc))
    return pos


def add_id_to_images(images, ids, gt_ids=None, loc='tl', color=(1,1,1), size=0.7, thickness=1):
    new_images = _to_disp_images(images)
    for idx, (disp, val) in enumerate(zip(new_images, ids)):
        if gt_ids is not None:
            color = (0,1,0) if ids[idx] == gt_ids[idx] else (1,0,0)
        # if val != 0:
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, str(val), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_error_to_images(images, errors, loc='bl', size=0.65, vmin=0., vmax=30.0, thickness=1, format_string='{:.1f}'):
    new_images = _to_disp_images(images)
    colors = color_map(to_numpy(errors), cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
    for disp, err, color in zip(new_images, errors, colors):
        pos = get_pos_in_image(loc, size, disp.shape)
        cv2.putText(disp, format_string.format(err), pos, cv2.FONT_HERSHEY_DUPLEX, size, color, thickness, cv2.LINE_AA)
    return new_images


def add_pose_to_images(images, poses, color=(1.0, 0, 0)):
    from utils import vis3d
    new_images = _to_disp_images(images)
    # return new_images # FIXME
    for disp, pose in zip(new_images, poses):
        vis3d.draw_head_pose(disp, pose, color)
    return new_images


def add_landmarks_to_images(images, landmarks, color=None, radius=2, gt_landmarks=None, lm_errs=None, lm_confs=None,
                            draw_dots=True, draw_wireframe=False, draw_gt_offsets=False, landmarks_to_draw=None,
                            offset_line_color=None):
    def draw_wireframe_lines(img, lms):
        pts = lms.reshape((-1,1,2)).astype(np.int32)
        cv2.polylines(img, [pts[:17]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # head outline
        cv2.polylines(img, [pts[17:22]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # left eyebrow
        cv2.polylines(img, [pts[22:27]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # right eyebrow
        cv2.polylines(img, [pts[27:31]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose vert
        cv2.polylines(img, [pts[31:36]], isClosed=False, color=color, lineType=cv2.LINE_AA)  # nose hor
        cv2.polylines(img, [pts[36:42]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # left eye
        cv2.polylines(img, [pts[42:48]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # right eye
        cv2.polylines(img, [pts[48:60]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # outer mouth
        cv2.polylines(img, [pts[60:68]], isClosed=True, color=color, lineType=cv2.LINE_AA)  # inner mouth

    def draw_offset_lines(img, lms, gt_lms, errs):
        if lm_errs is None:
            # if offset_line_color is None:
            offset_line_color = (1,1,1)
            colors = [offset_line_color] * len(lms)
        else:
            colors = color_map(errs, cmap=plt.cm.jet, vmin=0, vmax=15.0)
        for i, (p1, p2) in enumerate(zip(lms, gt_lms)):
            if landmarks_to_draw is None or i in landmarks_to_draw:
                if p1.min() > 0:
                    cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), colors[i], thickness=1, lineType=cv2.LINE_AA)

    new_images = _to_disp_images(images)
    landmarks = to_numpy(landmarks)
    gt_landmarks = to_numpy(gt_landmarks)
    lm_errs = to_numpy(lm_errs)
    img_size = new_images[0].shape[0]
    # default_color = (0,1,0)  # green
    default_color = (1,1,1)

    if gt_landmarks is not None and draw_gt_offsets:
        for img_id  in range(len(new_images)):
            dists = None
            if lm_errs is not None:
                dists = lm_errs[img_id]
            draw_offset_lines(new_images[img_id], landmarks[img_id], gt_landmarks[img_id], dists)

    for img_id, (disp, lm)  in enumerate(zip(new_images, landmarks)):
        if len(lm) in [68, 21, 19, 98]:
            if draw_dots:
                for lm_id in range(0,len(lm)):
                    if landmarks_to_draw is None or lm_id in landmarks_to_draw or len(lm) != 68:
                        lm_color = color
                        if lm_color is None:
                            if lm_errs is not None:
                                lm_color = color_map(lm_errs[img_id, lm_id], cmap=plt.cm.jet, vmin=0, vmax=1.0)
                            else:
                                lm_color = default_color
                        # if lm_errs is not None and lm_errs[img_id, lm_id] > 40.0:
                        #     lm_color = (1,0,0)
                        cv2.circle(disp, tuple(lm[lm_id].astype(int)), radius=1, color=lm_color, thickness=1, lineType=cv2.LINE_AA)
                        if lm_confs is not None:
                            max_radius = img_size * 0.025
                            try:
                                radius = max(2, int((1-lm_confs[img_id, lm_id]) * max_radius))
                            except ValueError:
                                radius = 2
                            # if lm_confs[img_id, lm_id] > 0.4:
                            cirle_color = (0,0,1)
                            if lm_confs[img_id, lm_id] < lmcfg.MIN_LANDMARK_CONF:
                                cirle_color = (1,0,0)
                            cv2.circle(disp, tuple(lm[lm_id].astype(int)), radius, cirle_color, 1, lineType=cv2.LINE_AA)

            # Draw outline if we actually have 68 valid landmarks.
            # Landmarks can be zeros for UMD landmark format (21 points).
            nlms = (np.count_nonzero(lm.sum(axis=1)))
            if nlms == 68:
                if draw_wireframe:
                    draw_wireframe_lines(disp, lm)
        else:
            # colors = ['tab:gray', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:cyan', 'tab:olive', 'tab:red', 'tab:blue']
            # colors_rgb = list(map(plt_colors.to_rgb, colors))

            colors = sns.color_palette("Set1", n_colors=10)
            for i in range(0,len(lm)):
                cv2.circle(disp, tuple(lm[i].astype(int)), radius=radius, color=colors[i], thickness=2, lineType=cv2.LINE_AA)
    return new_images


def add_emotion_to_images(images, emotions, gt_emotions=None):
    from datasets.emotiw import EmotiW
    return add_frames_to_images(images, emotions, label_colors=EmotiW.colors_rgb, gt_labels=gt_emotions)


def show_landmarks(img_in, landmarks, bbox=None, gt=None,  title='landmarks', pose=None, wait=10, color=(1, 0, 0)):
    from landmarks.lmutils import calc_landmark_nme_per_img
    img = img_in.copy()
    if img.max() > 1.01:
        img = img.astype(np.float32)/255.0
    if bbox is not None:
        tl = tuple([int(v) for v in bbox[:2]])
        br = tuple([int(v) for v in bbox[2:]])
        cv2.rectangle(img, tl, br, (1,1,1))
    if gt is not None:
        for lm in gt:
            lm_x, lm_y = lm[0], lm[1]
            cv2.circle(img, (int(lm_x), int(lm_y)), 1, (1, 1, 0), -1, lineType=cv2.LINE_AA)
    for lm in landmarks:
        lm_x, lm_y = lm[0], lm[1]
        cv2.circle(img, (int(lm_x), int(lm_y)), 2, color, -1, lineType=cv2.LINE_AA)
    if pose is not None:
        from utils import vis3d
        vis3d.draw_head_pose(img, pose, color=(1.0,1.0,1.0))
    # if img.shape[0] > 800:
    #     img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    if gt is not None:
        lm_err = calc_landmark_nme_per_img(gt, landmarks, ocular_norm='outer')
        img = add_error_to_images([img], lm_err)[0]
    cv2.imshow(title, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(wait)


def show_images_in_batch(images, labels):
    from datasets.emotiw import EmotiW
    disp_imgs = to_numpy(images)
    if labels is not None:
        labels = to_numpy(labels)
        disp_imgs = add_frames_to_images(images, labels, label_colors=EmotiW.colors_rgb)
    vis_square(disp_imgs, fx=0.4, fy=0.4)


def plot_va_scatter(ax, pv, pa, tv=None, ta=None, title='', color=None, with_gt=False, class_preds=None):
    X1 = tv
    Y1 = ta
    X2 = pv
    Y2 = pa

    # plot ground truth, if available
    if tv is not None and ta is not None:
        for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
            ax.plot([x1, x2], [y1, y2], color='k', alpha=0.5, zorder=0, linewidth=1.0)
            # ax.arrow(x1,y1,x2-x1,y2-y1, length_includes_head=True, width=0.0001, head_width=0.02, head_length=0.02, color='k', alpha=0.5)
        ax.scatter(X1, Y1, marker='*', s=10, color='g', alpha=0.75)

    c = color
    if class_preds is not None:
        c = [affectnet.AffectNet.colors[cl] for cl in class_preds]
    sc = ax.scatter(X2, Y2, marker='.', color=c, s=150)

    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    # ax.set_xlabel('Valence')
    # ax.set_ylabel('Arousal')
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('right')
    ax.xaxis.set_label_position('top')
    ax.set_title(title, pad=25.)
    return sc


def plot_valence_arousal(v, a):
    fig, ax = plt.subplots()
    plot_va_scatter(ax, v, a)


def draw_z(z_vecs):

    fy = 1
    width = 10
    z_zoomed = []
    for lvl, _ft in enumerate(to_numpy(z_vecs)):
        # _ft = (_ft-_ft.min())/(_ft.max()-_ft.min())
        vmin = 0 if lvl == 0 else -1

        canvas = np.zeros((int(fy*len(_ft)), width, 3))
        canvas[:int(fy*len(_ft)), :] = color_map(cv2.resize(_ft.reshape(-1,1), dsize=(width, int(fy*len(_ft))),
                                                            interpolation=cv2.INTER_NEAREST), vmin=-1.0, vmax=1.0)
        z_zoomed.append(canvas)
    return make_grid(z_zoomed, nCols=len(z_vecs), padsize=1, padval=0).transpose((1,0,2))


def overlay_heatmap(img, hm):
    img_new = img.copy()
    hm_colored = color_map(hm**3, vmin=0, vmax=1.0, cmap=plt.cm.inferno)
    mask = cv2.blur(hm, ksize=(3, 3))
    if len(mask.shape) > 2:
        mask = mask.mean(axis=2)
        mask = mask > 0.05
        for c in range(3):
            # img_new[...,c] = img[...,c] + hm[...,c]
            img_new[..., c][mask] = img[..., c][mask] * 0.7 + hm[..., c][mask] * 0.3
    else:
        # mask = mask > 0.05
        # img_new[mask] = img[mask] * 0.7 + hm_col[mask] * 0.3
        heatmap_opacity = 0.9
        img_new = img + hm_colored * heatmap_opacity
    img_new = img_new.clip(0, 1)
    return img_new


def draw_z_vecs(levels_z, ncols, size, vmin=-1, vmax=1, vertical=False):
    z_fy = 1.0
    width, height = [int(x) for x in size[:2]]
    def draw_z(z):
        z_zoomed = []
        for lvl, ft in enumerate(z):
            _ft = ft[:]
            # _ft = (_ft-_ft.min())/(_ft.max()-_ft.min())
            # canvas = np.zeros((height, width, 3))
            if vertical:
                _ft_reshaped = _ft.reshape(-1, 1)
            else:
                _ft_reshaped = _ft.reshape(1, -1)

            canvas = color_map(
                cv2.resize(_ft_reshaped, dsize=(width, height), interpolation=cv2.INTER_NEAREST),
                vmin=vmin,
                vmax=vmax
            )

            z_zoomed.append(canvas)
        return z_zoomed

    # pivots not used anymore FIXME: remove
    def draw_pivot(z_imgs, pivot):
        z_imgs_new = _to_disp_images(z_imgs)
        for new_img in z_imgs_new:
            y = int(pivot*z_fy)
            cv2.line(new_img, (0, y), (new_img.shape[1], y), (1, 1, 1), thickness=1)
        return z_imgs_new

    # pivots = [z.shape[1] for z in levels_z if z is not None]
    # z_vis_list_per_level = [draw_pivot(draw_z(z), p) for z,p in zip(levels_z, pivots) if z is not None]
    z_vis_list_per_level = [draw_z(z) for z in levels_z if z is not None]
    z_grid_per_sample = [make_grid(all_vis_sample, nCols=len(levels_z)) for all_vis_sample in zip(*z_vis_list_per_level)]
    return make_grid(z_grid_per_sample, nCols=ncols, normalize=False)


def draw_status_bar(text, status_bar_width, status_bar_height, dtype=np.float32, text_size=-1, text_color=(1,1,1)):
    img_status_bar = np.zeros((status_bar_height, status_bar_width, 3), dtype=dtype)
    if text_size <= 0:
        text_size = status_bar_height * 0.025
    cv2.putText(img_status_bar, text, (4,img_status_bar.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 1, cv2.LINE_AA)
    return img_status_bar


def draw_results(X_resized, X_recon, levels_z=None,
                 ids=None, ids_pred=None,
                 poses=None, poses_pred=None,
                 emotions=None, emotions_pred=None,
                 landmarks=None, landmarks_pred=None,
                 cs_errs=None,
                 ncols=15, fx=0.5, fy=0.5,
                 additional_status_text=''):
    from skimage.measure import compare_ssim
    # import torch

    clean_images = False
    if clean_images:
        ids = None
        poses=None
        poses_pred=None
        emotions=None
        emotions_pred=None
        landmarks=None

    nimgs = len(X_resized)
    ncols = min(ncols, nimgs)
    img_size = X_recon.shape[-1]

    disp_X = _to_disp_images(X_resized, denorm=True)
    disp_X_recon = _to_disp_images(X_recon, denorm=True)

    # reconstruction errors
    l1_dists = 255.0 * np.abs(np.stack(disp_X) - np.stack(disp_X_recon)).reshape(len(disp_X), -1).mean(axis=1)
    # l1_dists = 255.0 * to_numpy(torch.abs(X_resized - X_recon).reshape(len(disp_X), -1).mean(dim=1))

    # SSIM errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        ssim[i] = compare_ssim(disp_X[i], disp_X_recon[i], data_range=1.0, multichannel=True)

    ids = to_numpy(ids)
    poses = to_numpy(poses)
    emotions = to_numpy(emotions)
    landmarks = to_numpy(landmarks)
    cs_errs = to_numpy(cs_errs)

    text_size = img_size/256
    text_thickness = 2

    #
    # Visualise resized input images and reconstructed images
    #
    if ids is not None:
        disp_X = add_id_to_images(disp_X, ids)
    if ids_pred is not None:
        disp_X_recon = add_id_to_images(disp_X_recon, ids_pred, gt_ids=ids)

    if poses is not None:
        disp_X = add_pose_to_images(disp_X, poses, color=(0, 0, 0.5))
    if poses_pred is not None:
        if poses is not None:
            disp_X_recon = add_pose_to_images(disp_X_recon, poses, color=(0, 0, 0.5))
        disp_X_recon = add_pose_to_images(disp_X_recon, poses_pred, color=(0.5, 0, 0))

    if emotions is not None:
        disp_X = add_emotion_to_images(disp_X, emotions)
    if emotions_pred is not None:
        disp_X_recon = add_emotion_to_images(disp_X_recon, emotions_pred, gt_emotions=emotions)

    if landmarks is not None:
        disp_X = add_landmarks_to_images(disp_X, landmarks, draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS_19)
        disp_X_recon = add_landmarks_to_images(disp_X_recon, landmarks, draw_wireframe=False, landmarks_to_draw=lmcfg.LANDMARKS_19)

    if landmarks_pred is not None:
        disp_X = add_landmarks_to_images(disp_X, landmarks_pred, color=(1, 0, 0))
        disp_X_recon = add_landmarks_to_images(disp_X_recon, landmarks_pred, color=(1, 0, 0))


    if not clean_images:
        disp_X_recon = add_error_to_images(disp_X_recon, l1_dists, format_string='{:.1f}',
                                           size=text_size, thickness=text_thickness)
        disp_X_recon = add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                           vmax=0.8, vmin=0.2, size=text_size, thickness=text_thickness)
        if cs_errs is not None:
            disp_X_recon = add_error_to_images(disp_X_recon, cs_errs, loc='bl-2', format_string='{:>4.2f}',
                                               vmax=0.0, vmin=0.4, size=text_size, thickness=text_thickness)


    lm_errs = np.zeros(0)

    img_input = make_grid(disp_X, nCols=ncols, normalize=False)
    img_recon = make_grid(disp_X_recon, nCols=ncols, normalize=False)
    img_input = cv2.resize(img_input, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    img_recon = cv2.resize(img_recon, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    img_stack = [img_input, img_recon]

    #
    # Visualise hidden layers
    #
    VIS_HIDDEN = True
    if VIS_HIDDEN:
        img_z = draw_z_vecs(levels_z, size=(img_size, 30), ncols=ncols)
        img_z = cv2.resize(img_z, dsize=(img_input.shape[1], img_z.shape[0]), interpolation=cv2.INTER_NEAREST)
        img_stack.append(img_z)

    #
    # Write errors to status bar
    #
    try:
        acc = 100.0 * np.sum(ids_pred == ids) / float(len(ids))
    except:
        acc = np.nan

    try:
        pose_errs = np.rad2deg(np.abs(poses_pred - poses)).mean(axis=1)
    except:
        pose_errs = [np.nan]*3

    cs_errs_mean = np.mean(cs_errs) if cs_errs is not None else np.nan
    status_bar_text = ("l1 recon err: {:.2f}px  ssim: {:.3f}({:.3f})  "
                       # "pose err: {:.2f}/{:.2f}/{:.2f} deg  acc: {:.2f}%  " 
                       "lms err: {:2} {}").format(
        l1_dists.mean(),
        cs_errs_mean,
        1 - ssim.mean(),
        # pose_errs[0], pose_errs[1], pose_errs[2],
        # acc,
        lm_errs.mean(),
        additional_status_text
    )

    img_status_bar = draw_status_bar(status_bar_text,
                                     status_bar_width=img_input.shape[1],
                                     status_bar_height=30,
                                     dtype=img_input.dtype)
    img_stack.append(img_status_bar)

    return np.vstack(img_stack)
