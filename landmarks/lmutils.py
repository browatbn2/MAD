import numpy as np
import config as cfg
import landmarks.lmconfig as lmcfg
from utils.nn import to_numpy, to_image
from utils import vis
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
import torch
from datasets import ds_utils

layers = []

# outline_l = range(0, 6)
# outline_m = range(6, 11)
# outline_r = range(11, 17)
outline = range(0, 17)
eyebrow_l = range(17, 22)
eyebrow_r = range(22, 27)
nose = range(27, 31)
nostrils = range(31, 36)
eye_l = range(36, 42)
eye_r = range(42, 48)
mouth = range(48, 68)
# mouth_outer = range(48, 60)
# mouth_inner = range(60, 68)


# components = [outline_l, outline_m, outline_r, eyebrow_l, eyebrow_r, nose, nostrils, eye_l, eye_r, mouth_outer, mouth_inner, outline]
components = [outline, eyebrow_l, eyebrow_r, nose, nostrils, eye_l, eye_r, mouth]
# layers = np.array(layers)

new_layers = []
for idx in range(20):
    lm_ids = []
    for comp in components[1:]:
        if len(comp) > idx:
            lm = comp[idx]
            lm_ids.append(lm)
    new_layers.append(lm_ids)

outline_layers = [[lm] for lm in range(17)]

layers = components + new_layers + outline_layers

hm_code_mat = np.zeros((len(layers), 68), dtype=bool)
for l, lm_ids in enumerate(layers):
    hm_code_mat[l, lm_ids] = True


def generate_colors(n, r, g, b, dim):
    ret = []
    step = [0,0,0]
    step[dim] =  256 / n
    for i in range(n):
        r += step[0]
        g += step[1]
        b += step[2]
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r,g,b))
    return ret

_colors = generate_colors(17, 220, 0, 0, 2) + \
          generate_colors(10, 0, 240, 0, 0) + \
          generate_colors(9, 0, 0, 230, 1) + \
          generate_colors(12, 100, 255, 0, 2) + \
          generate_colors(20, 150, 0, 255, 2)
# lmcolors = np.array(_colors)
np.random.seed(0)
lmcolors = np.random.randint(0,255,size=(68,3))
lmcolors = lmcolors / lmcolors.sum(axis=1).reshape(-1,1)*255


def gaussian(x, mean, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x-mean)**2 / (2 * sigma**2))


def make_landmark_template(wnd_size, sigma):
    X, Y = np.mgrid[-wnd_size//2:wnd_size//2, -wnd_size//2:wnd_size//2]
    Z = np.sqrt(X**2 + Y**2)
    N = gaussian(Z, 0, sigma)
    # return (N/N.max())**2  # square to make sharper
    return (N/N.max())


def _fill_heatmap_layer(dst, lms, lm_id, lm_heatmap_window, wnd_size):
    posx, posy = min(lms[lm_id,0], cfg.INPUT_SIZE-1), min(lms[lm_id,1], cfg.INPUT_SIZE-1)

    img_size = cfg.INPUT_SIZE
    l = int(posx - wnd_size/2)
    t = int(posy - wnd_size/2)
    r = l + wnd_size
    b = t + wnd_size

    src_l = max(0, -l)
    src_t = max(0, -t)
    src_r = min(wnd_size, wnd_size-(r-img_size))
    src_b = min(wnd_size, wnd_size-(b-img_size))

    try:
        # dst[max(0,t):min(img_size, b), max(0,l):min(img_size, r)] = Batch.lm_heatmap_target[src_t:src_b, src_l:src_r]

        # if lmcfg.LANDMARK_TARGET == 'multi_channel':
            cn = lmcfg.LANDMARK_ID_TO_HEATMAP_ID[lm_id]


            wnd = lm_heatmap_window[src_t:src_b, src_l:src_r]

            weight = 1.0
            # if lm_id < 17:
            #     weight = 1.0
            dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)] = np.maximum(
                dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)], wnd*weight)
        # else:
        #     for cn in range(3):
        #         dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)] = np.maximum(
        #             dst[cn, max(0,t):min(img_size, b), max(0,l):min(img_size, r)],
        #             lm_heatmap_window[src_t:src_b, src_l:src_r] * 1# lmcolors[lm_id, cn]
        #         )
    except:
        pass


def __get_code_mat(num_landmarks):
    def to_binary(n, ndigits):
        bits =  np.array([bool(int(x)) for x in bin(n)[2:]])
        assert len(bits) <= ndigits
        zero_pad_bits = np.zeros(ndigits, dtype=bool)
        zero_pad_bits[-len(bits):] = bits
        return zero_pad_bits

    n_enc_layers = int(np.ceil(np.log2(num_landmarks)))

    # get binary code for each heatmap id
    codes = [to_binary(i+1, ndigits=n_enc_layers) for i in range(num_landmarks)]
    return np.vstack(codes)

def convert_to_encoded_heatmaps(hms):

    def merge_layers(hms):
        hms = hms.max(axis=0)
        return hms/hms.max()

    num_landmarks = len(hms)
    n_enc_layers = len(hm_code_mat)

    # create compressed heatmaps by merging layers according to transpose of binary code mat
    encoded_hms = np.zeros((n_enc_layers, hms.shape[1], hms.shape[2]))
    for l in range(n_enc_layers):
        selected_layer_ids = hm_code_mat[l,:]
        encoded_hms[l] = merge_layers(hms[selected_layer_ids].copy())
    decode_heatmaps(encoded_hms)

    return encoded_hms

def convert_to_hamming_encoded_heatmaps(hms):

    def merge_layers(hms):
        hms = hms.max(axis=0)
        return hms/hms.max()

    num_landmarks = len(hms)
    n_enc_layers = int(np.ceil(np.log2(num_landmarks)))
    code_mat = __get_code_mat(num_landmarks)

    # create compressed heatmaps by merging layers according to transpose of binary code mat
    encoded_hms = np.zeros((n_enc_layers, hms.shape[1], hms.shape[2]))
    for l in range(n_enc_layers):
        selected_layer_ids = code_mat[:, l]
        encoded_hms[l] = merge_layers(hms[selected_layer_ids].copy())
    # decode_heatmaps(encoded_hms)

    return encoded_hms


def decode_heatmap_blob(hms):
    assert len(hms.shape) == 4
    if hms.shape[1] == 68: # no decoding necessary
        return hms
    assert hms.shape[1] == len(hm_code_mat)
    hms68 = np.zeros((hms.shape[0], 68, hms.shape[2], hms.shape[3]), dtype=np.float32)
    for img_idx in range(len(hms)):
        hms68[img_idx] = decode_heatmaps(to_numpy(hms[img_idx]))[0]
    return hms68



def decode_heatmaps(hms):
    import cv2
    def get_decoded_heatmaps_for_layer(hms, lm):
        show = False
        enc_layer_ids = code_mat[:, lm]
        heatmap = np.ones_like(hms[0])
        for i in range(len(enc_layer_ids)):
            pos = enc_layer_ids[i]
            layer = hms[i]
            if pos:
                if show:
                    fig, ax = plt.subplots(1,4)
                    print(i, pos)
                    ax[0].imshow(heatmap, vmin=0, vmax=1)
                    ax[1].imshow(layer, vmin=0, vmax=1)
                # mask = layer.copy()
                # mask[mask < 0.1] = 0
                # heatmap *= mask
                heatmap *= layer
                if show:
                    # ax[2].imshow(mask, vmin=0, vmax=1)
                    ax[3].imshow(heatmap, vmin=0, vmax=1)

        return heatmap

    num_landmarks = 68

    # get binary code for each heatmap id
    code_mat = hm_code_mat


    decoded_hms = np.zeros((num_landmarks, hms.shape[1], hms.shape[1]))

    show = False
    if show:
        fig, ax = plt.subplots(1)
        ax.imshow(code_mat)
        fig_dec, ax_dec = plt.subplots(7, 10)
        fig, ax = plt.subplots(5,9)
        for i in range(len(hms)):
            ax[i//9, i%9].imshow(hms[i])

    lms = np.zeros((68,2), dtype=int)
    # lmid_to_show = 16

    for lm in range(0,68):

        heatmap = get_decoded_heatmaps_for_layer(hms, lm)

        decoded_hms[lm] = heatmap
        heatmap = cv2.blur(heatmap, (5, 5))
        lms[lm, :] = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[::-1]

        if show:
            ax_dec[lm//10, lm%10].imshow(heatmap)

    if show:
        plt.show()

    return decoded_hms, lms



def create_landmark_heatmaps(lms, sigma, landmarks_to_use):
    landmark_target = lmcfg.LANDMARK_TARGET
    lm_wnd_size = int(sigma * 5)
    lm_heatmap_window = make_landmark_template(lm_wnd_size, sigma)

    nchannels = len(landmarks_to_use)
    if landmark_target == 'colored':
        nchannels = 3

    hms = np.zeros((nchannels, cfg.INPUT_SIZE, cfg.INPUT_SIZE))
    for l in landmarks_to_use:
        # plane_id = cfg.LANDMARK_ID_TO_HEATMAP_ID[l]
        # create_landmark_heatmap(hms[plane_id], lms, l)
        _fill_heatmap_layer(hms, lms, l, lm_heatmap_window, lm_wnd_size)

    if landmark_target == 'single_channel':
        hms = hms.max(axis=0)
        hms /= hms.max()
    elif landmark_target == 'colored':
        # face_weights = face_weights.max(axis=0)
        hms = hms.clip(0,255)
        hms /= 255
    elif landmark_target == 'hamming':
        # hms = convert_to_hamming_encoded_heatmaps(hms)
        hms = convert_to_encoded_heatmaps(hms)

    # hms = cv2.blur(hms, (15,15))
    # hms *= 2.0
    return hms.astype(np.float32)


def calc_landmark_nme_per_img(gt_lms, pred_lms, ocular_norm='pupil', landmarks_to_eval=None):
    norm_dists = calc_landmark_nme(gt_lms, pred_lms, ocular_norm)
    # norm_dists = np.clip(norm_dists, a_min=None, a_max=40.0)
    if landmarks_to_eval is None:
        landmarks_to_eval = range(norm_dists.shape[1])
    return np.mean(np.array([norm_dists[:,l] for l in landmarks_to_eval]).T, axis=1)


def get_pupil_dists(gt):
    ocular_dists_inner = np.sqrt(np.sum((gt[:, 42] - gt[:, 39])**2, axis=1))
    ocular_dists_outer = np.sqrt(np.sum((gt[:, 45] - gt[:, 36])**2, axis=1))
    return np.vstack((ocular_dists_inner, ocular_dists_outer)).mean(axis=0)


def calc_landmark_nme(gt_lms, pred_lms, ocular_norm='pupil'):
    def reformat(lms):
        lms = to_numpy(lms)
        if len(lms.shape) == 2:
            lms = lms.reshape((1,-1,2))
        return lms
    gt = reformat(gt_lms)
    pred = reformat(pred_lms)
    assert(len(gt.shape) == 3)
    assert(len(pred.shape) == 3)
    if ocular_norm == 'pupil':
        ocular_dists = get_pupil_dists(gt)
    elif ocular_norm == 'outer':
        ocular_dists = np.sqrt(np.sum((gt[:, 45] - gt[:, 36])**2, axis=1))
    elif ocular_norm is None or ocular_norm == 'none':
        ocular_dists = np.ones((len(gt),1)) * 100.0 #* cfg.INPUT_SIZE
    else:
        raise ValueError("Ocular norm {} not defined!".format(ocular_norm))
    norm_dists = np.sqrt(np.sum((gt - pred)**2, axis=2)) / ocular_dists.reshape(len(gt), 1)
    return norm_dists * 100


def calc_landmark_ssim_error(X, X_recon, lms):
    input_images = vis._to_disp_images(X, denorm=True)
    recon_images = vis._to_disp_images(X_recon, denorm=True)
    nimgs = len(input_images)
    nlms = len(lms[0])
    wnd_size = int(cfg.INPUT_SCALE_FACTOR * 11)
    errs = np.zeros((nimgs, nlms), dtype=np.float32)
    for i in range(nimgs):
        S = compare_ssim(input_images[i], recon_images[i], data_range=1.0, multichannel=True, full=True)[1]
        S = S.mean(axis=2)
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(S.shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(S.shape[1]-1, x+wnd_size//2)
            wnd = S[t:b, l:r]
            errs[i, lid] = 1 - wnd.mean()
    return errs

def calc_landmark_cs_error(X, X_recon, lms, torch_ssim, training=False):
    nimgs = len(X)
    nlms = len(lms[0])
    wnd_size = int(cfg.INPUT_SCALE_FACTOR * 15)
    errs = torch.zeros((nimgs, nlms), requires_grad=training).cuda()
    for i in range(len(X)):
        torch_ssim(X[i].unsqueeze(0), X_recon[i].unsqueeze(0))
        cs_map = torch_ssim.cs_map[0].mean(dim=0)
        map_size = cs_map.shape[0]
        margin = (cfg.INPUT_SIZE - map_size) // 2
        S = torch.zeros((cfg.INPUT_SIZE, cfg.INPUT_SIZE), requires_grad=training).cuda()
        S[margin:-margin, margin:-margin] = cs_map
        for lid in range(nlms):
            x = int(lms[i, lid, 0])
            y = int(lms[i, lid, 1])
            t = max(0, y-wnd_size//2)
            b = min(S.shape[0]-1, y+wnd_size//2)
            l = max(0, x-wnd_size//2)
            r = min(S.shape[1]-1, x+wnd_size//2)
            wnd = S[t:b, l:r]
            errs[i, lid] = 1 - wnd.mean()
    return errs



def calc_landmark_recon_error(X, X_recon, lms, return_maps=False, reduction='mean'):
    assert len(X.shape) == 4
    assert reduction in ['mean', 'none']
    X = to_numpy(X)
    X_recon = to_numpy(X_recon)
    mask = np.zeros((X.shape[0], X.shape[2], X.shape[3]), dtype=np.float32)
    radius = cfg.INPUT_SIZE * 0.05
    for img_id in range(len(mask)):
        for lm in lms[img_id]:
            cv2.circle(mask[img_id], (int(lm[0]), int(lm[1])), radius=int(radius), color=1, thickness=-1)
    err_maps = np.abs(X - X_recon).mean(axis=1) * 255.0
    masked_err_maps = err_maps * mask

    debug = False
    if debug:
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(vis.to_disp_image((X * mask[:,np.newaxis,:,:].repeat(3, axis=1))[0], denorm=True))
        ax[1].imshow(vis.to_disp_image((X_recon * mask[:,np.newaxis,:,:].repeat(3, axis=1))[0], denorm=True))
        ax[2].imshow(masked_err_maps[0])
        plt.show()

    if reduction == 'mean':
        err = masked_err_maps.sum() / (mask.sum() * 3)
    else:
        # err = masked_err_maps.mean(axis=2).mean(axis=1)
        err = masked_err_maps.sum(axis=2).sum(axis=1) / (mask.reshape(len(mask), -1).sum(axis=1) * 3)

    if return_maps:
        return err, masked_err_maps
    else:
        return err


def to_single_channel_heatmap(lm_heatmaps):
    if lmcfg.LANDMARK_TARGET == 'colored':
        mc = [to_image(lm_heatmaps[0])]
    elif lmcfg.LANDMARK_TARGET == 'single_channel':
        mc = [to_image(lm_heatmaps[0, 0])]
    else:
        mc = to_image(lm_heatmaps.max(axis=1))
    return mc


#
# Visualizations
#

def show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1.0):

    vmax = gt_heatmaps.max()

    if len(gt_heatmaps[0].shape) == 2:
        gt_heatmaps = [vis.color_map(hm, vmin=0, vmax=vmax, cmap=plt.cm.jet) for hm in gt_heatmaps]
    nCols = 1 if len(gt_heatmaps) == 1 else nimgs
    img_gt_heatmaps = cv2.resize(vis.make_grid(gt_heatmaps, nCols=nCols, padval=0), None, fx=f, fy=f)

    disp_pred_heatmaps = pred_heatmaps
    if len(pred_heatmaps[0].shape) == 2:
        disp_pred_heatmaps = [vis.color_map(hm, vmin=0, vmax=vmax, cmap=plt.cm.jet) for hm in pred_heatmaps]
    img_pred_heatmaps = cv2.resize(vis.make_grid(disp_pred_heatmaps, nCols=nCols, padval=0), None, fx=f, fy=f)

    cv2.imshow('Landmark heatmaps', cv2.cvtColor(np.vstack((img_gt_heatmaps, img_pred_heatmaps)), cv2.COLOR_RGB2BGR))


def visualize_batch(batch, X_recon, X_lm_hm, lm_preds_max, lm_preds_cnn=None, ds=None, wait=0, ssim_maps=None,
                    landmarks_to_draw=lmcfg.LANDMARKS_TO_EVALUATE, ocular_norm='pupil', horizontal=False, f=1.0):

    nimgs = min(10, len(batch))
    gt_color = (0,1,0)

    lm_confs = None
    # show landmark heatmaps
    pred_heatmaps = None
    if X_lm_hm is not None:
        pred_heatmaps = to_single_channel_heatmap(to_numpy(X_lm_hm[:nimgs]))
        pred_heatmaps = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in pred_heatmaps]
        if batch.lm_heatmaps is not None:
            gt_heatmaps = to_single_channel_heatmap(to_numpy(batch.lm_heatmaps[:nimgs]))
            gt_heatmaps = np.array([cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in gt_heatmaps])
            show_landmark_heatmaps(pred_heatmaps, gt_heatmaps, nimgs, f=1)
        lm_confs = to_numpy(X_lm_hm).reshape(X_lm_hm.shape[0], X_lm_hm.shape[1], -1).max(axis=2)

    # scale landmarks
    lm_preds_max = lm_preds_max[:nimgs] * f
    if lm_preds_cnn is not None:
        lm_preds_cnn = lm_preds_cnn[:nimgs] * f
    lm_gt = to_numpy(batch.landmarks[:nimgs]) * f
    if lm_gt.shape[1] == 98:
        lm_gt = convert_landmarks(lm_gt, LM98_TO_LM68)

    input_images = vis._to_disp_images(batch.images[:nimgs], denorm=True)
    if batch.images_mod is not None:
        disp_images = vis._to_disp_images(batch.images_mod[:nimgs], denorm=True)
    else:
        disp_images = vis._to_disp_images(batch.images[:nimgs], denorm=True)
    disp_images = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in disp_images]

    recon_images = vis._to_disp_images(X_recon[:nimgs], denorm=True)
    disp_X_recon = [cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) for im in recon_images.copy()]

    # draw landmarks to input images
    if pred_heatmaps is not None:
        disp_images = [vis.overlay_heatmap(disp_images[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]

    nme_per_lm = calc_landmark_nme(lm_gt, lm_preds_max, ocular_norm=ocular_norm)
    lm_ssim_errs = calc_landmark_ssim_error(batch.images[:nimgs], X_recon[:nimgs], batch.landmarks[:nimgs])

    #
    # Show input images
    #
    disp_images = vis.add_landmarks_to_images(disp_images, lm_gt[:nimgs], color=gt_color,
                                              draw_dots=True, draw_wireframe=False, landmarks_to_draw=landmarks_to_draw)
    disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_max[:nimgs], lm_errs=nme_per_lm,
                                              color=(0.0, 0.0, 1.0),
                                              draw_dots=True, draw_wireframe=False,
                                              gt_landmarks=lm_gt, draw_gt_offsets=True,
                                              landmarks_to_draw=landmarks_to_draw)

    # if lm_preds_cnn is not None:
    #     disp_images = vis.add_landmarks_to_images(disp_images, lm_preds_cnn, color=(1, 1, 0),
    #                                               gt_landmarks=lm_gt, draw_gt_offsets=False,
    #                                               draw_wireframe=True, landmarks_to_draw=landmarks_to_draw)

    rows = [vis.make_grid(disp_images, nCols=nimgs, normalize=False)]

    #
    # Show reconstructions
    #
    X_recon_errs = 255.0 * torch.abs(batch.images - X_recon).reshape(len(batch.images), -1).mean(dim=1)
    disp_X_recon = vis.add_error_to_images(disp_X_recon[:nimgs], errors=X_recon_errs, format_string='{:>4.1f}')

    # modes of heatmaps
    # disp_X_recon = [overlay_heatmap(disp_X_recon[i], pred_heatmaps[i]) for i in range(len(pred_heatmaps))]
    lm_errs_max = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.LANDMARKS_NO_OUTLINE)
    lm_errs_max_outline = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.LANDMARKS_ONLY_OUTLINE)
    lm_errs_max_all = calc_landmark_nme_per_img(lm_gt, lm_preds_max, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.ALL_LANDMARKS)
    disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max, loc='br-2', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_outline, loc='br-1', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_max_all, loc='br', format_string='{:>5.2f}', vmax=15)
    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_max[:nimgs], color=(0, 0, 1),
                                               landmarks_to_draw=landmarks_to_draw,
                                               draw_wireframe=False,
                                               lm_errs=nme_per_lm,
                                               # lm_confs=lm_confs,
                                               lm_confs=1-lm_ssim_errs,
                                               gt_landmarks=lm_gt,
                                               draw_gt_offsets=True,
                                               draw_dots=True)
    disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_gt, color=gt_color,
                                               draw_wireframe=False,
                                               landmarks_to_draw=landmarks_to_draw)

    # landmarks from CNN prediction
    if lm_preds_cnn is not None:
        nme_per_lm = calc_landmark_nme(lm_gt, lm_preds_cnn, ocular_norm=ocular_norm)
        disp_X_recon = vis.add_landmarks_to_images(disp_X_recon, lm_preds_cnn, color=(1,1,0),
                                                   landmarks_to_draw=lmcfg.ALL_LANDMARKS,
                                                   draw_wireframe=False,
                                                   lm_errs=nme_per_lm,
                                                   gt_landmarks=lm_gt,
                                                   draw_gt_offsets=True,
                                                   draw_dots=True,
                                                   offset_line_color=(1,1,1))
        lm_errs_cnn = calc_landmark_nme_per_img(lm_gt, lm_preds_cnn, ocular_norm=ocular_norm, landmarks_to_eval=landmarks_to_draw)
        lm_errs_cnn_outline = calc_landmark_nme_per_img(lm_gt, lm_preds_cnn, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.LANDMARKS_ONLY_OUTLINE)
        lm_errs_cnn_all = calc_landmark_nme_per_img(lm_gt, lm_preds_cnn, ocular_norm=ocular_norm, landmarks_to_eval=lmcfg.ALL_LANDMARKS)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_cnn, loc='tr', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_cnn_outline, loc='tr+1', format_string='{:>5.2f}', vmax=15)
        disp_X_recon = vis.add_error_to_images(disp_X_recon, lm_errs_cnn_all, loc='tr+2', format_string='{:>5.2f}', vmax=15)

    # print ssim errors
    ssim = np.zeros(nimgs)
    for i in range(nimgs):
        ssim[i] = compare_ssim(input_images[i], recon_images[i], data_range=1.0, multichannel=True)
    disp_X_recon = vis.add_error_to_images(disp_X_recon, 1 - ssim, loc='bl-1', format_string='{:>4.2f}',
                                           vmax=0.8, vmin=0.2)
    # print ssim torch errors
    if ssim_maps is not None:
        disp_X_recon = vis.add_error_to_images(disp_X_recon, ssim_maps.reshape(len(ssim_maps), -1).mean(axis=1),
                                               loc='bl-2', format_string='{:>4.2f}', vmin=0.0, vmax=0.4)

    rows.append(vis.make_grid(disp_X_recon, nCols=nimgs))

    if ssim_maps is not None:
        disp_ssim_maps = to_numpy(ds_utils.denormalized(ssim_maps)[:nimgs].transpose(0, 2, 3, 1))
        for i in range(len(disp_ssim_maps)):
            disp_ssim_maps[i] = vis.color_map(disp_ssim_maps[i].mean(axis=2), vmin=0.0, vmax=2.0)
        grid_ssim_maps = vis.make_grid(disp_ssim_maps, nCols=nimgs, fx=f, fy=f)
        cv2.imshow('ssim errors', cv2.cvtColor(grid_ssim_maps, cv2.COLOR_RGB2BGR))

    X_gen_lm_hm = None
    X_gen_vis = None
    show_random_faces = False
    if show_random_faces:
        with torch.no_grad():
            z_random = self.enc_rand(nimgs, self.saae.z_dim).cuda()
            outputs = self.saae.P(z_random)
            X_gen_vis = outputs[:, :3]
            if outputs.shape[1] > 3:
                X_gen_lm_hm = outputs[:, 3:]
        disp_X_gen = to_numpy(ds_utils.denormalized(X_gen_vis)[:nimgs].permute(0, 2, 3, 1))

        if X_gen_lm_hm is not None:
            if lmcfg.LANDMARK_TARGET == 'colored':
                gen_heatmaps = [to_image(X_gen_lm_hm[i]) for i in range(nimgs)]
            elif lmcfg.LANDMARK_TARGET == 'multi_channel':
                X_gen_lm_hm = X_gen_lm_hm.max(dim=1)[0]
                gen_heatmaps = [to_image(X_gen_lm_hm[i]) for i in range(nimgs)]
            else:
                gen_heatmaps = [to_image(X_gen_lm_hm[i, 0]) for i in range(nimgs)]

            disp_X_gen = [vis.overlay_heatmap(disp_X_gen[i], gen_heatmaps[i]) for i in range(len(pred_heatmaps))]

            # inputs = torch.cat([X_gen_vis, X_gen_lm_hm.detach()], dim=1)
            inputs = X_gen_lm_hm.detach()

            # disabled for multi_channel LM targets
            # lm_gen_preds = self.saae.lm_coords(inputs).reshape(len(inputs), -1, 2)
            # disp_X_gen = vis.add_landmarks_to_images(disp_X_gen, lm_gen_preds[:nimgs], color=(0,1,1))

            disp_gen_heatmaps = [vis.color_map(hm, vmin=0, vmax=1.0) for hm in gen_heatmaps]
            img_gen_heatmaps = cv2.resize(vis.make_grid(disp_gen_heatmaps, nCols=nimgs, padval=0), None, fx=1.0,
                                          fy=1.0)
            cv2.imshow('generated landmarks', cv2.cvtColor(img_gen_heatmaps, cv2.COLOR_RGB2BGR))

        rows.append(vis.make_grid(disp_X_gen, nCols=nimgs))

    # self.saae.D.train(train_state_D)
    # self.saae.Q.train(train_state_Q)
    # self.saae.P.train(train_state_P)

    if horizontal:
        assert(nimgs == 1)
        disp_rows = vis.make_grid(rows, nCols=2)
    else:
        disp_rows = vis.make_grid(rows, nCols=1)
    wnd_title = 'recon errors '
    if ds is not None:
        wnd_title += ds.__class__.__name__
    cv2.imshow(wnd_title, cv2.cvtColor(disp_rows, cv2.COLOR_RGB2BGR))
    cv2.waitKey(wait)



# LM68_TO_LM96 = 1
LM98_TO_LM68 = 2

def convert_landmarks(lms, code):
    cvt_func = {
        LM98_TO_LM68: lm98_to_lm68,
    }
    if len(lms.shape) == 3:
        new_lms = []
        for i in range(len(lms)):
            new_lms.append(cvt_func[code](lms[i]))
        return np.array(new_lms)
    elif len(lms.shape) == 2:
        return cvt_func[code](lms)
    else:
        raise ValueError



def lm98_to_lm68(lm98):
    def copy_lms(offset68, offset98, n):
        lm68[range(offset68, offset68+n)] = lm98[range(offset98, offset98+n)]

    assert len(lm98), "Cannot convert landmarks to 68 points!"
    lm68 = np.zeros((68,2), dtype=np.float32)

    # outline
    # for i in range(17):
    lm68[range(17)] = lm98[range(0,33,2)]

    # left eyebrow
    copy_lms(17, 33, 5)
    # right eyebrow
    copy_lms(22, 42, 5)
    # nose
    copy_lms(27, 51, 9)

    # eye left
    lm68[36] = lm98[60]
    lm68[37] = lm98[61]
    lm68[38] = lm98[63]
    lm68[39] = lm98[64]
    lm68[40] = lm98[65]
    lm68[41] = lm98[67]

    # eye right
    lm68[36+6] = lm98[60+8]
    lm68[37+6] = lm98[61+8]
    lm68[38+6] = lm98[63+8]
    lm68[39+6] = lm98[64+8]
    lm68[40+6] = lm98[65+8]
    lm68[41+6] = lm98[67+8]

    copy_lms(48, 76, 20)

    return lm68






