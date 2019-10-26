import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import config as cfg
import networks.invresnet
from datasets import ds_utils
from utils import vis
from networks.archs import D_net_gauss, Discriminator
from networks import resnet_ae, archs
from utils.nn import to_numpy, calc_triplet_loss


def calc_acc(outputs, labels):
    assert(outputs.shape[1] == 8)
    assert(len(outputs) == len(labels))
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels)
    acc = corrects.double()/float(outputs.size(0))
    return acc.item()


class SAAE(nn.Module):
    def __init__(self, pretrained_encoder=False):
        super(SAAE, self).__init__()

        dim_ft = cfg.EXPRESSION_DIMS
        dim_pose = 3
        self.z_dim = dim_ft*3 + 3
        input_channels = 3

        if cfg.ARCH == 'dcgan':
            self.Q = archs.DCGAN_Encoder(self.z_dim).cuda()
            self.P = archs.DCGAN_Decoder(self.z_dim).cuda()
        elif cfg.ARCH == 'resnet':
            self.Q = resnet_ae.resnet18(pretrained=pretrained_encoder,
                                        num_classes=self.z_dim,
                                        input_size=cfg.INPUT_SIZE,
                                        input_channels=input_channels,
                                        layer_normalization=cfg.ENCODER_LAYER_NORMALIZATION).cuda()

            if cfg.DECODER_FIXED_ARCH:
                decoder_class = networks.invresnet.InvResNet
            else:
                decoder_class = networks.invresnet.InvResNet_old

            num_blocks = [cfg.DECODER_PLANES_PER_BLOCK] * 4
            self.P = decoder_class(networks.invresnet.InvBasicBlock,
                                   num_blocks,
                                   input_dims=self.z_dim,
                                   output_size=cfg.INPUT_SIZE,
                                   output_channels=input_channels,
                                   layer_normalization=cfg.DECODER_LAYER_NORMALIZATION).cuda()
        else:
            raise ValueError('Unknown network architecture!')

        self.D_z = D_net_gauss(self.z_dim).cuda()
        self.D = Discriminator().cuda()

        # Disentanglement
        self.factors = ['pose', 'id', 'shape', 'expression']
        self.E = archs.EncoderLvlParallel(self.z_dim, dim_ft).cuda()
        self.G = archs.DecoderLvlParallel(dim_pose + 3 * dim_ft, self.z_dim).cuda()
        self.fp, self.fi, self.fs, self.se = None, None, None, None

        # Emotion classification from Z vector
        self.znet = archs.ZNet(dim_ft).cuda()
        self.cross_entropy_loss = nn.CrossEntropyLoss().cuda()

        # lr_l = 0.00002
        lr_l = 0.00008
        betas_disent = (0.9, 0.999)
        self.optimizer_E = optim.Adam(self.E.parameters(), lr=lr_l, betas=betas_disent)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr_l, betas=betas_disent)
        self.optimizer_znet = optim.Adam(self.znet.parameters(), lr=lr_l)

        def count_parameters(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        print("Trainable params Q: {:,}".format(count_parameters(self.Q)))
        print("Trainable params P: {:,}".format(count_parameters(self.P)))
        print("Trainable params D_z: {:,}".format(count_parameters(self.D_z)))
        print("Trainable params D: {:,}".format(count_parameters(self.D)))
        print("Trainable params E: {:,}".format(count_parameters(self.E)))
        print("Trainable params G: {:,}".format(count_parameters(self.G)))

        self.total_iter = 0
        self.iter = 0
        self.z = None
        self.images = None
        self.current_dataset = None

    def z_vecs(self):
        return [to_numpy(self.z)]

    def z_vecs_pre(self):
        return [to_numpy(self.z_pre)]

    def id_vec(self):
        if cfg.WITH_PARALLEL_DISENTANGLEMENT:
            try:
                return to_numpy(self.f_parallel[1])
            except:
                return None

    def res_vec(self, exclude_fid):
        if cfg.WITH_PARALLEL_DISENTANGLEMENT:
            try:
                h = torch.cat([self.f_parallel[i] for i in range(len(self.f_parallel)) if i != exclude_fid], dim=1)
                return to_numpy(h)
            except:
                return None

    def exp_vec(self):
        if cfg.WITH_PARALLEL_DISENTANGLEMENT:
            try:
                return to_numpy(self.f_parallel[3])
            except:
                return None

    def f_vec(self, i):
        if cfg.WITH_PARALLEL_DISENTANGLEMENT:
            try:
                return to_numpy(self.f_parallel[i])
            except:
                return None

    def poses_pred(self):
        try:
            return to_numpy(self.f_parallel[0])
        except:
            return None

    def emotions_pred(self):
        try:
            if self.emotion_probs is not None:
                return np.argmax(to_numpy(self.emotion_probs), axis=1)
        except AttributeError:
            pass
        return None

    def forward(self, X, Y=None, skip_disentanglement=False):
        self.z_pre = self.Q(X)
        if skip_disentanglement:
            self.z = self.z_pre
            self.emotion_probs = None
        else:
            self.z = self.run_disentanglement(self.z_pre, Y=Y)[0]
        return self.P(self.z)

    def run_disentanglement(self,  z, Y=None, train=False):
        if train:
            return self.__train_disenglement_parallel(z, Y)
        else:
            return self.__forward_disentanglement_parallel(z, Y)

    def __forward_disentanglement_parallel(self, z, Y=None):
        iter_stats = {}

        with torch.no_grad():
            self.f_parallel = self.E(z)
            z_recon = self.G(*self.f_parallel)

            ft_id = 3
            try:
                y = Y[ft_id]
            except TypeError:
                y = None

            y_f = self.f_parallel[3]

            try:
                y_p = self.f_parallel[0]
                def calc_err(outputs, target):
                    return np.abs(np.rad2deg(F.l1_loss(outputs, target, reduction='none').detach().cpu().numpy()))
                iter_stats['err_pose'] = calc_err(y_p, Y[0])
            except TypeError:
                pass


            clprobs = self.znet(y_f)
            self.emotion_probs = clprobs

            if y is not None:
                emotion_labels = y[:, 0].long()
                loss_cls = self.cross_entropy_loss(clprobs, emotion_labels)
                acc_cls = calc_acc(clprobs, emotion_labels)
                iter_stats['loss_cls'] = loss_cls.item()
                iter_stats['acc_cls'] = acc_cls
                iter_stats['emotion_probs'] = to_numpy(clprobs)
                iter_stats['emotion_labels'] = to_numpy(emotion_labels)
                f_parallel_recon = self.E(self.Q(self.P(z_recon)[:,:3]))
                l1_err = torch.abs(torch.cat(f_parallel_recon, dim=1) - torch.cat(self.f_parallel, dim=1)).mean(dim=1)
                iter_stats['l1_dis_cycle'] = to_numpy(l1_err)

        return z_recon, iter_stats, None


    def __train_disenglement_parallel(self, z, Y=None, train=True):
        iter_stats = {}

        self.E.train(train)
        self.G.train(train)

        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()

        #
        # Autoencoding phase
        #

        fts = self.E(z)
        fp, fi, fs, fe = fts

        z_recon = self.G(fp, fi, fs, fe)

        loss_z_recon = F.l1_loss(z, z_recon) * cfg.W_Z_RECON
        if not cfg.WITH_Z_RECON_LOSS:
            loss_z_recon *= 0

        #
        # Info min/max phase
        #

        loss_I = loss_z_recon
        loss_G = torch.zeros(1, requires_grad=True).cuda()

        def calc_err(outputs, target):
            return np.abs(np.rad2deg(F.l1_loss(outputs, target, reduction='none').detach().cpu().numpy().mean(axis=0)))

        def cosine_loss(outputs, targets):
            return (1 - F.cosine_similarity(outputs, targets, dim=1)).mean()

        if Y[3] is not None and Y[3].sum() > 0:  # Has expression -> AffectNet
            available_factors = [3,3,3]
            if cfg.WITH_POSE:
                available_factors = [0] + available_factors
        elif Y[2][1] is not None:  # has vids -> VoxCeleb
            available_factors = [2]
        elif Y[1] is not None:  # Has identities
            available_factors = [1,1,1]
            if cfg.WITH_POSE:
                available_factors = [0] + available_factors
        elif Y[0] is not None: # Any dataset with pose
            available_factors = [0,1,3]

        lvl = available_factors[self.iter % len(available_factors)]

        name = self.factors[lvl]
        try:
            y = Y[lvl]
        except TypeError:
            y = None

        # if y is not None and name != 'shape':
        def calc_feature_loss(name, y_f, y, show_triplets=False, wnd_title=None):
            if name == 'id' or name == 'shape' or name == 'expression':
                display_images = None
                if show_triplets:
                    display_images = self.images
                loss_I_f, err_f = calc_triplet_loss(y_f, y, return_acc=True, images=display_images, feature_name=name,
                                                    wnd_title=wnd_title)
                if name == 'expression':
                    loss_I_f *= 2.0
            elif name == 'pose':
                # loss_I_f, err_f = F.l1_loss(y_f, y), calc_err(y_f, y)
                loss_I_f, err_f = F.mse_loss(y_f, y)*1, calc_err(y_f, y)
                # loss_I_f, err_f = cosine_loss(y_f, y), calc_err(y_f, y)
            else:
                raise ValueError("Unknown feature name!")
            return loss_I_f, err_f


        if y is not None and cfg.WITH_FEATURE_LOSS:

            show_triplets = (self.iter + 1) % self.print_interval  == 0

            y_f = fts[lvl]
            loss_I_f, err_f = calc_feature_loss(name, y_f, y, show_triplets=show_triplets)

            loss_I += cfg.W_FEAT * loss_I_f

            iter_stats[name+'_loss_f'] = loss_I_f.item()
            iter_stats[name+'_err_f'] = np.mean(err_f)

            # train expression classifier
            if name == 'expression':
                self.znet.zero_grad()
                emotion_labels = y[:,0].long()
                clprobs = self.znet(y_f.detach())  # train only znet
                # clprobs = self.znet(y_f)  # train enoder and znet
                # loss_cls = self.cross_entropy_loss(clprobs, emotion_labels)
                loss_cls = self.weighted_CE_loss(clprobs, emotion_labels)

                acc_cls = calc_acc(clprobs, emotion_labels)
                if train:
                    loss_cls.backward(retain_graph=False)
                self.optimizer_znet.step()
                iter_stats['loss_cls'] = loss_cls.item()
                iter_stats['acc_cls'] = acc_cls
                iter_stats['expression_y_probs'] = to_numpy(clprobs)
                iter_stats['expression_y'] = to_numpy(y)


        # cycle loss
        # other_levels = [0,1,2,3]
        # other_levels.remove(lvl)
        # shuffle_lvl = np.random.permutation(other_levels)[0]
        shuffle_lvl = lvl
        # print("shuffling level {}...".format(shuffle_lvl))
        if cfg.WITH_DISENT_CYCLE_LOSS:
            # z_random = torch.rand_like(z).cuda()
            # fts_random = self.E(z_random)

            # create modified feature vectors
            fts[0] = fts[0].detach()
            fts[1] = fts[1].detach()
            fts[2] = fts[2].detach()
            fts[3] = fts[3].detach()
            fts_mod = fts.copy()
            shuffled_ids = torch.randperm(len(fts[shuffle_lvl]))
            y_mod = None
            if y is not None:
                if name == 'shape':
                    y_mod = [y[0][shuffled_ids], y[1][shuffled_ids]]
                else:
                    y_mod = y[shuffled_ids]

            fts_mod[shuffle_lvl] = fts[shuffle_lvl][shuffled_ids]

            # predict full cycle
            z_random_mod = self.G(*fts_mod)
            X_random_mod = self.P(z_random_mod)[:,:3]
            z_random_mod_recon = self.Q(X_random_mod)
            fts2 = self.E(z_random_mod_recon)

            # recon error in unmodified part
            # h = torch.cat([fts_mod[i] for i in range(len(fts_mod)) if i != lvl], dim=1)
            # h2 = torch.cat([fts2[i] for i in range(len(fts2)) if i != lvl], dim=1)
            # l1_err_h = torch.abs(h - h2).mean(dim=1)
            # l1_err_h = torch.abs(torch.cat(fts_mod, dim=1) - torch.cat(fts2, dim=1)).mean(dim=1)

            # recon error in modified part
            # l1_err_f = np.rad2deg(to_numpy(torch.abs(fts_mod[lvl] - fts2[lvl]).mean(dim=1)))

            # recon error in entire vector
            l1_err = torch.abs(torch.cat(fts_mod, dim=1)[:,3:] - torch.cat(fts2, dim=1)[:,3:]).mean(dim=1)
            loss_dis_cycle = F.l1_loss(torch.cat(fts_mod, dim=1)[:,3:], torch.cat(fts2, dim=1)[:,3:]) * cfg.W_CYCLE
            iter_stats['loss_dis_cycle'] = loss_dis_cycle.item()

            loss_I += loss_dis_cycle

            # cycle augmentation loss
            if cfg.WITH_AUGMENTATION_LOSS and y_mod is not None:
                y_f_2 = fts2[lvl]
                loss_I_f_2, err_f_2 = calc_feature_loss(name, y_f_2, y_mod, show_triplets=show_triplets, wnd_title='aug')
                loss_I += loss_I_f_2 * cfg.W_AUG
                iter_stats[name+'_loss_f_2'] = loss_I_f_2.item()
                iter_stats[name+'_err_f_2'] = np.mean(err_f_2)

            #
            # Adversarial loss of modified generations
            #

            GAN = False
            if GAN and train:
                eps = 0.00001

                # #######################
                # # GAN discriminator phase
                # #######################
                update_D = False
                if update_D:
                    self.D.zero_grad()
                    err_real = self.D(self.images)
                    err_fake = self.D(X_random_mod.detach())
                    # err_fake = self.D(X_z_recon.detach())
                    loss_D = -torch.mean(torch.log(err_real + eps) + torch.log(1.0 - err_fake + eps)) * 0.1
                    loss_D.backward()
                    self.optimizer_D.step()
                    iter_stats.update({'loss_D': loss_D.item()})

                #######################
                # Generator loss
                #######################
                self.D.zero_grad()
                err_fake = self.D(X_random_mod)
                # err_fake = self.D(X_z_recon)
                loss_G += -torch.mean(torch.log(err_fake + eps))

                iter_stats.update({'loss_G': loss_G.item()})
                # iter_stats.update({'err_real': err_real.mean().item(), 'err_fake': loss_G.mean().item()})

            # debug visualization
            show = True
            if show:
                if (self.iter+1) % self.print_interval in [0,1]:
                    if Y[3] is None:
                        emotion_gt = np.zeros(len(z), dtype=int)
                        emotion_gt_mod = np.zeros(len(z), dtype=int)
                    else:
                        emotion_gt = Y[3][:,0].long()
                        emotion_gt_mod = Y[3][shuffled_ids,0].long()
                    with torch.no_grad():
                        self.znet.eval()
                        self.G.eval()
                        emotion_preds = torch.max(self.znet(fe.detach()), 1)[1]
                        emotion_mod = torch.max(self.znet(fts_mod[3].detach()), 1)[1]
                        emotion_mod_pred = torch.max(self.znet(fts2[3].detach()), 1)[1]
                        X_recon = self.P(z)[:,:3]
                        X_z_recon = self.P(z_recon)[:,:3]
                        X_random_mod_recon = self.P(self.G(*fts2))[:,:3]
                        self.znet.train(train)
                        self.G.train(train)
                        X_recon_errs = 255.0 * torch.abs(self.images - X_recon).reshape(len(self.images), -1).mean(dim=1)
                        X_z_recon_errs = 255.0 * torch.abs(self.images - X_z_recon).reshape(len(self.images), -1).mean(dim=1)

                    nimgs = 8

                    disp_input = vis.add_pose_to_images(ds_utils.denormalized(self.images)[:nimgs], Y[0], color=(0, 0, 1.0))
                    if name == 'expression':
                        disp_input = vis.add_emotion_to_images(disp_input, to_numpy(emotion_gt))
                    elif name == 'id':
                        disp_input = vis.add_id_to_images(disp_input, to_numpy(Y[1]))

                    disp_recon = vis.add_pose_to_images(ds_utils.denormalized(X_recon)[:nimgs], fts[0])
                    disp_recon = vis.add_error_to_images(disp_recon, errors=X_recon_errs, format_string='{:.1f}')

                    disp_z_recon = vis.add_pose_to_images(ds_utils.denormalized(X_z_recon)[:nimgs], fts[0])
                    disp_z_recon = vis.add_emotion_to_images(disp_z_recon, to_numpy(emotion_preds),
                                                             gt_emotions=to_numpy(emotion_gt) if name=='expression' else None)
                    disp_z_recon = vis.add_error_to_images(disp_z_recon, errors=X_z_recon_errs, format_string='{:.1f}')

                    disp_input_shuffle = vis.add_pose_to_images(ds_utils.denormalized(self.images[shuffled_ids])[:nimgs], fts[0][shuffled_ids])
                    disp_input_shuffle = vis.add_emotion_to_images(disp_input_shuffle, to_numpy(emotion_gt_mod))
                    if name == 'id':
                        disp_input_shuffle = vis.add_id_to_images(disp_input_shuffle, to_numpy(Y[1][shuffled_ids]))

                    disp_recon_shuffle = vis.add_pose_to_images(ds_utils.denormalized(X_random_mod)[:nimgs], fts_mod[0], color=(0, 0, 1.0))
                    disp_recon_shuffle = vis.add_emotion_to_images(disp_recon_shuffle, to_numpy(emotion_mod))

                    disp_cycle = vis.add_pose_to_images(ds_utils.denormalized(X_random_mod_recon)[:nimgs], fts2[0])
                    disp_cycle = vis.add_emotion_to_images(disp_cycle, to_numpy(emotion_mod_pred))
                    disp_cycle = vis.add_error_to_images(disp_cycle, errors=l1_err, format_string='{:.3f}',
                                                         size=0.6, thickness=2, vmin=0, vmax=0.1)

                    rows = [
                        # original input images
                        vis.make_grid(disp_input, nCols=nimgs),

                        # reconstructions without disentanglement
                        vis.make_grid(disp_recon, nCols=nimgs),

                        # reconstructions with disentanglement
                        vis.make_grid(disp_z_recon, nCols=nimgs),

                        # source for feature transfer
                        vis.make_grid(disp_input_shuffle, nCols=nimgs),

                        # reconstructions with modified feature vector (direkt)
                        vis.make_grid(disp_recon_shuffle, nCols=nimgs),

                        # reconstructions with modified feature vector (1 iters)
                        vis.make_grid(disp_cycle, nCols=nimgs)
                    ]
                    f = 1.0 / cfg.INPUT_SCALE_FACTOR
                    disp_img = vis.make_grid(rows, nCols=1, normalize=False, fx=f, fy=f)

                    wnd_title = name
                    if self.current_dataset is not None:
                        wnd_title += ' ' + self.current_dataset.__class__.__name__
                    cv2.imshow(wnd_title, cv2.cvtColor(disp_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(10)

        loss_I *= cfg.W_DISENT

        iter_stats['loss_disent'] = loss_I.item()

        if train:
            loss_I.backward(retain_graph=True)

        return z_recon, iter_stats, loss_G[0]


def vis_reconstruction(net, inputs, ids=None, clips=None, poses=None, emotions=None, landmarks=None, landmarks_pred=None,
                       pytorch_ssim=None, fx=0.5, fy=0.5, ncols=10, skip_disentanglement=False):
    net.eval()
    cs_errs = None
    with torch.no_grad():
        X_recon = net(inputs, Y=None, skip_disentanglement=skip_disentanglement)
        # second pass
        # X_recon = net(X_recon, Y=None, skip_disentanglement=skip_disentanglement)

        if pytorch_ssim is not None:
            cs_errs = np.zeros(len(inputs))
            for i in range(len(cs_errs)):
                cs_errs[i] = 1 - pytorch_ssim(inputs[i].unsqueeze(0), X_recon[i].unsqueeze(0)).item()

    return vis.draw_results(inputs, X_recon, net.z_vecs(),
                            ids=ids, poses=poses, poses_pred=net.poses_pred(),
                            emotions=emotions, emotions_pred=net.emotions_pred(),
                            cs_errs=cs_errs,
                            fx=fx, fy=fy, ncols=ncols)



