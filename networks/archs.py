import torch
import math
from torch import nn as nn
from torch.nn import functional as F
import config as cfg
from torchvision import models


def init_weights(net):
    pass
    # for m in net.modules():
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    #         # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         # m.weight.data.normal_(0, math.sqrt(2. / n))
    #         nn.init.xavier_uniform(m.weight.data)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.fill_(1)
    #         m.bias.data.zero_()
    #     elif isinstance(m, nn.Linear):
    #         nn.init.xavier_uniform(m.weight.data)

class ZNet(nn.Module):
    def __init__(self, z_dim, N=200):
        super(ZNet, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 8)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return self.lin3(x)

class LMNet(nn.Module):
    def __init__(self, z_dim, N=200):
        super(LMNet, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 68*2)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return self.lin3(x)


class D_net_gauss(nn.Module):
    def __init__(self, z_dim, N=1000):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)

    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))
        # return self.lin3(x).view(-1, 1).squeeze(1)


class EncoderLvl(nn.Module):
    def __init__(self, dim_in, dim_h, dim_f):
        super(EncoderLvl, self).__init__()
        N = 512
        self.seqential = nn.Sequential(
            nn.Linear(dim_in, N),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(N, dim_out)
        )
        self.lin_h = nn.Linear(N, dim_h)
        self.lin_f = nn.Linear(N, dim_f)

    def forward(self, x):
        x = self.seqential(x)
        h, f = self.lin_h(x), self.lin_f(x)
        h = torch.nn.functional.normalize(h)
        return h, f


class DecoderLvl(nn.Module):
    def __init__(self, dim, dim_out):
        super(DecoderLvl, self).__init__()
        N = 512
        self.seqential = nn.Sequential(
            nn.Linear(dim, N),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(512, 256),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(N, dim_out)
        )

    def forward(self, x):
        return self.seqential(x)


class EncoderLvlParallel(nn.Module):
    def __init__(self, dim_in, dim_ft):
        self.dim_in = dim_in
        dim_P, dim_I, dim_S, dim_E = 3, dim_ft, dim_ft, dim_ft
        super(EncoderLvlParallel, self).__init__()
        N = 1000
        self.seqential = nn.Sequential(
            # nn.Linear(dim_in, int(N/2)),
            # nn.Dropout(p=0.2),
            # nn.LeakyReLU(inplace=True),
            nn.Linear(dim_in, N),
            # nn.Linear(int(N/2), N),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),

            # nn.Linear(N, N),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(N, N),
            # nn.LeakyReLU(inplace=True),
        )
        self.lin_P = nn.Linear(N, dim_P)
        self.lin_I = nn.Linear(N, dim_I)
        self.lin_S = nn.Linear(N, dim_S)
        self.lin_E = nn.Linear(N, dim_E)
        self.conv1d = nn.Conv1d(1, 1, 1)
        self.bn =  nn.BatchNorm2d(1)

    def forward(self, x):
        # x = torch.unsqueeze(x, dim=1)
        # x = self.conv1d(x)
        # x = torch.unsqueeze(x, dim=1)
        # x = self.bn(x)
        # x = torch.squeeze(x, dim=1)
        # x = torch.squeeze(x, dim=1)
        x = self.seqential(x)
        norm = torch.nn.functional.normalize
        return [self.lin_P(x)] + [norm(f) for f in [self.lin_I(x), self.lin_S(x), self.lin_E(x)]]


class DecoderLvlParallel(nn.Module):
    def __init__(self, dim, dim_out):
        super(DecoderLvlParallel, self).__init__()
        N = 1000
        self.seqential = nn.Sequential(
            nn.Linear(dim, N),
            # nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),

            # nn.Linear(N, N),
            # nn.LeakyReLU(inplace=True),
            #
            # nn.Linear(N, N),
            # nn.LeakyReLU(inplace=True),

            nn.Linear(N, dim_out)
        )
        self.conv1d = nn.Conv1d(1, 1, 1)
        self.bn =  nn.BatchNorm2d(1)

    def forward(self, p, i, s, e):
        x = torch.cat((p, i, s, e), dim=1)
        # x = torch.unsqueeze(x, dim=1)
        # x = self.conv1d(x)
        # x = torch.unsqueeze(x, dim=1)
        # x = self.bn(x)
        # x = torch.squeeze(x, dim=1)
        # x = torch.squeeze(x, dim=1)
        return self.seqential(x)


class Q_net(nn.Module):
    def __init__(self, z_dim, f_dim, N=400):
        super(Q_net, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, f_dim)

    def forward(self, x):
        # x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        # x = F.dropout(self.lin2(x), p=0.1, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        return self.lin3(x)


class DCGAN_Encoder(nn.Module):
    def __init__(self, z_dim):
        super(DCGAN_Encoder, self).__init__()
        self.z_dim = z_dim
        ngf = 64
        nc = 3
        self.main = nn.Sequential(
            # # state size. (nc) x 128 x 128
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (nc) x 64 x 64
            nn.Conv2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(ngf * 8, self.z_dim, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        # return torch.nn.functional.normalize(x)
        return x

class DCGAN_Decoder(nn.Module):
    def __init__(self, z_dim):
        super(DCGAN_Decoder, self).__init__()
        self.z_dim = z_dim
        ngf = 64
        nc = 3
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # # state size. (nc) x 128 x 128
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), self.z_dim, 1, 1)
        x = self.main(x)
        x = self.tanh(x)
        return x


class EncoderExtractor(nn.Module):
    def __init__(self, original_model):
        super(EncoderExtractor, self).__init__()
        self.main = nn.Sequential(*list(original_model.main.children())[:1])

    def forward(self, x):
        return self.main(x)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         # self.z_dim = z_dim
#         ndf = 16
#         nc = 3
#         # self.main = nn.Sequential(
#         self.main = [
#             # input is (nc) x 256 x 256
#             # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             # nn.LeakyReLU(0.2, inplace=True),
#
#             # state size. (ndf) x 128 x 128
#             nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 64 x 64
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 32 x 32
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 16 x 16
#             nn.Conv2d(ndf*8, ndf * 16, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 16),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*16) x 8 x 8
#             nn.Conv2d(ndf*16, ndf * 32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 32),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*32) x 4 x 4
#             nn.Conv2d(ndf * 32, 1, 4, 1, 0, bias=False),
#         ]
#         if not cfg.RGAN:
#             self.main.append(nn.Sigmoid())
#         self.main = nn.Sequential(*self.main)
#
#     def forward(self, input):
#         output = self.main(input)
#         return output.view(-1, 1).squeeze(1)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # self.z_dim = z_dim
        ndf = 32
        nc = 3
        # self.main = nn.Sequential(
        self.main = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(ndf*8, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),

            nn.AdaptiveAvgPool2d((1, 1))
        ]
        if not cfg.RGAN:
            self.main.append(nn.Sigmoid())
        self.main = nn.Sequential(*self.main)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)