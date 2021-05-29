from .partialconv2d import PartialConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
PartialConv = PartialConv2d

# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class EdgeGenerator(nn.Module):
#     def __init__(self, in_channels_feature, kernel_s = 3, add_last_edge = True):
#         super(EdgeGenerator, self).__init__()
#
#         # self.p_conv = PartialConv2d(in_channels_feature + 1, 64, kernel_size = kernel_s, stride = 1, padding = kernel_s // 2, multi_channel = True, bias = False)
#         self.p_conv = PartialConv2d(in_channels_feature, 3, kernel_size = kernel_s, stride = 1, padding = kernel_s // 2, multi_channel = True, bias = False)
#
#         self.edge_resolver = Bottleneck(3, 3)
#         self.out_layer = nn.Conv2d(3, in_channels_feature, 1, bias = False)
#
#     def forward(self, in_x, mask):
#         x, mask_updated = self.p_conv(in_x, mask)
#
#         # print("x++++++++++" )
#         # print(x.shape)
#         # print("----------")
#         # print(mask_updated.shape)
#
#         x = self.edge_resolver(x)
#
#         # print("x2++++++++++" )
#         # print(x.shape)
#
#         feat_out = self.out_layer(x)
#
#         # print("feat_out++++++++++" )
#         # print(feat_out.shape)
#         #
#         # return
#         return feat_out, mask_updated
#
# class VSRLayer(nn.Module):
#     def __init__(self, in_channel, out_channel, stride = 2, kernel_size = 3, batch_norm = True, activation = "ReLU", deconv = False):
#         super(VSRLayer, self).__init__()
#         self.edge_generator = EdgeGenerator(in_channel, kernel_s = kernel_size)
#         # self.feat_rec = PartialConv(in_channel+1, out_channel, stride = stride, kernel_size = kernel_size, padding = kernel_size//2, multi_channel = True)
#         self.feat_rec = PartialConv(in_channel, out_channel, stride = stride, kernel_size = kernel_size, padding = kernel_size//2, multi_channel = True)
#
#         if deconv:
#             self.deconv = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
#         else:
#             self.deconv = lambda x:x
#
#         if batch_norm:
#             self.batchnorm = nn.BatchNorm2d(out_channel)
#         else:
#             self.batchnorm = lambda x:x
#
#         self.stride = stride
#
#         if activation == "ReLU":
#             self.activation = nn.ReLU(True)
#         elif activation == "Leaky":
#             self.activation = nn.LeakyReLU(0.2, True)
#         else:
#             self.activation = lambda x:x
#
#     # def forward(self, feat_in, mask_in, edge_in): # 带遮罩的图片 遮罩 带遮罩的边缘
#     def forward(self, feat_in, mask_in):  # 带遮罩的图片 遮罩
#
#         # edge_in = F.interpolate(edge_in, size = feat_in.size()[2:])
#         # edge_updated, mask_updated = self.edge_generator(torch.cat([feat_in, edge_in], dim = 1), torch.cat([mask_in, mask_in[:,:1,:,:]], dim = 1))
#         feat_updated, mask_updated = self.edge_generator(feat_in, mask_in) # 此时edge_generator只用于提取特征
#
#         # edge_reconstructed = edge_in * mask_in[:,:1,:,:] + feat_updated * (mask_updated[:,:1,:,:] - mask_in[:,:1,:,:])
#         # feat_out, feat_mask = self.feat_rec(torch.cat([feat_in, edge_reconstructed], dim = 1), torch.cat([mask_in, mask_updated[:,:1,:,:]], dim = 1))
#
#         print("feat_updated++++++++++" )
#         print(feat_updated.shape)
#         print("mask_updated----------")
#         print(mask_updated.shape)
#
#         #修改1
#         feat_out, feat_mask = self.feat_rec(feat_updated, mask_updated)
#
#         #修改2
#         # feat_out, feat_mask = self.feat_rec(feat_in, mask_in)
#
#         feat_out = self.deconv(feat_out)
#         feat_out = self.batchnorm(feat_out)
#         feat_out = self.activation(feat_out)
#         mask_updated = F.interpolate(mask_updated, size = feat_out.size()[2:])
#         feat_mask = F.interpolate(feat_mask, size = feat_out.size()[2:])
#         # return feat_out, feat_mask*mask_updated[:,0:1,:,:], edge_reconstructed
#
#         print("feat_out++++++++++" )
#         print(feat_out.shape)
#         print("feat_mask * mask_updated[:, 0:1, :, :]----------")
#         print((feat_mask * mask_updated[:, 0:1, :, :]).shape)
#         return feat_out, feat_mask * mask_updated[:, 0:1, :, :]




class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class EdgeGenerator(nn.Module):
    def __init__(self, in_channels_feature, kernel_s=3, add_last_edge=True):
        super(EdgeGenerator, self).__init__()

        self.p_conv = PartialConv2d(in_channels_feature, 64, kernel_size=kernel_s, stride=1, padding=kernel_s // 2,
                                    multi_channel=True, bias=False)

        self.edge_resolver = Bottleneck(64, 16)
        self.out_layer = nn.Conv2d(64, 1, 1, bias=False)

    def forward(self, in_x, mask):
        x, mask_updated = self.p_conv(in_x, mask)

        # print("x.shape++++++++++")
        # print(x.shape)

        x = self.edge_resolver(x)
        edge_out = self.out_layer(x)
        return edge_out, mask_updated


class VSRLayer(nn.Module):
    def __init__(self, in_channel, out_channel, stride=2, kernel_size=3, batch_norm=True, activation="ReLU",
                 deconv=False):
        super(VSRLayer, self).__init__()
        self.edge_generator = EdgeGenerator(in_channel, kernel_s=kernel_size)
        self.feat_rec = PartialConv(in_channel + 1, out_channel, stride=stride, kernel_size=kernel_size,
                                    padding=kernel_size // 2, multi_channel=True)
        if deconv:
            self.deconv = nn.ConvTranspose2d(out_channel, out_channel, 4, 2, 1)
        else:
            self.deconv = lambda x: x

        if batch_norm:
            self.batchnorm = nn.BatchNorm2d(out_channel)
        else:
            self.batchnorm = lambda x: x

        self.stride = stride

        if activation == "ReLU":
            self.activation = nn.ReLU(True)
        elif activation == "Leaky":
            self.activation = nn.LeakyReLU(0.2, True)
        else:
            self.activation = lambda x: x

    def forward(self, feat_in, mask_in):
        # edge_in = F.interpolate(edge_in, size=feat_in.size()[2:])
        #
        # print("edge_in.shape++++++++++")
        # print(edge_in.shape)

        # edge_updated, mask_updated = self.edge_generator(torch.cat([feat_in, edge_in], dim=1),
        #                                                  torch.cat([mask_in, mask_in[:, :1, :, :]], dim=1))

        # print("feat_in.shape++++++++++")
        # print(feat_in.shape)
        # print("mask_in.shape----------")
        # print(mask_in.shape)

        feat_updated, mask_updated = self.edge_generator(feat_in, mask_in) # 此时edge_generator只用于提取特征

        # print("edge_updated.shape++++++++++")
        # print(feat_updated.shape)
        # print("mask_updated.shape----------")
        # print(mask_updated.shape)

        # edge_reconstructed = edge_in * mask_in[:, :1, :, :] + edge_updated * (
        #             mask_updated[:, :1, :, :] - mask_in[:, :1, :, :])

        # print("edge_reconstructed.shape++++++++++")
        # print(edge_reconstructed.shape)

        # print("torch.cat([feat_in, edge_reconstructed], dim = 1).shape++++++++++")
        # print(torch.cat([feat_in, feat_updated], dim=1).shape)
        # print("torch.cat([mask_in, mask_updated[:,:1,:,:]], dim = 1).shape----------")
        # print(torch.cat([mask_in, mask_updated[:, :1, :, :]], dim=1).shape)
        # print("mask_updated[:,:1,:,:].shape----------")
        # print(mask_updated[:, :1, :, :].shape)

        feat_out, feat_mask = self.feat_rec(torch.cat([feat_in, feat_updated], dim=1),
                                            torch.cat([mask_in, mask_updated[:, :1, :, :]], dim=1))

        feat_out = self.deconv(feat_out)
        feat_out = self.batchnorm(feat_out)
        feat_out = self.activation(feat_out)
        mask_updated = F.interpolate(mask_updated, size=feat_out.size()[2:])
        feat_mask = F.interpolate(feat_mask, size=feat_out.size()[2:])
        return feat_out, feat_mask * mask_updated[:, 0:1, :, :]
