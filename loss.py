from .compute_similarity import compute_similarity, compute_similarity_edge_v1
from .net_tensor import VGGNet
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torchvision.transforms.functional as F
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
import cv2



class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


'''
修改7 similarity loss
'''
class SimilarityLoss(nn.Module):

    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

        # make VGG model
        self.model_vgg = VGGNet()


    def __call__(self, output, vec_sim_path):

        sim_, sim = compute_similarity(output, vec_sim_path, self.model_vgg)  # cuda tensor
        similarity_loss = self.criterion(sim, sim_)

        return similarity_loss


'''
修改7 similarity loss
'''
class SimilarityLoss_Edge_v1(nn.Module):

    def getEdge(self, outputs_tensor):
        outputs_tensor = outputs_tensor.cpu()

        res = []
        for output in outputs_tensor:
            img = F.to_pil_image(output).resize((224, 224))  # 得到Image并进行resize
            img_gray = rgb2gray(np.array(img))  # 得到灰度图
            img_edge = canny(img_gray).astype(np.float)  # 得到边缘 ndarray

            # 叠加成三通道
            img_edge = np.expand_dims(img_edge, 2)
            img_edge = np.concatenate([img_edge, img_edge, img_edge], axis=2)
            img_edge = img_edge.swapaxes(0, 2) # shape(3, 224, 224)

            # img_edge = np.expand_dims(img_edge, 0)
            res.append(img_edge)
        res = torch.tensor(np.array(res)).cuda().float()

        # 返回cuda tensor
        return res

    def __init__(self):
        super(SimilarityLoss_Edge_v1, self).__init__()
        self.criterion = torch.nn.L1Loss()

        # make VGG model
        self.model_vgg = VGGNet()


    def __call__(self, output, vec_sim_path):

        edge_tensor = self.getEdge(output)

        sim_, sim = compute_similarity_edge_v1(edge_tensor, vec_sim_path, self.model_vgg)  # cuda tensor
        similarity_loss = self.criterion(sim, sim_)

        return similarity_loss


'''
修改7 similarity loss
'''
class SimilarityLoss_ImageGray(nn.Module):

    def getEdge(self, outputs_tensor):
        outputs_tensor = outputs_tensor.cpu()

        res = []
        for output in outputs_tensor:
            img = F.to_pil_image(output).resize((224, 224))  # 得到Image并进行resize
            img_gray = rgb2gray(np.array(img))  # 得到灰度图
            img_edge = canny(img_gray).astype(np.float)  # 得到边缘 ndarray

            # 叠加成三通道
            img_edge = np.expand_dims(img_edge, 2)
            img_gray = np.expand_dims(img_gray, 2)
            img_edge = np.concatenate([img_edge, img_gray, img_edge], axis=2)
            img_edge = img_edge.swapaxes(0, 2) # shape(3, 224, 224)

            # img_edge = np.expand_dims(img_edge, 0)
            res.append(img_edge)
        res = torch.tensor(np.array(res)).cuda().float()

        # 返回cuda tensor
        return res

    def __init__(self):
        super(SimilarityLoss_ImageGray, self).__init__()
        self.criterion = torch.nn.L1Loss()

        # make VGG model
        self.model_vgg = VGGNet()


    def __call__(self, output, vec_sim_path):

        edge_tensor = self.getEdge(output)

        sim_, sim = compute_similarity_edge_v1(edge_tensor, vec_sim_path, self.model_vgg)  # cuda tensor
        similarity_loss = self.criterion(sim, sim_)

        return similarity_loss


# sift loss
class DescriptorLoss(nn.Module):

    def __init__(self):
        super(DescriptorLoss, self).__init__()
        self.criterion = torch.nn.L1Loss()

    # 提取sift特征
    def getSift(self, imgs_tensor, outputs_tensor):
        imgs_tensor = imgs_tensor.cpu()  # cuda tensor 转成cpu tensor
        outputs_tensor = outputs_tensor.cpu()

        # 得到sift类
        sift = cv2.xfeatures2d.SIFT_create()

        res_src = []
        res_des = []

        for img, output in zip(imgs_tensor, outputs_tensor):
            # 装换成pil图像
            img_pil = F.to_pil_image(img)
            output_pil = F.to_pil_image(output)

            # 转化成cv2数据
            img_pil_cv2 = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
            output_pil_cv2 = cv2.cvtColor(np.asarray(output_pil), cv2.COLOR_RGB2BGR)

            # 转化为灰度图
            img_gray = cv2.cvtColor(img_pil_cv2, cv2.COLOR_BGR2GRAY)
            output_gray = cv2.cvtColor(output_pil_cv2, cv2.COLOR_BGR2GRAY)

            # 找到图像的关键点 kp为结构体 包含位置信息
            kp = sift.detect(img_gray, None)

            # 得到关键点和特征向量的矩阵
            kp, src = sift.compute(img_gray, kp)
            kp, des = sift.compute(output_gray, kp)

            # 修改大小 使得形状为(1024, 128)
            n = 1024
            if src.shape[0] < 1024:
                # 补0
                #src = np.pad(src, ((0, n - src.shape[0]), (0, 0)))
                #des = np.pad(des, ((0, n - des.shape[0]), (0, 0)))
                src = np.pad(src, ((0, n - src.shape[0]), (0, 0)), 'constant',constant_values = (0,0))
                des = np.pad(des, ((0, n - des.shape[0]), (0, 0)), 'constant',constant_values = (0,0))
            else:  # 切除
                src = np.resize(src, (n, src.shape[1]))
                des = np.resize(des, (n, src.shape[1]))

            # 增加通道维度 此时shape(1, 1024, 128)
            src = np.expand_dims(src, 0)
            des = np.expand_dims(des, 0)

            res_src.append(src)
            res_des.append(des)

        res_src = torch.tensor(np.array(res_src)).cuda()
        res_des = torch.tensor(np.array(res_des)).cuda()

        # InstanceNorm2d归一化
        in_norm = nn.InstanceNorm2d(1)
        res_src = in_norm(res_src)
        res_des = in_norm(res_des)

        # 返回cuda tesnsor
        return res_src, res_des


    def __call__(self, x, y): # x : outputs, y : images
        # Compute sift features
        im, op = self.getSift(y, x)
        sift_loss = self.criterion(op, im)

        return sift_loss

