import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import InpaintGenerator, EdgeGenerator, Discriminator
from .loss import AdversarialLoss, PerceptualLoss, StyleLoss, SimilarityLoss, SimilarityLoss_Edge_v1, SimilarityLoss_ImageGray, DescriptorLoss
from .PRVSNet import PRVSNet, VGG16FeatureExtractor

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.epoch = 0

        self.path = config.PATH

        self.gen_weights_path = os.path.join(config.PATH, name + '_gen.pth')
        self.dis_weights_path = os.path.join(config.PATH, name + '_dis.pth')

    def load(self, gen_path, dis_path):
        if os.path.exists(gen_path):
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(gen_path)
            else:
                data = torch.load(gen_path, map_location=lambda storage, loc: storage)

            self.generator.load_state_dict(data['generator'])
            self.iteration = data['iteration']
            self.epoch = data['epoch']

        # load discriminator only when training
        if self.config.MODE == 1 and os.path.exists(dis_path):
            print('Loading %s discriminator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(dis_path)
            else:
                data = torch.load(dis_path, map_location=lambda storage, loc: storage)

            self.discriminator.load_state_dict(data['discriminator'])

    def save(self, strs):
        print('\nsaving %s...\n' % self.name)

        gen_weights_path = os.path.join(self.path, self.name + strs + '_gen.pth')
        dis_weights_path = os.path.join(self.path, self.name + strs + '_dis.pth')

        torch.save({
            'iteration': self.iteration,
            'epoch': self.epoch,
            'generator': self.generator.state_dict()
        }, gen_weights_path)

        torch.save({
            'discriminator': self.discriminator.state_dict()
        }, dis_weights_path)


class InpaintingModel(BaseModel):
    def __init__(self, config):
        super(InpaintingModel, self).__init__('InpaintingModel', config)

        # generator input: [rgb(3) + edge(1)]
        # discriminator input: [rgb(3)]
        # generator = InpaintGenerator()
        generator = PRVSNet()
        discriminator = Discriminator(in_channels=3, use_sigmoid=config.GAN_LOSS != 'hinge')
        if len(config.GPU) > 1:
            generator = nn.DataParallel(generator, config.GPU)
            discriminator = nn.DataParallel(discriminator , config.GPU)

        l1_loss = nn.L1Loss()
        perceptual_loss = PerceptualLoss()
        style_loss = StyleLoss()
        adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        sift_loss = DescriptorLoss()

        similarity_loss = SimilarityLoss()


        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        self.add_module('l1_loss', l1_loss)
        self.add_module('perceptual_loss', perceptual_loss)
        self.add_module('style_loss', style_loss)
        self.add_module('adversarial_loss', adversarial_loss)
        self.add_module('similarity_loss', similarity_loss)
        self.add_module('sift_loss', sift_loss)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(config.LR) * float(config.D2G_LR),
            betas=(config.BETA1, config.BETA2)
        )

    '''
    修改6 process加一个参数
    '''
    def process(self, images, edges, masks, vec_sim_path):
        self.iteration += 1

        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()


        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0


        # discriminator loss
        dis_input_real = images
        dis_input_fake = outputs.detach()
        dis_real, _ = self.discriminator(dis_input_real)                    # in: [rgb(3)]
        dis_fake, _ = self.discriminator(dis_input_fake)                    # in: [rgb(3)]
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2


        # generator adversarial loss
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)                    # in: [rgb(3)]
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        gen_loss += gen_gan_loss


        # generator l1 loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.config.L1_LOSS_WEIGHT / torch.mean(masks)
        gen_loss += gen_l1_loss


        # generator perceptual loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.config.CONTENT_LOSS_WEIGHT
        gen_loss += gen_content_loss


        # generator style loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.config.STYLE_LOSS_WEIGHT
        gen_loss += gen_style_loss

        '''
        修改8 similarity loss
        '''
        gen_sim_loss = self.similarity_loss(outputs, vec_sim_path)
        gen_sim_loss = gen_sim_loss * self.config.SIM_LOSS_WEIGHT
        gen_loss += gen_sim_loss


        '''
        # 增加的loss
        # generator sift loss
        '''
        gen_sift_loss = self.sift_loss(outputs, images)
        gen_sift_loss = gen_sift_loss * self.config.SIFT_LOSS_WEIGHT
        gen_loss += gen_sift_loss



        # create logs
        logs = [
            ("l_d2", dis_loss.item()),
            ("l_g2", gen_gan_loss.item()),
            ("l_l1", gen_l1_loss.item()),
            ("l_per", gen_content_loss.item()),
            ("l_sty", gen_style_loss.item()),
            ("sim_l", gen_sim_loss.item()),
            ("sift_l", gen_sift_loss.item()),
        ]

        return outputs, gen_loss, dis_loss, logs

    def forward(self, images, edges, masks):

        masked_images = (images * (1 - masks).float()) + masks
        # masked_images = images * masks[:, 0:1, :, :]

        masks = torch.cat([1 - masks] * 3, dim=1)
        masks = masks[:, :3, :, :]

        outputs, _ = self.generator(masked_images, masks)

        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        dis_loss.backward()
        self.dis_optimizer.step()

        gen_loss.backward()
        self.gen_optimizer.step()
