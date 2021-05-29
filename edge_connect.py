import os
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import InpaintingModel
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy


class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'
        elif config.MODEL == 2:
            model_name = 'inpaint'
        elif config.MODEL == 3:
            model_name = 'edge_inpaint'
        elif config.MODEL == 4:
            model_name = 'joint'

        self.epoch = config.Epoch

        self.debug = False
        self.model_name = model_name
        self.inpaint_model = InpaintingModel(config).to(config.DEVICE)

        self.psnr = PSNR(255.0).to(config.DEVICE)
        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            # self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=False)
            self.test_dataset = Dataset(config, config.TEST_FLIST, config.TEST_EDGE_FLIST, config.TEST_MASK_FLIST, augment=False, training=True)
        else:
            self.train_dataset = Dataset(config, config.TRAIN_FLIST, config.TRAIN_EDGE_FLIST, config.TRAIN_MASK_FLIST, augment=True, training=True)
            self.val_dataset = Dataset(config, config.VAL_FLIST, config.VAL_EDGE_FLIST, config.VAL_MASK_FLIST, augment=False, training=True)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')

        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')

    def load(self, gen_path, dis_path):
        self.inpaint_model.load(gen_path, dis_path)

    def save(self, strs):
        self.inpaint_model.save(strs)


    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        epoch = self.inpaint_model.epoch
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)

        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):

            if epoch > 0:
                self.inpaint_model.epoch = epoch
                self.save("_epoch_" + str(epoch))

            if epoch >= self.epoch:
                break

            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])


            '''
            修改3.0
            '''
            for items , vec_sim_path in train_loader:

                self.inpaint_model.train()

                #此时的边缘为全部的边缘
                images, images_gray, edges, masks = self.cuda(*items)
                outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks, vec_sim_path)
                outputs_merged = (outputs * masks) + (images * (1 - masks))

                # metrics
                psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
                mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
                logs.append(('psnr', psnr.item()))
                logs.append(('mae', mae.item()))

                # backward
                self.inpaint_model.backward(gen_loss, dis_loss)
                iteration = self.inpaint_model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                progbar.add(len(images), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    # self.sample()
                    iteration = self.inpaint_model.iteration
                    inputs = (images * (1 - masks)) + masks
                    # outputs = self.inpaint_model(images, edges, masks)
                    # outputs_merged = (outputs * masks) + (images * (1 - masks))


                    image_per_row = 2
                    if self.config.SAMPLE_SIZE <= 6:
                        image_per_row = 1

                    images = stitch_images(
                        self.postprocess(images),
                        self.postprocess(inputs),
                        self.postprocess(edges),
                        self.postprocess(outputs),
                        self.postprocess(outputs_merged),
                        img_per_row=image_per_row
                    )

                    path = os.path.join(self.samples_path, self.model_name)
                    name = os.path.join(path, str(iteration).zfill(5) + ".png")
                    create_dir(path)
                    print('\nsaving sample ' + name)
                    images.save(name)


                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # print("**" + str(iteration) + "**")
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save("_" + str(int(iteration / self.config.SAVE_INTERVAL)))

        print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            drop_last=True,
            shuffle=True
        )

        total = len(self.val_dataset)

        self.inpaint_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        '''
        修改3.1
        '''
        for items , vec_sim_path in val_loader:
            iteration += 1
            images, images_gray, edges, masks = self.cuda(*items)


            outputs, gen_loss, dis_loss, logs = self.inpaint_model.process(images, edges, masks, vec_sim_path)
            outputs_merged = (outputs * masks) + (images * (1 - masks))

            # metrics
            psnr = self.psnr(self.postprocess(images), self.postprocess(outputs_merged))
            mae = (torch.sum(torch.abs(images - outputs_merged)) / torch.sum(images)).float()
            logs.append(('psnr', psnr.item()))
            logs.append(('mae', mae.item()))



            logs = [("it", iteration), ] + logs
            progbar.add(len(images), values=logs)

    def test(self):
        # self.edge_model.eval()
        self.inpaint_model.eval()

        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        '''
        修改3.2
        '''
        for items , _ in test_loader:
            name = self.test_dataset.load_name(index)
            images, images_gray, edges, masks = self.cuda(*items)
            index += 1

            outputs = self.inpaint_model(images, edges, masks)

            inputs = (images * (1 - masks)) + masks

            outputs_merged = (outputs * masks) + (images * (1 - masks))

            images = stitch_images(
                self.postprocess(images),

                self.postprocess(outputs_merged),

                self.postprocess(inputs),
                self.postprocess(edges),
                self.postprocess(outputs),
                img_per_row=1
            )


            # output = self.postprocess(outputs_merged)[0]
            path = os.path.join(self.results_path, name)
            print(index, name)
            images.save(path)
            # imsave(output, path)

            if self.debug:
                edges = self.postprocess(1 - edges)[0]
                masked = self.postprocess(images * (1 - masks) + masks)[0]
                fname, fext = name.split('.')

                imsave(edges, os.path.join(self.results_path, fname + '_edge.' + fext))
                imsave(masked, os.path.join(self.results_path, fname + '_masked.' + fext))

        print('\nEnd test....')

    def sample(self, it=None):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return

        # self.edge_model.eval()
        self.inpaint_model.eval()

        # model = self.config.MODEL
        items = next(self.sample_iterator)
        images, images_gray, edges, masks = self.cuda(*items)

        iteration = self.inpaint_model.iteration
        inputs = (images * (1 - masks)) + masks
        outputs = self.inpaint_model(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))

        if it is not None:
            iteration = it

        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1

        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(outputs),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )


        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
