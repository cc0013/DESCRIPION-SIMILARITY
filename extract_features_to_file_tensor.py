
import os
import h5py
import numpy as np
import argparse
# import torch
# import torchvision.transforms as transforms
import imghdr
from torch.utils.data import DataLoader
from net_tensor import VGGNet
from tqdm import tqdm
import os
import glob
import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray, gray2rgb

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help="Path to database which contains images to be indexed")
parser.add_argument('index', help="Name of index file")
parser.add_argument('--cn_input_size', type=int, default=224)
parser.add_argument('--bsize', type=int, default=16)
parser.add_argument('--recursive_search', action='store_true', default=False)




class Dataset(torch.utils.data.Dataset):
    def __init__(self, flist):
        super(Dataset, self).__init__()

        self.data = self.load_flist(flist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):

        # load image
        img = imread(self.data[index])

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        #resize->256->224
        img = self.resize(img)

        return self.to_tensor(img)

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        #img = scipy.misc.imresize(img, [height, width])
        img = np.array(Image.fromarray(img).resize((256, 256)).resize((224, 224)))
        return img

    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]

        return []

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item


def main(args):

    feats = np.empty((args.bsize, 512), dtype=np.float32)
    names = [] 

    # Preparation
    args.data_dir = os.path.expanduser(args.data_dir)
    if torch.cuda.is_available() == False:
        raise Exception("At least one gpu must be available.")
    else:
        gpu = torch.device('cuda:0')

    dataset = Dataset(args.data_dir)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.bsize,
        num_workers=4,
        shuffle=False
    )

    # extract features

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    model_vgg = VGGNet()
    pbar = tqdm(total=len(data_loader))

    flag = False
    for x in data_loader:
        x = x.to(gpu)
        features = model_vgg.extract_features(x) # tensor
        features = features.cpu().detach().numpy()

        if flag == False:
            feats = features
            flag = True
            continue

        feats = np.concatenate((feats, features)) # numpy.ndarray

        pbar.update()

    pbar.close()

    names = get_imlist(args.data_dir)
    names = np.string_(names)

    # Save features
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(args.index, 'w')
    h5f.create_dataset('dataset_1', data=feats) # type(feats):numpy.ndarray
    h5f.create_dataset('dataset_2', data=names) # type(names):numpy.ndarray
    h5f.close()

    print("write successfuly!")

def get_imlist(path):
    return [f for f in os.listdir(path) if is_imfile(f)==True]


def is_imfile(filepath):
    filepath = os.path.join(args.data_dir, filepath)
    filepath = os.path.expanduser(filepath)
    if os.path.isfile(filepath) and imghdr.what(filepath):
        return True
    else:
        return False

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
