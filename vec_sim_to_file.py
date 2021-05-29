

import h5py
import argparse
from net_tensor import VGGNet
from tqdm import tqdm
import os
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from imageio import imread
from skimage.color import rgb2gray, gray2rgb

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help="Path to train dataset")
parser.add_argument('index_file_path', help="Path to indexed file")
parser.add_argument('--cn_input_size', type=int, default=224)
parser.add_argument('--top', type=int, default=10)
parser.add_argument('--result_path', type=str, default="./vector_and_similarity")


def resize(img, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    # img = scipy.misc.imresize(img, [height, width])
    img = np.array(Image.fromarray(img).resize((256, 256)).resize((224, 224)).convert("RGB"))
    return img

def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def main(args):

    args.data_dir = os.path.expanduser(args.data_dir)
    args.index_file_path = os.path.expanduser(args.index_file_path)

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)


    # read in indexed images' features vector and corresponding images names
    h5f = h5py.File(args.index_file_path, 'r')
    feats = h5f['dataset_1'][:] # features of dataset(train)
    h5f.close()

    # init VGGNet16 model
    model = VGGNet()

    # image preprocess
    # transf = transforms.Compose([
    #     transforms.Resize(args.cn_input_size),
    #     transforms.CenterCrop(args.cn_input_size),
    #     transforms.ToTensor()
    # ])


    for path in tqdm(os.listdir(args.data_dir)):
        img_path = os.path.join(args.data_dir, path)

        # extract features
        # load image
        img = imread(img_path)

        # gray to rgb
        if len(img.shape) < 3:
            img = gray2rgb(img)

        #resize->256->224
        img = resize(img)
        img = to_tensor(img)

        # img = Image.open(img_path)
        # img = img.convert("RGB") #　！将图片模式统一为RGB模式
        # img = transf(img)
        img = img.unsqueeze(0)
        img = img.cuda()

        queryVector = model.extract_features(img).detach().cpu() #type(query): tensor(1, 512)
        features = torch.tensor(feats) # cpu tensor


        # scores = np.dot(queryVector, feats.T) #type(scores): ndarray(1, x),x为数据集的大小
        scores = torch.matmul(queryVector, features.T) #tensor
        scores = scores.numpy()
        scores = scores.squeeze(0)


        rank_ID = np.argsort(scores)[::-1] 
        rank_score = scores[rank_ID] 
        similar_vec = feats[rank_ID[0:args.top]] 


        similar_sco = rank_score[0:args.top] 
 
        '''
        name = os.path.join(args.result_path, path.split('.')[0] + '.h5')
        h5f1 = h5py.File(name, 'w')
        h5f1.create_dataset('vector', data=similar_vec)
        h5f1.create_dataset('similarity', data=similar_sco)
        h5f1.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
