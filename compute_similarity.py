import h5py
import torch
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image


def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # center crop
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    # img = scipy.misc.imresize(img, [height, width])
    img = np.array(Image.fromarray(img).resize((height, width)))
    return img


'''
type(output):tensor
type(filePath):tuple
'''
def compute_similarity(output, filePath, model):

    flag = False
    for i in range(len(filePath)):
        vec, sim_ = open_h5file(filePath[i])
        vec = torch.tensor(vec).cuda() 
        sim_ = torch.tensor(sim_).cuda() 
        sim_ = sim_.unsqueeze(0) # shape(1, top)

        output_single = F.to_pil_image(output[i].cpu()).resize((224, 224))# shape(3, 224, 224)

        output_single = F.to_tensor(output_single).float().unsqueeze(0) # shape(1, 3, 224, 224)

        output_single = output_single.cuda()

        query_vec = model.extract_features(output_single) # shape(1, 512)
        sim = torch.matmul(query_vec, vec.T)#shape(1, top)
        # sim = sim.unsqueeze(0)

        if flag == False:
            similarity_ = sim_
            similarity = sim
            flag = True
            continue

        similarity_ = torch.cat((similarity_, sim_))
        similarity = torch.cat((similarity, sim))

    return similarity_, similarity # cuda tensor



def compute_similarity_edge_v1(output, filePath, model):

    flag = False
    for i in range(len(filePath)):
        vec, sim_ = open_h5file(filePath[i])
        vec = torch.tensor(vec).cuda() 
        sim_ = torch.tensor(sim_).cuda() 
        sim_ = sim_.unsqueeze(0) # shape(1, top)

        # output_single = F.to_pil_image(output[i].cpu()).resize((224, 224))# shape(3, 224, 224)
        #
        # output_single = F.to_tensor(output_single).float().unsqueeze(0) # shape(1, 3, 224, 224)

        output_single = output[i].unsqueeze(0) # shape(1, 3, 224, 224)

        output_single = output_single.cuda()

        query_vec = model.extract_features(output_single) # shape(1, 512)
        sim = torch.matmul(query_vec, vec.T)#shape(1, top)
        # sim = sim.unsqueeze(0)

        if flag == False:
            similarity_ = sim_
            similarity = sim
            flag = True
            continue

        similarity_ = torch.cat((similarity_, sim_))
        similarity = torch.cat((similarity, sim))

    return similarity_, similarity # cuda tensor

def open_h5file(filePath):
    h5f = h5py.File(filePath, 'r')
    vector = h5f['vector'][:]
    similarity = h5f['similarity'][:]
    h5f.close()
    return vector, similarity

# '''
# # 计算修补后的local与支撑图片的local的相似度
# '''
# def compute_similarity_local(output, hole_area, local_vector, model):
#
#     output_local = crop(output, hole_area)
#     output_local_vec = model.extract_features(output_local) # tensor(n, 512)
#
#     batch_sim = []
#     for i in range(output_local_vec.shape[0]): # 循环16次
#         output_local_vec_single = output_local_vec[i].unsqueeze(0)
#
#         batch_tempt = []
#         for j in range(local_vector.shape[0]): # 循环10次
#             batch_tempt.append(local_vector[j][i].unsqueeze(0))
#         sim_local_vector_single = torch.cat(batch_tempt)
#
#         similarity = torch.matmul(output_local_vec_single, sim_local_vector_single.T) # (1, 10)
#         batch_sim.append(similarity)
#
#     local_sim = torch.cat(batch_sim)
#
#     return local_sim
