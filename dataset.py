import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
# import pyflow
from skimage import img_as_float
from random import randrange
import os.path
from PIL import ImageFile
import torchvision.transforms as transforms
import cv2
#import pyflow
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn.functional as F

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_rar_file(filename):
    return any(filename.endswith(extension) for extension in [".rar", ".zip"])


def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    # random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []

        for i in seq:
            index = int(filepath[char_len - 7:char_len - 4]) - i
            file_name = filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png'

            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png').convert('RGB'),
                               scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath, 'im' + str(nFrames) + '.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC) for j in reversed(seq)]

    return target, input, neigbor


def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames / 2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []
        if nFrames % 2 == 0:
            seq = [x for x in range(-tt, tt) if x != 0]  # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt, tt + 1) if x != 0]
        # random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len - 7:char_len - 4]) + i
            file_name1 = filepath[0:char_len - 7] + '{0:03d}'.format(index1) + '.png'

            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

    else:
        target = modcrop(Image.open(join(filepath, 'im4.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4 - tt, 5 + tt) if x != 4]
        # random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
                (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC))
    return target, input, neigbor


def load_img_future_de_rain(filepath, nFrames, img_id):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    target, input, neigbor = None, None, None
    if filepath.split('/')[3].split('-')[0] == 'SPAC':
        targetPath = os.path.dirname(filepath) + '/' + filepath.split('/')[5].split('_')[0] + '_GT'
        target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
        input = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
        # print(targetPath + '/' + str(img_id).zfill(5) + '.jpg')
        # print(filepath + '/' + str(img_id).zfill(5) + '.jpg')
        # exit()
        neigbor = []
        seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
        for j in seq:
            neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
            # print(filepath + '/' + str(j).zfill(5) + '.jpg')
        # print(filepath + '/' + str(img_id).zfill(5) + '.jpg')
        # print(targetPath)
        # exit()
    elif filepath.split('/')[3].split('_')[0] == 'frames':
        target = Image.open(filepath + '/' + 'gtc-' + str(img_id) + '.jpg').convert('RGB')
        input = Image.open(filepath + '/' + 'rfc-' + str(img_id) + '.jpg').convert('RGB')  # .resize((888, 496))
        neigbor = []
        # print(filepath + '/' + 'gtc-5.jpg')
        # print(filepath + '/' + 'rfc-5.jpg')
        seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
        for j in seq:
            neigbor.append(Image.open(filepath + '/' + 'rfc-' + str(j) + '.jpg').convert('RGB'))
            # print(filepath + '/' + 'rfc-' + str(j) + '.jpg')
        if target.size == (889, 500):
            target = target.crop((0, 0, 888, 496))
            input = input.crop((0, 0, 888, 496))
            for j in range(len(neigbor)):
                neigbor[j] = neigbor[j].crop((0, 0, 888, 496))
    elif filepath.split('/')[3] == 'rain_real':
        # print(filepath + '/' + str(img_id).zfill(5) + '.jpg')
        # exit()
        target = None
        input = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
        input = input.resize((int(1280 * 0.8), int(720 * 0.8)), Image.ANTIALIAS)
        # input = input.resize((832, 512), Image.ANTIALIAS)
        neigbor = []
        seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
        for j in seq:
            tmp_nei = Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB')
            tmp_nei = tmp_nei.resize((int(1280 * 0.8), int(720 * 0.8)), Image.ANTIALIAS)
            # tmp_nei = tmp_nei.resize((832, 512), Image.ANTIALIAS)
            neigbor.append(tmp_nei)

        # exit()
    # if target is None:
    #     print('read false')
    #     exit()
    return target, input, neigbor


def load_img_future_de_rain_flow(filepath, nFrames, img_id, phase='train'):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    #target, input, neigbor, tar_rain = None, None, None, None
    
    # if filepath.split('/')[3].split('_')[1] == 'Flow':
    num_dir = filepath.split('/')[3].split('_')[0] + '_GT'  # t1_GT  a1_GT
    if phase == 'train':
        targetPath = 'Dataset/RainMotion/Train_GT/' + num_dir
    else:
        targetPath = 'Dataset/RainMotion/Test_GT/' + num_dir
    #target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #inputs = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #tar_rain = Image.open(filepath + '/rs-' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    neigbor = []
    target = []
    seq = [x for x in range(img_id - tt, img_id + 1 + tt)]
    # seq = [img_id]
    for j in seq:
        neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
        target.append(Image.open(targetPath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
    '''
    a = filepath.split('/')[-1].split('_')[0][1]
    b = filepath.split('/')[-1].split('_')[2][1]
    base_path = filepath + '/motion_{}_{}.txt'.format(a, b)
    motion = np.loadtxt(base_path, delimiter=',')
    tar_motion = np.ones([128, 128, 2])
    # tar_motion = np.ones([480, 640, 2])
    tar_motion[:, :, 0] = tar_motion[:, :, 0] * motion[img_id - 1][0]
    # tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][1]
    tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][2]
    '''
    # exit()
    if target is None:
        print('read false')
        exit()
    return target, neigbor




def load_img_future_de_rain_ntu(filepath, nFrames, img_id, phase='train'):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    #target, input, neigbor = None, None, None
    # if filepath.split('/')[3].split('_')[1] == 'Flow':
    num_dir = filepath.split('/')[3].split('_')[0] + '_GT'  # t1_GT  a1_GT
    if phase == 'train':
        targetPath = 'Dataset/NTURain/Train_GT/' + num_dir
    else:
        targetPath = 'Dataset/NTURain/Test_GT/' + num_dir
    #target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #inputs = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #tar_rain = Image.open(filepath + '/rs-' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    neigbor = []
    target = []
    seq = [x for x in range(img_id - tt, img_id + 1 + tt)]
    # seq = [img_id]
    for j in seq:
        neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
        target.append(Image.open(targetPath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
    '''
    a = filepath.split('/')[-1].split('_')[0][1]
    b = filepath.split('/')[-1].split('_')[2][1]
    base_path = filepath + '/motion_{}_{}.txt'.format(a, b)
    motion = np.loadtxt(base_path, delimiter=',')
    tar_motion = np.ones([128, 128, 2])
    # tar_motion = np.ones([480, 640, 2])
    tar_motion[:, :, 0] = tar_motion[:, :, 0] * motion[img_id - 1][0]
    # tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][1]
    tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][2]
    '''
    # exit()
    if target is None:
        print('read false')
        exit()
    return target, neigbor

def load_img_future_de_haze_revide(filepath, nFrames, img_id, phase='train'):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    #print(filepath)
    #target, input, neigbor = None, None, None
    # if filepath.split('/')[3].split('_')[1] == 'Flow':
    num_dir = filepath.split('/')[3]  # t1_GT  a1_GT
    if phase == 'train':
        targetPath = 'Dataset/REVIDE/Train_GT/' + num_dir
    else:
        targetPath = 'Dataset/REVIDE/Test_GT/' + num_dir
    #target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #inputs = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #tar_rain = Image.open(filepath + '/rs-' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    neigbor = []
    target = []
    seq = [x for x in range(img_id - tt, img_id + 1 + tt)]
    # seq = [img_id]
    for j in seq:
        neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
        target.append(Image.open(targetPath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
    '''
    a = filepath.split('/')[-1].split('_')[0][1]
    b = filepath.split('/')[-1].split('_')[2][1]
    base_path = filepath + '/motion_{}_{}.txt'.format(a, b)
    motion = np.loadtxt(base_path, delimiter=',')
    tar_motion = np.ones([128, 128, 2])
    # tar_motion = np.ones([480, 640, 2])
    tar_motion[:, :, 0] = tar_motion[:, :, 0] * motion[img_id - 1][0]
    # tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][1]
    tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][2]
    '''
    # exit()
    if target is None:
        print('read false')
        exit()
    return target, neigbor


def load_img_future_de_snow_kitti(filepath, nFrames, img_id, phase='train'):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    #print(filepath)
    #target, input, neigbor = None, None, None
    # if filepath.split('/')[3].split('_')[1] == 'Flow':
    num_dir = filepath.split('/')[3]  # t1_GT  a1_GT
    if phase == 'train':
        targetPath = 'Dataset/KITTI_snow/Train_GT/' + num_dir
    else:
        targetPath = 'Dataset/KITTI_snow/Test_GT/' + num_dir
    #target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #inputs = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    #tar_rain = Image.open(filepath + '/rs-' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    neigbor = []
    target = []
    seq = [x for x in range(img_id - tt, img_id + 1 + tt)]
    # seq = [img_id]
    for j in seq:
        neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
        target.append(Image.open(targetPath + '/' + str(j).zfill(5) + '.png').convert('RGB'))
    '''
    a = filepath.split('/')[-1].split('_')[0][1]
    b = filepath.split('/')[-1].split('_')[2][1]
    base_path = filepath + '/motion_{}_{}.txt'.format(a, b)
    motion = np.loadtxt(base_path, delimiter=',')
    tar_motion = np.ones([128, 128, 2])
    # tar_motion = np.ones([480, 640, 2])
    tar_motion[:, :, 0] = tar_motion[:, :, 0] * motion[img_id - 1][0]
    # tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][1]
    tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][2]
    '''
    # exit()
    if target is None:
        print('read false')
        exit()
    return target, neigbor





def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo);
    iw = iw - (iw % modulo);
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_tar, img_nn, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_tar[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    #img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = [j.crop((ty, tx, ty + tp, tx + tp)) for j in img_tar]  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]
    #img_rain = img_rain.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return  img_tar, img_nn, info_patch


def adjust_light(image, gamma):
    #gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)

    return image


def augment(img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}


    if random.random() < 0.5 and flip_h:
        #img_in = ImageOps.flip(img_in)
        img_tar = [ImageOps.flip(j) for j in img_tar]
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            #img_in = ImageOps.mirror(img_in)
            img_tar = [ImageOps.mirror(j) for j in img_tar]
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            #img_in = img_in.rotate(180)
            img_tar = [j.rotate(180) for j in img_tar]
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    #if random.random() < 0.5:
    #    gamma = random.random() * 3 + 0.5
    #    img_tar = [adjust_light(j, gamma) for j in img_tar]
    #    img_nn = [adjust_light(j, gamma) for j in img_nn]


    return img_tar, img_nn, info_aug


def rescale_img(img_in, new_size_in):
    #size_in = img_in.size
    #print(size_in)
    #new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    #print(img_in.size)
    return img_in

max_flow = 150.0
def rescale_flow(x,max_range,min_range):
    #remove noise
    x[x > max_flow] = max_flow
    x[x < -max_flow] = -max_flow
    
    max_val = max_flow 
    min_val = -max_flow 
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.
    
    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    #flow = rescale_flow(flow,-1,1)
    return flow


class MultiDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                 future_frame, transform=None):
        super(MultiDatasetFromFolder, self).__init__()
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        self.image_dir = image_dir
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame
        self.image_num = []
        self.index_compute = []
        self.image_filenames = []
        for i in range(len(image_dir)):
            alist = os.listdir(image_dir[i])  # image_dir : /dataset/frames_heavy_train_JPEG
            image_num = 0
            index_compute = []
            image_filenames = [join(image_dir[i], x) for x in alist]  #


            for j in range(len(image_filenames)):
                image_list = os.listdir(image_filenames[j])
                for img in image_list:
                    if img.endswith('jpg') and 'rs' not in img:
                        image_num += 1
                # image_num += len(os.listdir(self.image_filenames[i]))
                image_num = image_num - self.nFrames + 1
                index_compute.append(image_num)
            self.image_filenames.append(image_filenames)
            self.image_num.append(index_compute[-1])
            self.index_compute.append(index_compute)

    def __getitem__(self, index):
        samples = []
        for key in range(len(self.image_dir)):
            idx = index % self.image_num[key]
            index_compute = self.index_compute[key]
            image_filenames = self.image_filenames[key]
            file_id = 0
            idx = idx + 1
            for i in range(len(index_compute)):
                if index_compute[i] >= idx:
                    file_id = i
                    break
            img_id = idx if file_id == 0 else idx - int(index_compute[file_id - 1])

            if key == 0:
                target, neigbor = load_img_future_de_rain_flow(
                    image_filenames[file_id], self.nFrames, img_id, phase='train')
                #dc = torch.tensor([1,0,0]).float()
            elif key == 1:
                target, neigbor = load_img_future_de_haze_revide(
                    image_filenames[file_id], self.nFrames, img_id, phase='train')
                #dc = torch.tensor([1,1,0]).float()
            elif key == 2:
                target, neigbor = load_img_future_de_snow_kitti(
                    image_filenames[file_id], self.nFrames, img_id, phase='train')
                #dc = torch.tensor([0,0,1]).float()

            if self.patch_size != 0:
                target, neigbor, _ = get_patch(target, neigbor, self.patch_size, 1,
                                                            self.nFrames)

            if self.data_augmentation:
                target, neigbor, _ = augment(target, neigbor)

            #ds_flow = [rescale_img(j, (64,64)) for j in neigbor]
            #flow = [get_flow(ds_flow[int(self.nFrames/2)],j) for j in ds_flow]
            #print(flow[1].shape)
        
            #flow = [F.interpolate(j, scale_factor=4, mode='bicubic') for j in flow]
            #flow = [get_flow(neigbor[int(self.nFrames/2)],j) for j in neigbor]
        

            if self.transform:
                target = [self.transform(j) for j in target]
                #input = self.transform(input)
                #bicubic = [self.transform(j) for j in bicubic]
                neigbor = [self.transform(j) for j in neigbor]
                #flow = [self.transform(j) for j in flow]
                #tar_rain = self.transform(tar_rain)
                #tar_motion = self.transform(tar_motion)
                #flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            
            targets = torch.stack(target,0)
            neigbors = torch.stack(neigbor,0)
            #flows = torch.stack(flow,0)
            samples.append({'target':targets,'neigbor':neigbors,'dc':key})
            #print(neigbors.shape)
            #print(flow.shape)
        return samples

    def __len__(self):
        return max(self.image_num)


class DeRainDatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DeRainDatasetFromFolderTest, self).__init__()
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.other_dataset = other_dataset
        self.future_frame = future_frame

        self.image_num = 0
        self.index_compute = []

        self.image_filenames = [join(image_dir, x) for x in os.listdir(image_dir)]  #
        self.image_filenames = sorted(self.image_filenames)
        image_num = 0
        for i in range(len(self.image_filenames)):
            image_list = os.listdir(self.image_filenames[i])
            for img in image_list:
                if img.endswith('jpg') and 'rs' not in img:
                    image_num += 1
            # image_num += len(os.listdir(self.image_filenames[i]))
            image_num = image_num - self.nFrames + 1
            self.index_compute.append(image_num)
        self.image_num = self.index_compute[-1]

    def __getitem__(self, index):
        file_id = 0
        index = index + 1
        for i in range(len(self.index_compute)):
            if self.index_compute[i] >= index:
                file_id = i
                break
        img_id = index if file_id == 0 else index - int(self.index_compute[file_id - 1])
        #if not os.path.exists(
        #        'Results/TransWeather_NTURain/' + str(int(file_id) + 1) + '_' + str(1)):
        #    os.makedirs(
        #        'Results/TransWeather_NTURain/' + str(int(file_id) + 1) + '_' + str(1))
        if not os.path.exists(
                'Results/VIWSNET_RainMotion/' + str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1)):
            os.makedirs(
                'Results/VIWSNET_RainMotion/' + str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1))

        if self.future_frame:
            target, neigbor = load_img_future_de_rain_flow(
                self.image_filenames[file_id], self.nFrames, img_id, phase='test')

        #flow = [[] for j in neigbor]
        #ds_flow = [rescale_img(j, (64,64)) for j in neigbor]
        #flow = [get_flow(ds_flow[int(self.nFrames/2)],j) for j in ds_flow]

        #bicubic = rescale_img(input, self.upscale_factor)
        #if img_id == self.index_compute[file_id] or self.index_compute[file_id-1]+1:
        #    save_all=True
        #else:
        #    save_all=False
        #file = str(int(file_id) + 1) + '_' + str(1) + '/recover' + str(
        #    img_id + int(self.nFrames / 2)) + '.png'
        file = str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1) + '/recover' + str(
            img_id + int(self.nFrames / 2)) + '.png'        

        if self.transform:
            #target = self.transform(target)
            #target = []
            #input = self.transform(input)
            #bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            #flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]

            #tar_rain = self.transform(tar_rain)
            #tar_motion = self.transform(tar_motion)
        neigbors = torch.stack(neigbor,0)
        #flow = torch.stack(flow,0)
        return neigbors, file

    def __len__(self):
        return self.image_num  # ---------------


class DeSnowDatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DeSnowDatasetFromFolderTest, self).__init__()
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.other_dataset = other_dataset
        self.future_frame = future_frame

        self.image_num = 0
        self.index_compute = []

        self.image_filenames = [join(image_dir, x) for x in os.listdir(image_dir)]  #
        self.image_filenames = sorted(self.image_filenames)
        image_num = 0
        for i in range(len(self.image_filenames)):
            image_list = os.listdir(self.image_filenames[i])
            for img in image_list:
                if img.endswith('jpg') and 'rs' not in img:
                    image_num += 1
            # image_num += len(os.listdir(self.image_filenames[i]))
            image_num = image_num - self.nFrames + 1
            self.index_compute.append(image_num)
        self.image_num = self.index_compute[-1]

    def __getitem__(self, index):
        file_id = 0
        index = index + 1
        for i in range(len(self.index_compute)):
            if self.index_compute[i] >= index:
                file_id = i
                break
        img_id = index if file_id == 0 else index - int(self.index_compute[file_id - 1])
        if not os.path.exists(
                'Results/VIWSNET_KITTI/' + str(int(file_id) + 1).zfill(2)):
            os.makedirs(
                'Results/VIWSNET_KITTI/' + str(int(file_id) + 1).zfill(2))

        if self.future_frame:
            target, neigbor = load_img_future_de_snow_kitti(
                self.image_filenames[file_id], self.nFrames, img_id, phase='test')


        file = str(int(file_id) + 1).zfill(2) + '/recover' + str(
            img_id + int(self.nFrames / 2)) + '.png'
        #file = str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1) + '/recover' + str(
        #    img_id + int(self.nFrames / 2)) + '.png'        

        if self.transform:

            neigbor = [self.transform(j) for j in neigbor]

        neigbors = torch.stack(neigbor,0)
        #flow = torch.stack(flow,0)
        return neigbors, file

    def __len__(self):
        return self.image_num  # ---------------

class DeHazeDatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DeHazeDatasetFromFolderTest, self).__init__()
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        self.other_dataset = other_dataset
        self.future_frame = future_frame

        self.image_num = 0
        self.index_compute = []

        self.image_filenames = [join(image_dir, x) for x in os.listdir(image_dir)]  #
        self.image_filenames = sorted(self.image_filenames)
        image_num = 0
        for i in range(len(self.image_filenames)):
            image_list = os.listdir(self.image_filenames[i])
            for img in image_list:
                if img.endswith('jpg') and 'rs' not in img:
                    image_num += 1
            # image_num += len(os.listdir(self.image_filenames[i]))
            image_num = image_num - self.nFrames + 1
            self.index_compute.append(image_num)
        self.image_num = self.index_compute[-1]

    def __getitem__(self, index):
        file_id = 0
        index = index + 1
        for i in range(len(self.index_compute)):
            if self.index_compute[i] >= index:
                file_id = i
                break
        img_id = index if file_id == 0 else index - int(self.index_compute[file_id - 1])
        if not os.path.exists(
                'Results/VIWSNET_REVIDE/' + str(int(file_id) + 1).zfill(2)):
            os.makedirs(
                'Results/VIWSNET_REVIDE/' + str(int(file_id) + 1).zfill(2))

        if self.future_frame:
            target, neigbor = load_img_future_de_haze_revide(
                self.image_filenames[file_id], self.nFrames, img_id, phase='test')


        file = str(int(file_id) + 1).zfill(2) + '/recover' + str(
            img_id + int(self.nFrames / 2)) + '.png'
        #file = str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1) + '/recover' + str(
        #    img_id + int(self.nFrames / 2)) + '.png'        

        if self.transform:

            neigbor = [self.transform(j) for j in neigbor]
            #flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]

        neigbors = torch.stack(neigbor,0)
        #flow = torch.stack(flow,0)
        return neigbors, file

    def __len__(self):
        return self.image_num  # ---------------