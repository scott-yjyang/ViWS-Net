from __future__ import print_function
import argparse
from math import log10

import os, logging
os.environ["CUDA_VISIBLE_DEVICES"] = "0,5"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.models import vgg16
from modeling.model import ViWSNet

from data import get_training_set_derain
import pdb
import socket
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime, sys
import numpy as np
import random



# Training settings batchSize和卡数相同
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=19, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=list, default=['Dataset/RainMotion/Train','Dataset/REVIDE/Train','Dataset/KITTI_snow/Train'])  #
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=224, help='0 to use original frame size')
parser.add_argument('--lr_decay', type=float, default=0.5, help='lr decay')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='ViWSNet')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='.pth',
                    help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--log_path', default='logs/', help='Location to save logging')


opt = parser.parse_args()
# gpus_list = range(0, opt.gpus)
# gpus_list = [0, 1, 2]
gpus_list = [0,1]
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

torch.manual_seed(opt.seed)
random.seed(opt.seed)
np.random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)
torch.backends.cudnn.benchmark = True


log_root = os.path.join(opt.log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                            )
print("log path:", log_root)
os.makedirs(log_root, exist_ok=True)
writer = SummaryWriter(log_dir=log_root)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)




min_loss = 1e6
def train(epoch):
    epoch_loss = 0
    #cls_loss = 0
    total_cl_loss = 0
    total_cls_loss = 0
    model.train()
    for iteration, sample in tqdm(enumerate(training_data_loader, 1)):
        target = None
        neigbor = None
        #flow = None
        dc = None
        for weather in sample:
            if target is None:
                target = weather['target']
                neigbor = weather['neigbor']
                #flow = weather['flow']
                dc = weather['dc']
            else:
                target = torch.cat([target, weather['target']], 0)
                neigbor = torch.cat([neigbor, weather['neigbor']], 0)
                #flow = torch.cat([flow, weather['flow']], 0)
                dc = torch.cat([dc, weather['dc']], 0)
        #print(dc)
        if neigbor.shape[0] % 2 == 1:
            neigbor = torch.cat([neigbor,neigbor[0:1]],0)
            target = torch.cat([target,target[0:1]],0)
            #flow = torch.cat([flow,flow[0:1]],0)
            dc = torch.cat([dc,dc[0:1]],0)
        
        B = neigbor.shape[0]
        
        perm = torch.randperm(B)
        neigbor = neigbor[perm]
        target = target[perm]
        dc = dc[perm]


        #print(neigbor.shape)
        #bicubic = neigbor.reshape(neigbor.shape[0]*neigbor.shape[1],neigbor.shape[2],neigbor.shape[3],neigbor.shape[4])
        target = target.reshape(target.shape[0]*target.shape[1],target.shape[2],target.shape[3],target.shape[4])
        
        if cuda:
            target = target.cuda()
            neigbor = neigbor.cuda()
            dc = dc.cuda()

        optimizer.zero_grad()
        t0 = time.time()
        
        p = float(iteration + epoch * len(training_data_loader)) / opt.nEpochs / len(training_data_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        try:
            #prediction = model(neigbor, flow, B//len(gpus_list), opt.nFrames)
            prediction, pred_dc = model(neigbor, B//len(gpus_list), opt.nFrames, alpha, phase='train')
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        
        
        centre_frame = prediction[int(opt.nFrames/2)::opt.nFrames]
        centre_target = target[int(opt.nFrames/2)::opt.nFrames]
        # centre_dc = pred_dc[int(opt.nFrames/2)::opt.nFrames]
        cls_loss = 0.001*cls_crtierion(pred_dc,dc)
        loss = criterion(centre_frame, centre_target) + 0.04 * loss_network(centre_frame, centre_target) + cls_loss
        

        # # exit()
        t1 = time.time()
        epoch_loss += loss.data
        total_cls_loss += cls_loss.data

        loss.backward()

        optimizer.step()


        # torch.cuda.empty_cache()
        writer.add_scalar('train' + '/loss_iter', loss, epoch*len(training_data_loader)+iteration)

    writer.add_scalar('train'+'/loss_epoch', (epoch_loss-total_cls_loss) / len(training_data_loader), epoch)
    writer.add_scalar('train'+'/loss_cls', total_cls_loss / len(training_data_loader), epoch)
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}, Wea. Loss: {:.6f}, CL. Loss: {:.6f}".format(epoch, (epoch_loss-total_cls_loss) / len(training_data_loader), total_cls_loss / len(training_data_loader), total_cl_loss / len(training_data_loader)))
    # --- Calculate the average training PSNR in one epoch --- #    global min_loss
    global min_loss
    if (epoch_loss-total_cls_loss) / len(training_data_loader) < min_loss:
        min_loss = (epoch_loss-total_cls_loss) / len(training_data_loader)
        model_out_path = os.path.join(log_root, opt.model_type + "_epoch_best.pth")
        torch.save(model.state_dict(), model_out_path)
        #print("Checkpoint saved to {}".format(model_out_path))



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = os.path.join(log_root, opt.model_type + "_epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set_derain(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list,
                                    opt.other_dataset, opt.patch_size, opt.future_frame)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'ViWSNet':
    params = dict(
    
        finetune = './models/ckpt_S.pth',
        hidden_dim=512,
        dropout=0.1,
        nheads=8,
        dim_feedforward=2048,
        dec_layers=6,
        num_queries=48,
        num_types = 3
    )
    model = ViWSNet(params)



torch.cuda.set_device(gpus_list[0])
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.SmoothL1Loss()
cls_crtierion = nn.CrossEntropyLoss()

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.cuda()
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')


if opt.pretrained:
    model_name = os.path.join(opt.pretrained_sr)
    if os.path.exists(model_name):
        print(model_name)
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()
    cls_crtierion = cls_crtierion.cuda()



optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.80)
for epoch in tqdm(range(opt.start_epoch, opt.nEpochs + 1)):
    train(epoch)

    if (epoch + 1) % 100 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr_decay * param_group['lr']
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch + 1) % (opt.snapshots) == 0:
        checkpoint(epoch+1)
