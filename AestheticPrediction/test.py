from model.base import Base
from model.head import Head
import torch
import h5py
import numpy as np
import os
from utils import build_h5,extract_features, get_model_from_ckpt,train,eval,lr_rescheduler, makedirs,save_model,logging
from torch.optim import Adam
import gc
from dataloader import get_dataloader
import matplotlib.pyplot as plt
from PIL import Image
import os
from torchsummary import summary
from model.finetune_model import Fmodel
import argparse
torch.manual_seed(0)
torch.cuda.set_per_process_memory_fraction(0.3, 0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_type", help = 'inceptionv3 or inceptionresnetv2', type = str, default = 'inceptionv3')
parser.add_argument("--feature_type", help = 'narrow or wide', type = str, default = 'narrow')
parser.add_argument("--resize", action ='store_true')
parser.add_argument("--augment", action ='store_true')
parser.add_argument("--fraction", type = float, help = 'fraction of data', default = 1)
parser.add_argument("--hard_dp", help = 'whether to use hard dropout', action ='store_true')
parser.add_argument("--num_level", type = int, help = 'number of level of features, if not specified then it means all of the features would be used',default=11)
parser.add_argument("--head_type", type = str, help = "single_1FC, single_3FC, multi_3FC, pool_3FC")
parser.add_argument("--continue_training", action ='store_true')
parser.add_argument("--ckpt",  help = 'checkpoint to load from', type = str)
parser.add_argument("--finetune",  help='whether in finetune mode or not', action ='store_true')
args = parser.parse_args()

base_model_type = args.base_model_type
feature_type = args.feature_type
resize = args.resize
augment = args.augment
fraction = args.fraction
num_level = args.num_level
continue_training = args.continue_training
head_type = args.head_type
hard_dp = args.hard_dp
ckpt = args.ckpt
finetune = args.finetune
if hard_dp:
    dropout = [0.5,0.5,0.75]
else:
    dropout = [0.25,0.25,0.5]

if finetune:
    n_aug=20
else:
    n_aug=8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_folder,fea_folder = makedirs('./experiment/',base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp)
h5_paths = [os.path.join(os.path.split(fea_folder)[0] if split!='train' else fea_folder, split + '_fea.h5') for split in ['train', 'val', 'test']]
batch_size = 128

bmodel = Base(model = base_model_type)
head = Head(head_type,bmodel.channel_size,dropout,num_level)
#print(summary(head,(10048,)))

if not finetune:
    head.to(device)
    dataloaders = get_dataloader(data_type='fea', h5_paths = h5_paths, batch_size = batch_size)
    model,_,_,_,_,_ = get_model_from_ckpt(model = head,
     ckpt = ckpt)
else:
    fmodel = Fmodel(base_model_type,feature_type, head, num_level,bmodel)
    dataloaders = get_dataloader(data_type='img', resize = True, augment= augment, batch_size = batch_size, finetune = True)
    fmodel.to(device)
    model,_,_,_,_,_ = get_model_from_ckpt(model = fmodel,
     ckpt = ckpt)

loss_fn = torch.nn.MSELoss()
plcc, srcc, acc, loss = eval(model,iter(dataloaders[2]),loss_fn,device,n_aug)
print(f'Test mse-{loss}, acc-{acc}, srcc-{srcc}, plcc-{plcc}')


# hard_dp = False
# if hard_dp:
#     dropout = [0.5,0.5,0.75]
# else:
#     dropout = [0.25,0.25,0.5]

# base_model_type = 'inceptionv3'
# num_level = 11
# feature_type = 'narrow'
# head_type = 'single_1FC'
# if feature_type == 'wide':
#     head_type = 'pool_3FC'
    
# resize = True
# augment = False
# finetune = True
# n_aug = 8