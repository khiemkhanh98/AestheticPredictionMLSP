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
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

hard_dp = False
if hard_dp:
    dropout = [0.5,0.5,0.75]
else:
    dropout = [0.25,0.25,0.5]

base_model_type = 'inceptionv3'
num_level = 11
feature_type = 'narrow'
head_type = 'single_3FC'
if feature_type == 'wide':
    head_type = 'pool_3FC'
    
root = './experiment/'
resize = True
augment = False
n_aug = 8


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder,fea_folder = makedirs(root,base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp)
h5_paths = [os.path.join(os.path.split(fea_folder)[0] if split!='train' else fea_folder, split + '_fea.h5') for split in ['train', 'val', 'test']]
batch_size = 128
fealoaders = get_dataloader(data_type='fea', h5_paths = h5_paths, batch_size = batch_size)

head = Head(head_type,fealoaders[0].dataset.channel_size,dropout)
head.to(device)
head,_,_,_,_,_ = get_model_from_ckpt(model = head,
     ckpt = 'experiment/inceptionv3/11/narrow/single_3FC/resize/head_ep29_vloss0.39_plcc0.62_srcc0.60_acc0.78_.pth')
loss_fn = torch.nn.MSELoss()
plcc, srcc, acc, loss = eval(head,fealoaders[2],loss_fn,device,n_aug)
print(f'Test mse-{loss}, acc-{acc}, srcc-{srcc}, plcc-{plcc}')
