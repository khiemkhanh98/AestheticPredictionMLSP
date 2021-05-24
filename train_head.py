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
from model.finetune_model import Fmodel
from torchsummary import summary
torch.cuda.set_per_process_memory_fraction(0.5, 0)
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
ckpt_folder,fea_folder = makedirs(root,base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp)
h5_paths = [os.path.join(os.path.split(fea_folder)[0] if split!='train' else fea_folder, split + '_fea.h5') for split in ['train', 'val', 'test']]
train_batch_size = 128
fealoaders = get_dataloader(data_type='fea', resize = resize, augment= augment, h5_paths = h5_paths, batch_size = train_batch_size)

continue_training = True
lr = 1e-4
head = Head(head_type,fealoaders[0].dataset.channel_size,dropout)
head.to(device)
#print(summary(head,(10048,)))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(head.parameters(),lr = lr)
latest_ep = 0
epochs = 500
mini_iter = None
best_train_loss = 1000
best_val_loss = 1000
patience_ep = 0
patience_tloss = 0

if continue_training:
    head,optimizer,latest_ep,best_train_loss,patience_tloss,patience_ep = get_model_from_ckpt(ckpt_folder, head, 'head', 'ep',optimizer,ckpt='experiment/inceptionv3/11/narrow/single_3FC/resize/head_ep4_vloss0.41_plcc0.55_srcc0.54_acc0.76_.pth')

for epoch in range(epochs): 
    patience_ep += 1
    epoch = int(latest_ep+1)+epoch
    head,loss = train(head,fealoaders[0],device,loss_fn,optimizer,mini_iter,epoch)
    if epoch%1 == 0:
        plcc, srcc, acc, val_loss = eval(head,fealoaders[1],loss_fn,device,n_aug)
        logging(epoch, val_loss,acc,plcc,srcc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(head,optimizer,ckpt_folder,epoch,plcc,srcc,acc,val_loss,best_train_loss,patience_ep,patience_tloss,'head')

    if  loss > best_train_loss:
        patience_tloss+=1
    else:
        best_train_loss = loss
        patience_tloss = 0

    if (patience_ep==20) or (patience_tloss==5):
        optimizer = lr_rescheduler(optimizer)
        patience_tloss = 0
        patience_ep = 0
    

