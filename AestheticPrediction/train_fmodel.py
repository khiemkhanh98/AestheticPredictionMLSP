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
import warnings
warnings.filterwarnings("ignore")
#torch.cuda.set_per_process_memory_fraction(0.6, 0)

hard_dp = False
base_model_type = 'inceptionv3'
num_level = 11
feature_type = 'narrow'
head_type = 'single_3FC'
if feature_type == 'wide':
    head_type = 'pool_3FC'
    
root = './experiment/'
resize = True
augment = True
finetune = True
n_aug = 20
num_level=11

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_folder,fea_folder = makedirs(root,base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp)
splits = ['train', 'val', 'test']
batch_size = 32
imgloaders = get_dataloader(data_type='img', resize = True, augment= augment, batch_size = batch_size, finetune = finetune)
lr = 1e-4
loss_fn = torch.nn.MSELoss()

bmodel = Base(model = base_model_type)
head = Head(head_type,bmodel.channel_size,hard_dp,num_level)
head,_,_,_,_,_ = get_model_from_ckpt(ckpt_folder, head, 'head', 'vloss', \
                ckpt='experiment/inceptionv3/11/narrow/single_3FC/resize+aug/head_ep19_vloss0.38_plcc0.60_srcc0.59_acc0.77_.pth')
fmodel = Fmodel(base_model_type,feature_type, head, num_level,bmodel)
fmodel.to(device)
optimizer = torch.optim.Adam(fmodel.parameters(),lr = lr)

latest_ep = 0
epochs = 500
# mini_iter = len(imgloaders[0])//4
mini_iter = None
best_train_loss = 1000
best_val_loss = 1000
patience_ep = 0
patience_tloss = 0
continue_training = False

if continue_training:
    fmodel,optimizer,latest_ep,best_train_loss,patience_tloss,patience_ep = get_model_from_ckpt(ckpt_folder, fmodel, 'fmodel','ep',optimizer)
fmodel.unfreeze() 
print(summary(fmodel,(3,299,299)))
print('latest epoch: ', latest_ep)
#train_iterator = iter(imgloaders[0])
for epoch in range(epochs): 
    # if epoch%4==0:
        # train_iterator = iter(imgloaders[0])
    patience_ep += 1
    epoch = int(latest_ep+1)+epoch
    fmodel,loss = train(fmodel,iter(imgloaders[0]),device,loss_fn,optimizer,mini_iter,epoch)
    
    if epoch%1 == 0:
        plcc, srcc, acc, val_loss = eval(fmodel,iter(imgloaders[1]),loss_fn,device,n_aug)
        logging(epoch, loss, val_loss,acc,plcc,srcc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        save_model(fmodel,optimizer,ckpt_folder,epoch,plcc,srcc,acc,val_loss,best_train_loss,patience_ep,patience_tloss,'fmodel')

    if  loss > best_train_loss:
        patience_tloss+=1
    else:
        best_train_loss = loss
        patience_tloss = 0

    if (patience_ep==20) or (patience_tloss==5):
        optimizer = lr_rescheduler(optimizer)
        patience_tloss = 0
        patience_ep = 0
    

