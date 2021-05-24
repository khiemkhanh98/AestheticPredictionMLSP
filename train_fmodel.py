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
torch.cuda.set_per_process_memory_fraction(0.5, 0)

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
finetune = False
n_aug = 8


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder = makedirs(root,base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp)
splits = ['train', 'val', 'test']
#h5_paths = [os.path.join(folder,split + '_fea.h5') for split in splits]
train_batch_size = 128
imgloaders = get_dataloader(data_type='img', head_type = head_type, resize = resize, augment= augment, batch_size = train_batch_size)

continue_training = False
lr = 1e-4
loss_fn = torch.nn.MSELoss()
bmodel = Base(model = base_model_type, level = num_level)
head = Head(head_type,bmodel.channel_size,dropout)

#head,_,_,_,_,_ = get_model_from_ckpt(folder, head, 'head', 'vloss')
fmodel = Fmodel(base_model_type,feature_type, head, num_level,bmodel)
optimizer = torch.optim.Adam(fmodel.parameters(),lr = lr)
if continue_training:
    fmodel,optimizer,latest_ep,best_train_loss,patience_tloss,patience_ep = get_model_from_ckpt(folder, fmodel, 'fmodel', 'ep',optimizer)
fmodel.to(device)

latest_ep = 0
epochs = 500
mini_iter = None
best_train_loss = 1000
best_val_loss = 1000
patience_ep = 0
patience_tloss = 0

for epoch in range(epochs): 
    patience_ep += 1
    epoch = int(latest_ep+1)+epoch
    fmodel,loss = train(fmodel,imgloaders[0],device,loss_fn,optimizer,mini_iter,epoch)
    if epoch%1 == 0:
        plcc, srcc, acc, val_loss = eval(fmodel,imgloaders[1],loss_fn,device,n_aug)
        logging(epoch, val_loss,acc,plcc,srcc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        save_model(fmodel,optimizer,folder,epoch,plcc,srcc,acc,val_loss,best_train_loss,patience_ep,patience_tloss,'fmodel')

    if  loss > best_train_loss:
        patience_tloss+=1
    else:
        best_train_loss = loss
        patience_tloss = 0

    if (patience_ep==20) or (patience_tloss==5):
        optimizer = lr_rescheduler(optimizer)
        patience_tloss = 0
        patience_ep = 0
    

