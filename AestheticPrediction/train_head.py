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
import argparse
#torch.cuda.set_per_process_memory_fraction(0.4, 0)
torch.manual_seed(0)
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
args = parser.parse_args()

n_aug = 8
base_model_type = args.base_model_type
feature_type = args.feature_type
resize = args.resize
augment = args.augment
fraction = args.fraction
num_level = args.num_level
continue_training = args.continue_training
head_type = args.head_type
hard_dp = args.hard_dp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_folder,fea_folder = makedirs('./experiment/',base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp)
h5_paths = [os.path.join(os.path.split(fea_folder)[0] if split!='train' else fea_folder, split + '_fea.h5') for split in ['train', 'val', 'test']]
train_batch_size = 128
fealoaders = get_dataloader(data_type='fea', resize = resize, augment= augment, h5_paths = h5_paths, batch_size = train_batch_size)

lr = 1e-4
bmodel = Base(model = base_model_type)
head = Head(head_type,bmodel.channel_size,hard_dp,num_level)
del bmodel
head.to(device)
#print(summary(head,(np.sum(head.num_ch),5,5,)))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(head.parameters(),lr = lr)
latest_ep = 0
epochs = 500
mini_iter=None
if augment:
    mini_iter = len(fealoaders[0])//8
train_iterator = iter(fealoaders[0])
best_train_loss = 1000
best_val_loss = 1000
patience_ep = 0
patience_tloss = 0

if continue_training:
    head,optimizer,latest_ep,best_train_loss,patience_tloss,patience_ep = get_model_from_ckpt(ckpt_folder, head, 'head', 'ep',optimizer)

print('Latest epoch: ', latest_ep)
for epoch in range(epochs): 
    patience_ep += 1
    epoch = int(latest_ep+1)+epoch
    if (mini_iter!=None) and (epoch%8==0):
        train_iterator = iter(fealoaders[0])
    head,loss = train(head,train_iterator,device,loss_fn,optimizer,mini_iter,epoch)
    if epoch%1 == 0:
        plcc, srcc, acc, val_loss = eval(head,iter(fealoaders[1]),loss_fn,device,n_aug)
        logging(epoch, loss, val_loss,acc,plcc,srcc)

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
    

# hard_dp = False
# if hard_dp:
#     dropout = [0.5,0.5,0.75]
# else:
#     dropout = [0.25,0.25,0.5]

# base_model_type = 'inceptionv3'
# num_level = 5
# feature_type = 'narrow'
# head_type = 'single_3FC'
# if feature_type == 'wide':
#     head_type = 'pool_3FC'
    
# resize = True
# augment = False