import numpy as np
import h5py
import os
import glob
import re
import torch
from scipy.stats import pearsonr
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

def metrics(pred, target, n_aug):
    if n_aug!=1:
        target = np.array(target).reshape(n_aug,-1).mean(axis=0)
        pred = np.array(pred).reshape(n_aug,-1).mean(axis=0)
    else:
        target = np.array(target).reshape(-1)
        pred = np.array(pred).reshape(-1)

    acc = np.mean((pred >= 5) == (target>=5))
    plcc = pearsonr(pred, target)[0]
    xranks = pd.Series(pred).rank()
    yranks = pd.Series(target).rank()
    srcc = pearsonr(xranks, yranks)[0]
    return plcc, srcc, acc


def get_model_from_ckpt(folder=None,model=None, model_type=None,metric=None, optimizer = None,ckpt=None):
    if ckpt==None:
        checkpoints = glob.glob(os.path.join(folder,f'{model_type}*.pth'))
        metrics_list = np.array([float(re.findall(f"{model_type}.*?{metric}(.*?)\_", checkpoint)[0]) for checkpoint in checkpoints])
        if metric == 'vloss':
            ckpt = checkpoints[int(np.where(metrics_list == metrics_list.min())[0][-1])]
            best_metric = int(metrics_list.min())
        else:
            ckpt = checkpoints[int(np.where(metrics_list == metrics_list.max())[0][-1])]
            best_metric = int(metrics_list.max())
    else:
        try:
            best_metric = float(re.findall(f"{model_type}.*?{metric}(.*?)\_", ckpt)[0])
        except:
            best_metric = None

    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt['model'])
    try:
        best_train_loss = ckpt['best_train_loss']
        patience_tloss = ckpt['patience_vloss']
        patience_ep = ckpt['patience_ep']
    except:
        best_train_loss = 100
        patience_tloss = 0
        patience_ep = 0

    if optimizer!=None:
        optimizer.load_state_dict(ckpt['optimizer'])        
    
    return model,optimizer,best_metric,best_train_loss,patience_tloss,patience_ep


def train(model,dataloader,device,loss_fn,optimizer,mini_iter = None,epoch=200):
    running_loss = []
    model.train()

    for batch in tqdm(range(len(dataloader) if mini_iter == None else mini_iter)):
        feature, label = next(dataloader)
        if isinstance(feature, list):
            feature = [f.to(device) for f in feature]
        else:
            feature= feature.to(device)

        label = label[:,1].to(device)
        out = model(feature)
        loss = loss_fn(out,label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.detach().item())

        if batch==mini_iter:
            return model,np.mean(running_loss)

    return model,np.mean(running_loss)


def eval(model,dataloader,loss_fn,device,n_aug):
    model.eval()
    with torch.no_grad():
        running_loss = []
        MOS = []
        labels = []
        inds = []

        for batch in tqdm(range(len(dataloader))):
            feature, label = next(dataloader)
            ind = label[:,0].type(torch.long)

            if isinstance(feature, list):
                feature = [f.to(device) for f in feature]
            else:
                feature = feature.to(device)
            label = label[:,1].to(device)

            out = model(feature)
            loss = loss_fn(out,label)
            running_loss.append(loss.detach().item())
            MOS += out.detach().cpu().numpy().tolist()
            labels += label.detach().cpu().numpy().tolist()
            inds += ind.tolist()
            
        labels = np.array(labels)
        labels[inds] = deepcopy(labels) 
        plcc, srcc, acc = metrics(MOS,labels,n_aug)
        return plcc, srcc, acc , np.mean(running_loss)

def lr_rescheduler(optimizer):
    print('lr: ', optimizer.param_groups[0]['lr'])
    if optimizer.param_groups[0]['lr']/10 > 1e-6:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
    else:
        optimizer.param_groups[0]['lr'] = 1e-6
    return optimizer


def save_model(model,optimizer,ckpt_folder,epoch,plcc,srcc,acc,val_loss,best_train_loss,patience_ep,patience_tloss,model_type):
    torch.save({
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'best_train_loss' : best_train_loss,
        'patience_ep' : patience_ep,
        'patience_tloss' : patience_tloss,
    }, os.path.join(ckpt_folder,f'{model_type}_ep{epoch}_vloss{val_loss:.2f}_plcc{plcc:.2f}_srcc{srcc:.2f}_acc{acc:.2f}_.pth'))
    

def logging(epoch,train_loss,val_loss,acc,plcc,srcc):
    print(f'Epoch {epoch}: train mse loss {train_loss}, val mse loss-{val_loss}, val acc-{acc}, srcc-{srcc}, plcc-{plcc}')


def makedirs(root='./experiment',base_model_type='inceptionv3',num_level=11,feature_type='narrow',head_type=None,resize=True,augment=False,hard_dp=False):
    feature_path = os.path.join('feature',base_model_type,feature_type,'resize' if resize else 'no_resize','aug' if augment else 'no_aug')
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    if head_type is not None:
        if base_model_type == 'inceptionv3':
            folder = os.path.join(root,base_model_type,str(num_level),feature_type,head_type)
        else:
            folder = os.path.join(root,base_model_type,feature_type,head_type)
        
        config = ''
        if resize:
            config+='resize'
        else:
            config+='no_resize'
        if augment:
            config+='+aug'
        if hard_dp:
            config+='+hard_dp'

        ckpt_path = os.path.join(folder,config)
        
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
    return ckpt_path,feature_path

def build_h5(h5_path,imgloader,channel_size,feature_type):
    if os.path.isfile(h5_path):
        h5_file = h5py.File(h5_path, 'a')
    else:
        h5_file = h5py.File(h5_path, 'w')
        h5_file.create_dataset("label", (len(imgloader.dataset),1), dtype = 'float16')

        if feature_type == 'narrow':     
            h5_file.create_dataset('feature',(len(imgloader.dataset),np.sum(channel_size)), dtype = 'float16')
        else:
            h5_file.create_dataset('feature',(len(imgloader.dataset),np.sum(channel_size),5,5), dtype = 'float16')

    return h5_file

def extract_features(imgloader, device, bmodel, feature_type, h5_file,batch_size):
    with torch.no_grad():
        for i,(img,label) in tqdm(enumerate(imgloader)):    
            label = label[:,1]
            img= img.to(device)
            MLSP = bmodel.get_MLSP(img,feature_type)
            h5_file['label'][i*len(img):(i+1)*len(img)] = label.numpy().astype(np.float16) 
            if i <= int(len(imgloader.dataset)//batch_size-1):   
                h5_file['feature'][i*len(img):(i+1)*len(img)] = MLSP.detach().cpu().numpy().astype(np.float16)
            else:
                h5_file['feature'][-len(img):] = MLSP.detach().cpu().numpy().astype(np.float16)