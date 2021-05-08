import numpy as np
import h5py
import numpy as np
import os
import glob
import re
import torch

def metrics(x,y):
    """
    this function evaluate plcc, srcc, acc.
    make sure you read the source code to understand how it works
    and remember that the result is average accross the augmentation. E.g. the output is like following A1,B2,C1,A2,B1,C2 (A,B,C are images; 1,2 are augmentation type), the MOS 
    score for A = mean(A1+A2), B = ...., and only after that then the order of the MOS can be sorted, from which the plcc/srcc can be calculated. (That's what I understand from
    the source code, so i think it's better you recheck it). 
 
    """
    plcc = np.random.rand(1)
    srcc = np.random.rand(1)
    acc = np.random.rand(1)
    return plcc, srcc, acc

def makedirs(root,base_model_type,num_level,feature_type,head_type,resize,augment,hard_dp):
    if base_model_type == 'inceptionv3':
        folder = os.path.join(root,base_model_type,str(num_level),feature_type,head_type)
    else:
        folder = os.path.join(root,base_model_type,feature_type,head_type)
    config = ''

    if resize:
        config+='resize'
    if augment:
        config+='+augment'
    if hard_dp:
        config+='+hard_dp'

    path = os.path.join(folder,config)
    if not os.path.exists(path):
        os.makedirs(path)
    
    return path

def build_h5(folder,h5_path,imgloader,channel_size,feature_type,head_type,num_level):
    if os.path.isfile(h5_path):
        h5_file = h5py.File(h5_path, 'a')
    else:
        h5_file = h5py.File(h5_path, 'w')
        h5_file.create_dataset('label/', (len(imgloader.dataset),1), dtype = 'float16')
        
        if feature_type == 'narrow':      
            if not head_type == 'multi_3FC':
                h5_file.create_dataset('feature/', (len(imgloader.dataset),np.sum(channel_size)), dtype = 'float16')
            else:
                for level in range(num_level):
                    h5_file.create_dataset(f'feature/{level}', (len(imgloader.dataset),channel_size[level]), dtype = 'float16')
        else:
            h5_file.create_dataset('feature/', (len(imgloader.dataset),np.sum(channel_size),5,5), dtype = 'float16')
    return h5_file

def extract_features(imgloader, device, bmodel, feature_type, h5_file,batch_size,head_type,num_level):
    for i,(img,label) in enumerate(imgloader):
        img= img.to(device)
        MLSP = bmodel.get_MLSP(img,feature_type, head_type)
        h5_file['label'][i*batch_size:(i+1)*batch_size] = label.numpy().astype(np.float16)    
        if not head_type == 'multi_3FC':
            h5_file['feature'][i*batch_size:(i+1)*batch_size] = MLSP.detach().cpu().numpy().astype(np.float16)
        else:
            for level in range(num_level):
                h5_file[f'feature/{level}'][i*batch_size:(i+1)*batch_size] = MLSP[level].detach().cpu().numpy().astype(np.float16)
    h5_file.close()


def get_model_from_ckpt(folder,model, model_type,metric, optimizer = None):
    checkpoints = glob.glob(os.path.join(folder,f'{model_type}*.pth'))
    metrics_list = np.array([float(re.findall(f"{model_type}.*?{metric}(.*?)\_", checkpoint)[0]) for checkpoint in checkpoints])
    ckpt = checkpoints[int(np.where(metrics_list == metrics_list.max())[0])]
    ckpt = torch.load(ckpt)
    model.load_state_dict(ckpt['model'])
    best_metric = int(metrics_list.max())
    
    if optimizer!=None:
        optimizer.load_state_dict(ckpt['optimizer'])        
    
    return model,optimizer,best_metric

def train(model,dataloader,device,loss_fn,optimizer,mini_iter,epoch):
    running_loss = 0
    model.train()
    for batch, (feature, label) in enumerate(dataloader):
        if isinstance(feature, list):
            feature = [f.to(device) for f in feature]
        else:
            feature= feature.to(device)
        label = label.to(device)

        out = model(feature)
        loss = loss_fn(out,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item()
        if batch % mini_iter == 0:
            print(f'Epoch {epoch} batch{batch} train_loss: {running_loss/mini_iter}')
            running_loss = 0

        return model


def eval(model,dataloader,loss_fn,device):
    with torch.no_grad():
        running_loss = 0
        MOS = []
        labels = []
        model.eval()
        for batch, (feature, label) in enumerate(dataloader):
            if isinstance(feature, list):
                feature = [f.to(device) for f in feature]
            else:
                feature = feature.to(device)
            label = label.to(device)

            out = model(feature)
            loss = loss_fn(out,label)
            running_loss += loss.detach().item()
            MOS.append(out.detach().cpu().numpy())
            labels.append(out.detach().cpu().numpy())

    plcc, srcc, acc = metrics(MOS,labels)
    return plcc, srcc, acc , running_loss/(batch+1)

def lr_scheduler(optimizer, epoch, lr, patience_cnt):
    if ((epoch%20)== 0) or patience_cnt:
        lr = lr/10
    optimizer.param_groups[0]['lr'] = lr
    return optimizer


def save_model(model,optimizer,ckpt_folder,epoch,plcc,srcc,acc,model_type):
    torch.save({
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }, os.path.join(ckpt_folder,f'{model_type}_ckpt_ep{epoch}_plcc{plcc[0]:.2f}_srcc{srcc[0]:.2f}_acc{acc[0]:.2f}_.pth'))

def logging(epoch,val_loss,acc,plcc,srcc):
    print(f'Epoch {epoch}: val loss-{val_loss} val acc-{acc}, srcc-{srcc}, plcc-{plcc}')