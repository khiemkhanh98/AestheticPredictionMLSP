import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader,SubsetRandomSampler
import os
from torchvision.transforms.functional import five_crop,crop,resize,hflip,normalize
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import zipfile
import io
import time
import matplotlib.pyplot as plt
import random
from PIL import ImageFile
import warnings
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore")

#https://stackoverflow.com/questions/5040797/shuffling-numpy-array-along-a-given-axis
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def fixed_crop_flip(img,option):
    crop_size = (int(0.875*img.shape[1]), int(0.875*img.shape[2]))
    img = five_crop(img,crop_size)[option%4]
    if option>3:
        img = hflip(img)
    return img

def random_crop_flip(img,ind = None):
    crop_size = (int(0.875*img.shape[1]), int(0.875*img.shape[2]))
    if ind != None:
        r = np.random.RandomState(ind)
        crop_coor = (r.randint(0,img.shape[1]-crop_size[0]), r.randint(0,img.shape[2]-crop_size[1]))
        flip_prob = r.rand()
    else:
        crop_coor = (np.random.randint(0,img.shape[1]-crop_size[0]), np.random.randint(0,img.shape[2]-crop_size[1]))
        flip_prob = np.random.rand()

    img = crop(img,*crop_coor,*crop_size)
    if flip_prob>0.5:
        img = hflip(img)
    
    return img

class ImgDataset(data.Dataset):
    def __init__(self, path, mode, resize, augment , finetune, fraction = 1): 
        super(ImgDataset,self).__init__()
        self.mode = mode  ##'train'/'test'/'split'
        self.resize = resize
        self.augment = augment
        self.finetune = finetune
        self.path = path
        df = pd.read_csv('./data/AVA_data_official_test.csv')
        self.label = df.loc[df.set == mode][['image_name','MOS']].set_index('image_name')
        self.img_files = self.label.index.values.tolist()
        self.img_files = self.img_files[:int(fraction*len(self.img_files))]

        if mode!='train' or augment:
            if not finetune:
                self.augment_list = np.array(list(range(8))*len(self.img_files)).reshape(-1,8)
                self.augment_list = shuffle_along_axis(self.augment_list,axis=1).flatten('F')
                self.img_files = self.img_files*8
            else:
                if mode!='train':
                    self.img_files = self.img_files*20
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        file_path = self.img_files[index]        
        img = Image.open('/content/images/' + file_path)

        if len(img.size)==2:
            img = img.convert('RGB')
        img = T.ToTensor()(img)
        img = normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = self.label.loc[file_path]
        if self.resize:
            img = resize(img,(256,256))
        
        if not self.finetune:
            if self.augment:      
                img = fixed_crop_flip(img,self.augment_list[index])
        else:
            if self.augment:
                if self.mode!='train':
                    img = random_crop_flip(img,index)
                else:
                    img = random_crop_flip(img)
    
        return img, torch.cat((torch.tensor([index]),torch.tensor(label.values)),0)

    
class FeatureDataset(data.Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.data = h5py.File(h5_path, 'r')
        self.h5_path = h5_path
        self.features = self.data['feature']
        self.labels = self.data['label']
        self.size = len(self.labels)

        # self.size = len(self.data['label'])
        # self.data.close()
        # del self.data

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # if not hasattr(self, 'data'):
        #     self.data = h5py.File(self.h5_path, 'r')
        #     self.features = self.data['feature']
        #     self.labels = self.data['label']

        x = self.features[index][()].astype(np.float32)
        x = torch.from_numpy(x)
 
        y = self.labels[index][()].astype(np.float32)
        y = torch.from_numpy(y)
        
        return x, torch.cat((torch.tensor([index]),y),0)


def get_dataloader(data_type, resize = False, augment= False, finetune = False, batch_size = 64, h5_paths = None,img_paths = './data/',extract=False,fraction = 1):
    dataloaders = []
    pin_memory = False
    splits = ['train', 'val', 'test']
    num_workers = 2
    shuffle = False
    for i,mode in enumerate(splits):
        if mode!='train' or extract:  
            drop_last = False 
            batch_size = batch_size*4
            if extract:
                num_workers = 1   ## to ensure img order and because h5 file does not work well when writing to non-adjacent position     
        else:
            drop_last = True
            if not extract and data_type=='img':
                shuffle=True

        if data_type == 'img':        
            dataset = ImgDataset(path = img_paths, mode = splits[i], resize = resize, augment = augment, finetune = finetune, fraction = fraction)
            
        else:
            dataset = FeatureDataset(h5_paths[i])

        dataloaders.append(DataLoader(dataset, batch_size=batch_size,num_workers=num_workers,drop_last=drop_last,shuffle=shuffle))
    return dataloaders