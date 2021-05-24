import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
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
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class ImgDataset(data.Dataset):
    def __init__(self, path, mode, resize, augment , finetune):
        super(ImgDataset,self).__init__()
        self.mode = mode  ##'train'/'test'/'split'
        self.resize = resize
        self.augment = augment
        self.finetune = finetune
        self.path = path
        df = pd.read_csv('./data/AVA_data_official_test.csv')
        self.label = df.loc[df.set == mode][['image_name','MOS']].set_index('image_name')
        self.img_files = self.label.index.values.tolist()

        if mode!='train' or augment:
            if not finetune:
                self.augment_list = np.array(list(range(8))*len(self.img_files)).reshape(-1,8)
                self.augment_list = shuffle_along_axis(self.augment_list,axis=1).flatten('F')
                self.img_files = self.img_files*8
            else:
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
    
        return img, torch.tensor(label.values)

    
class FeatureDataset(data.Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.data = h5py.File(h5_path, 'r')

        self.features = self.data['feature']
        self.channel_size = self.features.shape[1] 

        self.labels = self.data['label']
        self.size = len(self.labels)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = self.features[index][()].astype(np.float32)
        x = torch.from_numpy(x)
 
        y = self.labels[index][()].astype(np.float32)
        y = torch.from_numpy(y)
        return (x, y)


def get_dataloader(data_type, resize = False, augment= False, finetune = False, batch_size = 64, h5_paths = None,img_paths = './data/',extract=False):
    dataloaders = []
    pin_memory = True
    splits = ['train', 'val', 'test']
    
    for i in range(3):
        if i == 0:
            num_workers = 2
            if not extract:
                shuffle = False
            else:
                shuffle = False
        else:
            num_workers = 1
            shuffle = False

        if i!=0 or extract:  
            drop_last = False      
        else:
            drop_last = True

        if data_type == 'img':        
            dataset = ImgDataset(path = img_paths, mode = splits[i], resize = resize, augment = augment, finetune = finetune)
        else:
            dataset = FeatureDataset(h5_paths[i])

        dataloaders.append(DataLoader(dataset, batch_size=batch_size,num_workers=num_workers,drop_last=drop_last,worker_init_fn=seed_worker))
    return dataloaders