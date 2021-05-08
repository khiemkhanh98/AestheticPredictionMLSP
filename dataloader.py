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
    def __init__(self, path, mode, resize, augment , finetune ):
        super(ImgDataset,self).__init__()
        self.mode = mode  ##'train'/'test'/'split'
        self.resize = resize
        self.augment = augment
        self.finetune = finetune
        self.path = path
        self.img_files = os.listdir(os.path.join(path,mode))  
        self.label = pd.read_csv(path + mode + '_labels.txt', header=None, delimiter = r'\s+').set_index(0)

        if not finetune:
            if augment:
                self.augment_list = np.array(list(range(8))*len(self.img_files)).reshape(-1,8)
                self.augment_list = shuffle_along_axis(self.augment_list,axis=1).flatten('F')
                self.img_files = self.img_files*8
        else:
            if mode != 'train':
                self.img_files = self.img_files*20

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        file_path = self.img_files[index]
        img = T.ToTensor()(Image.open(os.path.join(self.path,self.mode,file_path)))
        img = normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = self.label.loc[int(file_path[:-4])]

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
    def __init__(self, h5_path, head_type):
        super().__init__()
        self.data = h5py.File(h5_path, 'r')
        self.head_type = head_type
        if not head_type == 'multi_3FC':
             self.features = self.data['feature']
             self.channel_size = self.features.shape[1] 
        else:
            self.num_level = len(self.data['feature'].keys())
            self.features = [self.data[f'feature/{level}'] for level in range(self.num_level)]
            self.channel_size = [feature.shape[1] for feature in self.features]

        self.labels = self.data['label']
        self.size = len(self.labels)

    def __len__(self):
        return int(self.size/8)

    def __getitem__(self, index):
        # get data
        if not self.head_type == 'multi_3FC':
            x = self.features[index][()].astype(np.float32)
            x = torch.from_numpy(x)
        else:
            x = []
            for level in range(self.num_level):
                x.append(torch.from_numpy(self.features[level][index][()].astype(np.float32)))

        # get label
        y = self.labels[index][()].astype(np.float32)
        y = torch.from_numpy(y)

        return (x, y)


def get_dataloader(data_type, resize = False, augment= False, finetune = False, 
            batch_size = 64, head_type = None,h5_paths = None,img_paths = './data/'):
    splits = ['train', 'val', 'test']
    dataloaders = []

    if data_type == 'img':
        for split in splits:
            imgdataset = ImgDataset(path = img_paths, mode = split, resize = resize, augment = augment, finetune = finetune)
            if split == 'train':
                shuffle = True
            else:
                shuffle = False
            dataloaders.append(DataLoader(imgdataset, batch_size=batch_size,shuffle = shuffle, num_workers=2,pin_memory = True))
        
    else:
        for h5_path in h5_paths:
            feadataset = FeatureDataset(h5_path,head_type)
            if 'train' in h5_path:
                shuffle = True
            dataloaders.append(DataLoader(feadataset, batch_size=batch_size,shuffle=shuffle, num_workers=2,pin_memory = True))

    return dataloaders