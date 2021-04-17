import h5py
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import os


class ImgDataset(data.Dataset):
    """
    1. I don'think that you must perform the augmentation (when augment = True, finetune = false)-> save imgs to disk -> load it back in dataloader.
    I think that instead you only need to store the original imgs on disk and perform 'online'/'hot' augmentation, meaing that you
    store origin imgs -> load them back into dataloader -> perform augmentation. (This approach has adv that it saves your Drive storage, but takes longer time
    to process. If your Drive is large enough, then you should follow the first approach that store everything on disk to save loading time)

    2. Make sure the dataloader output the img in the following order (e.g. there are 3 imgs in dataset -A,B,C and 3 types of augmentation -1,2,3), then the order should be: A1,C3,B2,A2,C2,B1....
    This means that I only see an image once every epoch, and after 8 epochs I will see that image 8 times with 8 different augmentation (aug = True, ftune = False). Please ensure
    the image order in every epoch is the same (e.g. ACB) for test and val dataloader since in the eval step, you will have to average them, so if you fck up the order, then you will also fck up the
    evaluation step. Regarding the train dataloader, if possible, you should shuffle the order (this is rule of thumb of training DL model). But if it's too hard for you, then just make them all the same
    """
    def __init__(self, mode, resize, augment , finetune ):
        'mode specifies train/test/val dataloader type, augment means flip+crop'
        super(ImgDataset,self).__init__()
        
    def __len__(self):
        """
        Make sure that you return me 8*length of the dataset if augmentation is on
        but length of the dataset if finetuning is on (finetune use random aug so no 8 
        fixed position) for train set and 20*length_dataset for finetune test/val set
        """
        return 80
        
    def __getitem__(self, index):
        return (torch.rand(3,256,256), torch.rand(1))

    
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
            batch_size = 1024, head_type = None,h5_paths = None):
    splits = ['train', 'val', 'test']
    dataloaders = []

    if data_type == 'img':
        for split in splits:
            imgdataset = ImgDataset(mode = split, resize = resize, augment = augment, finetune = finetune)
            dataloaders.append(DataLoader(imgdataset, batch_size=batch_size))
        
    else:
        for h5_path in h5_paths:
            feadataset = FeatureDataset(h5_path,head_type)
            dataloaders.append(DataLoader(feadataset, batch_size=batch_size,shuffle=False, num_workers=2,pin_memory = True))

    return dataloaders