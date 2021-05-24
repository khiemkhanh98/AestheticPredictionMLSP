from model.base import Base
from model.head import Head
import torch
import h5py
import numpy as np
import os
from utils import build_h5,extract_features,makedirs
from torch.optim import Adam
import gc
from dataloader import get_dataloader
from torchsummary import summary

base_model_type = 'inceptionv3'
feature_type = 'narrow'
resize = True
augment = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_,fea_folder = makedirs(base_model_type=base_model_type,feature_type=feature_type,resize=resize,augment=augment)
h5_paths = [os.path.join(os.path.split(fea_folder)[0] if split!='train' else fea_folder, split + '_fea.h5') for split in ['train', 'val', 'test']]
if resize == True:
    extract_batch_size = 128
else:
    extract_batch_size = 1
imgloaders = get_dataloader(data_type='img',resize=resize,augment=augment,batch_size = extract_batch_size,extract=True)

bmodel = Base(model = base_model_type, level = 11)
bmodel.to(device)
bmodel.eval()
# print(summary(bmodel,(3,256,256,)))

for h5_path,imgloader in zip(h5_paths,imgloaders):
    with build_h5(h5_path,imgloader,bmodel.channel_size,feature_type) as h5_file:
        extract_features(imgloader, device, bmodel, feature_type, h5_file,extract_batch_size)

print('done')