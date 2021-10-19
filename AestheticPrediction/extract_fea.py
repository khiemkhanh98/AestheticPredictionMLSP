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
import argparse
torch.cuda.set_per_process_memory_fraction(0.3, 0)

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_type", help = 'inceptionv3 or inceptionresnetv2', type = str, default = 'inceptionv3')
parser.add_argument("--feature_type", help = 'narrow or wide', type = str, default = 'narrow')
parser.add_argument("--resize", action ='store_true')
parser.add_argument("--augment", action ='store_true')
parser.add_argument("--fraction", type = float, help = 'fraction of data', default = 1)

args = parser.parse_args()
base_model_type = args.base_model_type
feature_type = args.feature_type
resize = args.resize
augment = args.augment
fraction = args.fraction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_,fea_folder = makedirs(base_model_type=base_model_type,feature_type=feature_type,resize=resize,augment=augment)
h5_paths = [os.path.join(os.path.split(fea_folder)[0] if split!='train' else fea_folder, split + '_fea.h5') for split in ['train', 'val', 'test']]
if resize == True:
    extract_batch_size = 128
else:
    extract_batch_size = 1
imgloaders = get_dataloader(data_type='img',resize=resize,augment=augment,batch_size = extract_batch_size,extract=True,fraction=fraction)

bmodel = Base(model = base_model_type)
bmodel.to(device)
bmodel.eval()
# print(summary(bmodel,(3,256,256,)))

for h5_path,imgloader in zip(h5_paths[0:1],imgloaders[0:1]):
    with build_h5(h5_path,imgloader,bmodel.channel_size,feature_type) as h5_file:
        extract_features(imgloader, device, bmodel, feature_type, h5_file,extract_batch_size)
        print(h5_path)
        h5_file.close()

print('done')

# base_model_type = 'inceptionv3'
# feature_type = 'wide'
# resize = False
# augment = False