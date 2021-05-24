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
import matplotlib.pyplot as plt
from PIL import Image
import os
from model.finetune_model import Fmodel
import pandas as pd

hard_dp = False
if hard_dp:
    dropout = [0.5,0.5,0.75]
else:
    dropout = [0.25,0.25,0.5]

base_model_type = 'inceptionv3'
num_level = 11
feature_type = 'narrow'
head_type = 'single_1FC'
if feature_type == 'wide':
    head_type = 'pool_3FC'
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bmodel = Base(model = base_model_type, level = num_level)

head = Head(head_type,bmodel.channel_size,dropout)
head,_,_,_,_,_ = get_model_from_ckpt(model = head,
    ckpt = 'experiment/inceptionv3/11/narrow/single_1FC/resize/head_ep26_vloss0.34_plcc0.60_srcc0.57_acc0.78_.pth')
fmodel = Fmodel(base_model_type,feature_type, head, num_level, bmodel)
fmodel.to(device)
fmodel.eval()

df = pd.read_csv('./data/label.csv').set_index('img')
img = Image.open('/content/images/179045.jpg')
label = df.loc[179045]

plt.imshow(img)
plt.show()

if len(img.size)==2:
    img = img.convert('RGB')
img = T.ToTensor()(img)
img = normalize(img,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
pred = fmodel(img)
print('pred: ', pred[0], ' , label: ', label[0])