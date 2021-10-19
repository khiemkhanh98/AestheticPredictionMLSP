import torch
import torch.nn as nn
from torch.nn import Conv2d,AvgPool2d,Linear,Sequential,Dropout,BatchNorm2d,ModuleList,BatchNorm1d
import torch.nn.functional as F
import numpy as np
import torchvision

class View(nn.Module):
    def __init__(self, shape):
        super(View,self).__init__()
        self.shape = shape

    def forward(self, x):
        x  =x.view(*self.shape)
        return x

class head_block():
    def conv_block(self,inc,outc,ker,padding = 1,avgpool = False):
        modules = []
        if avgpool:
            modules.append(AvgPool2d(3,1,1))
        modules.append(Conv2d(inc,outc,ker,padding = padding))
        modules.append(nn.BatchNorm2d(outc))
        modules.append(nn.ReLU())
        return Sequential(*modules)

    def block_3FC(self,num_channel, dropout, x=2048):
        modules = []
        modules.append(Linear(num_channel,x))
        modules.append(nn.ReLU())
        modules.append(BatchNorm1d(int(x)))
        
        modules.append(Dropout(p=dropout[0]))
        modules.append(View([-1,int(x)]))
        modules.append(Linear(x,int(x/2)))
        
        modules.append(nn.ReLU())
        modules.append(BatchNorm1d(int(x/2)))
        modules.append(Dropout(p=dropout[1]))

        modules.append(View([-1,int(x/2)]))
        modules.append(Linear(int(x/2),int(x/8)))
        modules.append(nn.ReLU())
        modules.append(Dropout(p=dropout[2]))

        return Sequential(*modules)

    def single_1FC(self,num_channel,dropout):
        num_channel = int(np.sum(num_channel))
        return Linear(num_channel,1)

    def single_3FC(self,num_channel,dropout, x=2048):
        num_channel = int(np.sum(num_channel))
        return Sequential(self.block_3FC(num_channel,dropout,x),Linear(int(x/8),1))

    def multi_3FC(self,num_channels,dropout):
        blocks = []
        for num_channel in num_channels:
            blocks.append(self.block_3FC(num_channel,dropout,num_channel))
        blocks.append(Linear(int((np.sum(num_channels)/8)),1))
        return ModuleList(blocks)

    def pool_3FC(self,num_channel, dropout):
        num_channel = int(np.sum(num_channel))
        blocks = []
        blocks.append(self.conv_block(num_channel,1024,1,0))
        blocks.append(self.conv_block(num_channel,1024,3,1))
        blocks.append(self.conv_block(num_channel,1024,1,0,True))
        blocks.append(self.block_3FC(3072,dropout, 3072))
        blocks.append(Linear(int(3072/8),1))
        return ModuleList(blocks)

class Head(nn.Module):
    def __init__(self,head_type,num_channel,hard_dp,num_level = 11):
        super(Head, self).__init__()
        
        if hard_dp:
            dropout = [0.5,0.5,0.75]
        else:
            dropout = [0.25,0.25,0.5]

        if ('single' in head_type) and (num_level!=11):
            num_chanel = num_channel[-num_level:]

        self.head = getattr(head_block(),head_type)(num_channel, dropout)
        self.head_type = head_type
        self.num_ch = num_channel

    def forward(self,features):
        if self.head_type == 'multi_3FC':
            features = torch.split(features,self.num_ch,dim=1)
            x = torch.cat([block(feature) for feature,block in zip(features,self.head[:-1])],dim=1)
            x = self.head[-1](x)
        elif self.head_type == 'pool_3FC':
            x = torch.cat([block(features) for block in self.head[:-2]],dim=1)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(-1,3072)
            x = self.head[-2](x)
            x = self.head[-1](x)
        else:
            features = features[-np.sum(self.num_ch):]
            x = self.head(features)
        return x

