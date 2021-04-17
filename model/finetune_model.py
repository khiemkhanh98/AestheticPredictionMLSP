import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from model.base import Base
from model.head import Head

class Fmodel(nn.Module):
    def __init__(self,base_model_type,feature_type, head):
        super(Fmodel,self).__init__()
        self.bmodel = Base(base_model_type)
        self.head = head
        self.feature_type = feature_type
        self.bmodel.unfreeze()
    
    def forward(self,img):
        #self.bmodel(img)
        x = self.bmodel.get_MLSP(img,self.feature_type,self.head.head_type)
        return self.head(x)
        
