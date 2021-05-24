import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from model.base import Base
from model.head import Head

class Fmodel(nn.Module):
    def __init__(self,base_model_type,feature_type, head, num_level, bmodel = None):
        super(Fmodel,self).__init__()
        if bmodel == None:
            self.bmodel = Base(base_model_type,num_level)
        else:
            self.bmodel = bmodel
        self.head = head
        self.feature_type = feature_type
    
    def forward(self,img):
        #self.bmodel(img)
        x = self.bmodel.get_MLSP(img,self.feature_type,self.head.head_type)
        x = self.head(x)
        return x

    def unfreeze(self):
        self.bmodel.model.unfreeze()

    def eval(self):
        self.bmodel.eval()
        self.head.eval()
    
    def train(self):
        #self.bmodel.train()
        self.head.train()
        
