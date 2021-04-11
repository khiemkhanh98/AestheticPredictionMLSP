import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
from base_model import get_model,get_MLSP

class finetune_model(nn.Module):
    def __init__(self,base_model,head,feature_type, concat_features):
        self.base_model = base_model
        self.head = head
        self.feature_type
        self.concat_features = concat_features
    
    def forward(self,img):
        self.base_model(img)
        x = get_MLSP(self.feature_type, self.concat_features)
        return self.head
