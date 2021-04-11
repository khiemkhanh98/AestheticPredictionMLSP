import pretrainedmodels
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

features = []

def attach_fea_out(self,input,output):
    global features
    features.append(output)

def attach_fea_in(self,input,output):
    global features
    features.append(input[0])

def get_model(model,trainable = False):
    global features 
    os.environ['TORCH_HOME'] = '/gdrive/MyDrive/SRP/Code/Baseline/AestheticPredictionMLSP/pretrained_model'

    if model == 'inceptionresnetv2':
        base_model = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
        used_blocks = ['mixed_5b', 'repeat','repeat_1','repeat_2','mixed_6a','mixed_7a']
        unused_blocks = ['conv2d_7b','avgpool_1a','last_linear']

        for block in used_blocks:
            if 'mixed' in block:
                getattr(base_model,block).register_forward_hook(attach_fea_out)
            else:           
                sub_blocks = [submod for submod in getattr(base_model,block).children()]
                for sub_block in sub_blocks:
                    sub_block.conv2d.register_forward_hook(attach_fea_in)
        base_model.block8.conv2d.register_forward_hook(attach_fea_in)
    
    if model == 'inceptionv3':
        base_model = models.inception_v3(pretrained=True,aux_logits=False)
        used_blocks = ['Mixed_5b', 'Mixed_5c','Mixed_5d','Mixed_6a','Mixed_6b','Mixed_6c','Mixed_6d','Mixed_6e','Mixed_7a','Mixed_7b','Mixed_7c']
        unused_blocks = ['avgpool','fc']

        for block in used_blocks:
            getattr(base_model,block).register_forward_hook(attach_fea_out)

    for block in unused_blocks:
        setattr(base_model,block,nn.Identity())
    
    if ~trainable:
        for param in base_model.parameters():
            param.requires_grad = False

    fake_img = torch.rand(1,3,256,256)
    base_model(fake_img)
    channel_size = [block.size()[1] for block in features]
    features = []

    return base_model,channel_size

def get_MLSP(feature_type, concat_features = True):
    global features
    if feature_type == 'narrow':
        MLSP = [F.adaptive_avg_pool2d(block, (1, 1)) for block in features]
        for i in range(len(MLSP)):
            MLSP[i] = MLSP[i].squeeze(2).squeeze(2)

    if feature_type == 'wide':
        MLSP = [F.interpolate(img,mode = 'area', size = 5) for block in features]
    
    features = []
    if not concat_features:
        return MLSP
        
    MLSP = torch.cat(MLSP,dim = 1)
    return MLSP
    
