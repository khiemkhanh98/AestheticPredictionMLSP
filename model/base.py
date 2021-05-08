import pretrainedmodels
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Base(nn.Module):
    def freeze(self):
        for param in self.base_model.parameters():
                param.requires_grad = False
    
    def unfreeze(self):
        for param in self.base_model.parameters():
                param.requires_grad = True

    def attach_fea_out(self,classname,input,output):
        self.features.append(output)

    def attach_fea_in(self,classname,input,output):
        self.features.append(input[0])

    def __init__(self,model,level,trainable = False):
        super(Base,self).__init__()
        self.features = []
        self.channel_size = []
        os.environ['TORCH_HOME'] = './pretrained_model'

        if model == 'inceptionresnetv2':
            self.base_model = pretrainedmodels.__dict__[model](num_classes=1000, pretrained='imagenet')
            used_blocks = ['mixed_5b', 'repeat','repeat_1','repeat_2','mixed_6a','mixed_7a']
            unused_blocks = ['conv2d_7b','avgpool_1a','last_linear']

            for block in used_blocks:
                if 'mixed' in block:
                    getattr(self.base_model,block).register_forward_hook(self.attach_fea_out)
                else:           
                    sub_blocks = [submod for submod in getattr(base_model,block).children()]
                    for sub_block in sub_blocks:
                        sub_block.conv2d.register_forward_hook(attach_fea_in)
            self.base_model.block8.conv2d.register_forward_hook(attach_fea_in)
        
        if model == 'inceptionv3':
            self.base_model = models.inception_v3(pretrained=True,aux_logits=False)
            used_blocks = ['Mixed_5b', 'Mixed_5c','Mixed_5d','Mixed_6a','Mixed_6b','Mixed_6c','Mixed_6d','Mixed_6e','Mixed_7a','Mixed_7b','Mixed_7c'][-level:]
            unused_blocks = ['avgpool','fc']

            for block in used_blocks:
                getattr(self.base_model,block).register_forward_hook(self.attach_fea_out)

            for block in unused_blocks:
                setattr(self.base_model,block,nn.Identity())
        
        if not trainable:
            self.freeze()

        fake_img = torch.rand(1,3,256,256) ## pass fake img to the model to get the channel size
        self.base_model(fake_img)
        self.channel_size = [block.size()[1] for block in self.features]
        self.features = []

    def forward(self,img):
        self.base_model(img)

    def get_MLSP(self,img,feature_type, head_type):
        self.base_model(img)
        if feature_type == 'narrow':
            MLSP = [F.adaptive_avg_pool2d(block, (1, 1)) for block in self.features]
            for i in range(len(MLSP)):
                MLSP[i] = MLSP[i].squeeze(2).squeeze(2)

        if feature_type == 'wide':
            MLSP = [F.interpolate(block,mode = 'area', size = 5) for block in self.features]
        
        self.features = []
        if head_type == 'multi_3FC':
            return MLSP

        MLSP = torch.cat(MLSP,dim = 1)
        return MLSP
    
