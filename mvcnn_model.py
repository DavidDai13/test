# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""
import torch   
import torch.nn as nn   
from torchvision import models

class MVCNN(nn.Module):
    """definition."""
    
    def __init__(self, num_classes,use_gpu=True):
        super(MVCNN, self).__init__()
        self._num_classes = num_classes
        self.use_gpu = use_gpu
        
        """Build pre-trained resnet34 model for feature extraction of 3d model render images
        """
        self.resnet34_model=models.resnet34(pretrained=True)
        for param in self.resnet34_model.parameters():
            param.requires_grad = True
     
        self.feature_size = self.resnet34_model.fc.in_features   
        print(self.feature_size)
        self.resnet34_model = nn.Sequential(*list(self.resnet34_model.children())[:-1])  #remove last layer

        self.fc = nn.Linear(self.feature_size,self._num_classes)
        #logits = fc(feature)  
    
    
    def forward(self, x):
        """
        Args:
            x: input a batch of image
            
        Returns:
            pooled_view: Extracted features, maxpooling of multiple features of 12 view_images of 3D model
                
            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        x = x.transpose(0, 1)
        
        view_pool = []
        
        for v in x:
            v = v.type(torch.cuda.FloatTensor)

            feature = self.resnet34_model(v)
            feature = feature.view(feature.size(0), 512)   #
            
            view_pool.append(feature)
        
        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.max(pooled_view, view_pool[i])  #max_pooling
            
        logits = self.fc(pooled_view)
        return pooled_view,logits


    

