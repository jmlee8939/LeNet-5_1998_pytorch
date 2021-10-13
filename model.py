import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
import math
from PIL import Image
import cv2

# costumized Tanh for LeNet-5 1998
class Tanh(nn.Module):
    def forward(self, x):
        return 1.7159*torch.tanh(x*2/3)

# Layer C1
# - convolutional layer with 6 feature maps.
# - 5 X 5 kernel
class Layer_C1(nn.Module):
    def __init__(self):
        super(Layer_C1, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.reset_parameters()
    
    def forward(self, x):
        x = self.c1(x)
        return x
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.c1.weight, a=math.sqrt(2))
        self.c1.bias.data.fill_(0.01)
    
# Layer S2
# - subsampling layer (=Average pool) 
# - kernel size 2 X 2
# - map_size 14 X 14
class Layer_S2(nn.Module):
    def __init__(self):
        super(Layer_S2, self).__init__()
        # kernel_size = 2, in_channels = 6
        self.kernel_size = 2
        self.weight = Parameter(torch.Tensor(1,6,1,1))
        self.bias = Parameter(torch.Tensor(1,6,1,1))
        self.reset_parameters()
    
    def forward(self, x):
        x = F.avg_pool2d(x, self.kernel_size)
        x = x*self.weight + self.bias
        return x
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        self.bias.data.fill_(0.01)

# Layer C3
# - convolutional layer
# - 16 feature maps
# - each S2 feature map is connected to 10 neighborhoods subset of C3 feature map 
class Layer_C3(nn.Module):
    def __init__(self):
        super(Layer_C3, self).__init__()
        # in_channels = 6, out_channels = 16, kernel_size = 5
        self.weight = Parameter(torch.Tensor(10, 6, 5, 5))
        self.bias = Parameter(torch.Tensor(1, 16, 1, 1))
        self.kernel_size = 5
        self.out_channels = 16
        self.reset_parameters()
        
    def map_combine_list(self):
        connection_list = [[0,4,5,6,9,10,11,12,14,15],
                [0,1,5,6,7,10,11,12,13,15],
                [0,1,2,6,7,8,11,13,14,15],
                [1,2,3,6,7,8,9,12,14,15],
                [2,3,4,7,8,9,10,12,13,15],
                [3,4,5,8,9,10,11,13,14,15]]
        return connection_list
    
    def forward(self,x):
        B_size = x.size(0)
        output_size = x.size(3)-self.kernel_size+1
        output = torch.zeros(B_size,self.out_channels,output_size,output_size)
        list_ = self.map_combine_list()
        for i in range(len(list_)):
            output[:,list_[i],:,:] += (F.conv2d(x[:,i,:,:].unsqueeze(1), 
                                    self.weight[:,i,:,:].unsqueeze(1)) + self.bias[:,list_[i],:,:])
        return output
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(2))
        self.bias.data.fill_(0.01)

# Layer S4
# - subsampling layer (=Average pool) 
# - kernel size 2 X 2
# - map_size 14 X 14
class Layer_S4(nn.Module):
    def __init__(self):
        super(Layer_S4, self).__init__()
        # kernel_size = 2, in_channels = 16
        self.kernel_size = 2
        self.weight = Parameter(torch.Tensor(1,16,1,1))
        self.bias = Parameter(torch.Tensor(1,16,1,1))
        self.reset_parameters()
    
    def forward(self, x):
        x = F.avg_pool2d(x, self.kernel_size)
        x = x*self.weight + self.bias
        return x
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        self.bias.data.fill_(0.01)        
        
        
# Layer C5 
# - convolutional layer with 6 feature maps.
# - 5 X 5 kernel
class Layer_C5(nn.Module):
    def __init__(self):
        super(Layer_C5, self).__init__()
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.reset_parameters()
    
    def forward(self, x):
        x = self.c5(x)
        return x
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.c5.weight, a=math.sqrt(2))
        self.c5.bias.data.fill_(0.01)      

        
# RBF kernel function 
# - output layer is composed of Euclidean radial basis function units.
# - kernel is in './RBF_kerenl'
class RBF(nn.Module):
    def __init__(self):
        # in_features = 84, out_features = 10, tensor kernel
        super(RBF, self).__init__()
        self.in_features = 84
        self.out_features = 10
        self.kernel = self.rbf_tensor()

    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        c = self.kernel.unsqueeze(0).expand(size)
        output = (x - c).pow(2).sum(-1)
        return output
    
    # making kernel image to tensor    
    def rbf_tensor(self):
        kernel_list = []
        for i in range(10):
            file = './RBF_kernel/' + str(i) + '_RBF.jpg'
            image = cv2.imread(file, 0)
            image = cv2.threshold(image,127,1,cv2.THRESH_BINARY)[1]*-1+1
            kernel_list.append(image.flatten())
        return(torch.Tensor(kernel_list))

# LeNet5
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            # layer C1 
            # 6 feature maps, kernel 5x5  
            Layer_C1(),
            Tanh(),
            # layer S2
            # sub-sampling layer (=Average pool) 2x2, map_size=14
            Layer_S2(),
            Tanh(),
            # layer C3
            # 16 feature maps, kernel 5x5
            Layer_C3(),
            Tanh(),
            # layer S4
            # sub-sampling layer (=Average pool) 2X2
            Layer_S4(),
            Tanh(),
            # layer C5
            # 120 feature maps, kernel 5x5
            Layer_C5(),
            Tanh()
        )

        self.classifier = nn.Sequential(
            # layer F6
            nn.Linear(in_features=120, out_features=84),
            Tanh(),
            RBF(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

