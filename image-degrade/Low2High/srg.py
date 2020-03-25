#Generator of SRGAN

import numpy as numpy
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as functional
import torch.nn as nn

class BasicBlock(torch.nn.Module):
    
    def __init__(self, planes=64, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()

        self.conv = nn.Conv2d(planes, planes, kernel_size=3, stride=1)
        self.bn = nn.BatchNorm2d(planes)
    
    def forward(self, x):
        res = x
        out = self.conv(x)
        out = nn.PReLU(self.bn(out))
        out = self.conv(out)
        out = self.bn(out)

        out += res

        return out


class Generator(torch.nn.Module):

    #  B is the number of ResBlocks, by default B = 16
    def __init_(self, planes=[3,64,256,3], stride=1, B=16):
        self.bn = nn.BatchNorm2d(planes)
        #planes[0] is input plane
        self.conv1 = nn.Conv2d(planes[0], planes[1], kernel_size=9, stride=1)

        self.conv2 = nn.Conv2d(planes[1], planes[2], kernel_size=3, stride=1)

        self.conv3 = nn.Conv2d(planes[2], planes[3], kernel_size=3, stride=1) 
        self.nn_up1 = nn.PixelShuffle(2)
        self.nn_up2 = nn.PixelShuffle(2)

        self.conv4 = nn.Conv2d(planes[3], planes[3], kernel_size=9, stride=1) 
        
    
    def _layers(self, B=16):
        netLayers = []
      
        for i in range(B):
            netLayers.append(BasicBlock(64, 64, 3, 1))
        
        return torch.nn.Sequential(*netLayers)


    def forward(self, x):

        out = nn.PReLU(self.conv1(x))

        res = out
        out = self._layers(16)
        
        out = self.bn(self.conv2(out))

        out += res

        out = self.conv3(out)
        out = nn.PReLU(self.nn_up1(out))

        out = self.conv3(out)
        out = nn.PReLU(self.nn_up2(out))
        
        out = self.conv4(out)

        return out


mod = Generator([3,64,256,3])    
