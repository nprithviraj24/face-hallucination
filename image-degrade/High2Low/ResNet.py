import torch
import numpy as numpy
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as functional


#debugger
import pdb


class BasicBlock(torch.nn.Module):
    expansion=1

    ## constructor
    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(BasicBlock, self).__init__()

        #declare conv layers with batch norms
        self.conv1 = torch.nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, stide=1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self,x):

        #Save the residue
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn1(self.conv1(x))

        if self.dim_change is not None:
            res = self.dim_change(res)
        
        output += res
        output = F.relu(output)

        return output
    

class Bottleneck(torch.nn.Module):

    expansion = 4

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(BasicBlock, self).__init__()

        #declare conv layers with batch norms
        self.conv1 = torch.nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, stide=1, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, planes*self.expansion, kernel_size=1)
        self.bn3 = torch.nn.BatchNorm2d(planes*self.expansion)
        self.dim_change = dim_change

    def forward(self, x):
        res = x 
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn1(self.conv2(x)))
        out = self.bn3(self.conv3(output))

        if self.dim_change is not None:
            res = self.dim_change(res)
        
        out += res
        out = F.relu(out)

        return out



#Now lets build the actual network
class ResNet(torch.nn.Module):

    def __init__(self, block, num_layers, classes=10 ):
        self.input_plane = 64
        self.conv1 = torch.nn.Conv2d(3, self.input_plane, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._layer(block, 64, num_layers[0], stride=1)
        self.layer2 = self._layer(block, 128, num_layers[1], stride=2)
        self.layer3 = self._layer(block, 256, num_layers[2], stride=2)
        self.layer4 = self._layer(block, 512, num_layers[3], stride=2)
        self.averagePool = torch.nn.AvgPool2d(kernel_size=4, stride=1)
        self.fc = torch.nn.Linear(512*block.expansion, classes)

    
    def _layer(self, block, planes, num_layers, stride=1):
        dim_change = None
        if stride != 1 or planes != self.input_planes*block.expansion:
            dim_change = torch.nn.Sequential(
                            torch.nn.Conv2d(
                               self.input_planes,
                               planes*block.expansion,
                               kernel_size=1,
                               stride=1
                               ),
                            torch.nn.BatchNorm2d(planes*block.expansion)
                          )
        
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes*block.expansion

        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes*block.expansion
        
        # reference of netLayers is sent to sequential function to create a  pytorch model
        return torch.nn.Sequential(*netLayers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x=F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

















        
