# Import necessary modules
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
# Uniformly set the hyperparameters of Linears
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = linear

    def forward(self, x):
        return self.linear(x)


# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module
# Uniformly set the hyperparameters of Conv2d
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = conv

    def forward(self, x):
        return self.conv(x)


# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


# "learned affine transform" A
class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        # self.conv.weight.data.normal_()
        # torch.nn.init.xavier_uniform(self.conv.weight)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)
        # torch.nn.init.xavier_uniform(self.lff.weight)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

head = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU()
)

b1 = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU()
)

b2 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU()
)

b3 = nn.Sequential(

    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU()
)

b4 = nn.Sequential(
nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
)

class Degrade(nn.Module):
    def __init__(self, latent_dim=256):
        super(Degrade, self).__init__()
        self.head = head

        self.block1 = b1 #inc = 256, outc =  256
        self.affineT1 = FC_A(dim_latent=latent_dim, n_channel=256)
        self.adaIn1 = AdaIn(n_channel=256)

        self.block2 = b2 #inc = 256, outc =
        self.affineT2 = FC_A(dim_latent=latent_dim, n_channel=128)
        self.adaIn2 = AdaIn(n_channel=128)

        self.block3 = b3 #inc = 256, outc =
        self.affineT3 = FC_A(dim_latent=latent_dim, n_channel=64) # self.block2 = b2
        self.adaIn3 = AdaIn(n_channel=64) # self.block3 = b3
        # self.block4 = b4

        self.block4 = b4
        self.affineT4 = FC_A(dim_latent=latent_dim, n_channel=64)
        self.adaIn4 = AdaIn(n_channel=64)

        self.tail = nn.Conv2d(64, 3, (3, 3))

    def forward(self, inp, lat_style):
        out = self.head(inp)
        out = self.block1(out)
        # print(out.shape)
        out = self.adaIn1(image=out, style=self.affineT1(lat_style))

        out = self.block2(out)
        # print(out.shape)
        out = self.adaIn2(image=out, style=self.affineT2(lat_style))

        out = self.block3(out)
        # print(out.shape)
        out = self.adaIn3(image=out, style=self.affineT3(lat_style))

        out = self.block4(out)
        # print(out.shape)
        out = self.adaIn4(image=out, style=self.affineT4(lat_style))

        out = self.tail(out)

        return out

#
# test = Degrade()
# print(test(torch.randn(1,512, 8,8), torch.randn(1,256) ).shape)