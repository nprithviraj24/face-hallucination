import math
# from spectral_normalization import SpectralNorm

import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)





class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block0 = DResBlock(64)
        self.block1 = DResBlock(64)
        self.block2 = DResBlock(64)
        self.block3 = DResBlock(64)

        self.block4 = DResBlock(64)
        self.block5 = DResBlock(64)
        self.block6 = DResBlock(64)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # self.fc0 = SpectralNorm(nn.Linear())
        self.fc1 = SpectralNorm(nn.Linear(1024, 256))
        self.fc2 = SpectralNorm(nn.Linear(256, 1))


    def forward(self, x):

        b0 = self.block0(self.conv(x))
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        #print(b2.shape)
        b3 = self.block3(b2 )
        b3 = self.pool(b3)
        # print("b3: ", b3.shape)
        b4 = self.block4(b3)
        b4 = self.pool(b4)
        # print("b4: ", b4.shape)

        b5 = self.block5(b4)
        out = self.pool(b5)
        # print("b5:  ", b5.shape)
        #
        # b6 = self.block5(b5)
        # out = self.pool(b6)
        out = out.view(out.size(0), -1)
        # print(": out   ", out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class DResBlock(nn.Module):
    def __init__(self, channels):
        super(DResBlock, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.relu = nn.ReLU()
        self.conv2 = SpectralNorm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        
    def forward(self, x):
        residual = self.relu(x)
        residual = self.conv1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        return x + residual
