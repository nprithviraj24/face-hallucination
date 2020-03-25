import math
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, device, scale_factor=4):
        super(Generator, self).__init__()
        
        firstPhase = [ResidualBlock(64) for _ in range(12)]
        self.firstPhase = nn.Sequential(*firstPhase)

        secondPhase = [ResidualBlock(64) for _ in range(3)]
        self.secondPhase = nn.Sequential(*secondPhase)

        thirdPhase = [ResidualBlock(64) for _ in range(2)]
        self.thirdPhase = nn.Sequential(*thirdPhase)
  
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.last = nn.Conv2d(64, 3, kernel_size=3)

    def forward(self, x):
        res1 = x
        x = self.firstPhase(x)
        x = res1 + x

        res2 = self.upsample(x)
        x = self.secondPhase(res2)
        x = res2 + x

        res3 = self.upsample(x)
        x = self.thirdPhase(res3)
        x = res3 + x

        last = self.last(x)
        return last


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.bn1(x)
        residual = self.relu(residual)
        residual = self.conv1(residual)

        residual = self.bn2(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        

        return x + residual
