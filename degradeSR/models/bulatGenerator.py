import math
import torch
from torch import nn

gNoise = [50, 10]  ##mean and stddev


class Generator(nn.Module):
    def __init__(self, device, scale_factor=4):
        super(Generator, self).__init__()

        upsample_block_num = int(math.log(scale_factor, 2))
        self.device = device

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = ResidualBlock(64)
        self.block8 = ResidualBlock(64)
        self.block9 = ResidualBlock(64)

        block10 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block10.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block10 = nn.Sequential(*block10)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def forward(self, x):
        # noise_tensor = torch.FloatTensor(x.size()).normal_(gNoise[0], gNoise[1] ).to(self.device)
        # #64*64 plus noise
        # x = torch.add(x, noise_tensor )

        block1 = self.block1(x)
        block1 = self.pool(block1)
        # 32*32
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block3 = self.pool(block3)
        # 16*16
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block5 = self.pool(block5)
        # 8*8
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block7 = self.pool(block7)
        # 4*4
        block8 = self.block8(block7)
        block9 = self.block9(block8)
        # upsample
        block10 = self.block10(block9)

        return block10


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


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
