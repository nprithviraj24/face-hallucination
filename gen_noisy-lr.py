##Transforming a dataset:
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms.functional as tfunc
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


import numbers
import numpy as np
from PIL import ImageFilter

class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class DynamicGNoise(nn.Module):
    def __init__(self, shape, std=0.0):
        super().__init__()
        self.noise = torch.zeros(3,shape, shape)
        self.std = std

    def forward(self, x):
        x = torchvision.transforms.functional.to_tensor(x)
        if not self.training: return x
        self.noise.data.normal_(0, std=self.std)

        # print(x.size(), self.noise.size())
        x = x + self.noise
        return torchvision.transforms.functional.to_pil_image(x)


data_dir = "/tmp/Datasets/celebA-264/safe"
save_dir = "/tmp/Datasets/celebA-264/lr/264"

z=32
y=150
trnfms = tfs.Compose([
    # tfs.ToTensor(),
    GaussianSmoothing([0,2]),
    # tfs.ToPILImage(),

    tfs.Resize(256, interpolation=5),  #2: Bicubic,  #:5 downsampling

    tfs.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.5, hue=0.1),

    tfs.CenterCrop(y),

    DynamicGNoise(y),

    tfs.Resize(z),

    # GaussianSmoothing(3,5,1),

    tfs.ToTensor(),

    # AddGaussianNoise(0,0.01)
    ])

data = datasets.ImageFolder(data_dir, trnfms)
# smoothing = GaussianSmoothing(3,3,1.5)
loader = DataLoader(dataset=data, batch_size=1, shuffle=True, drop_last=False)

for btch, (x,_) in tqdm(enumerate(loader), total=len(loader)):

    vutils.save_image(x, save_dir + str(btch) + ".png")






