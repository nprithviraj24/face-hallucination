import math
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np
from torch import nn

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31

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


def get_data_loader(image_type,  **kwargs):
    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """
    # assert image_type =='lr' or image_type == 'hr' or image_type=='celeba', "Image type should lr/hr/celeba."

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

    # get training and test directories
    # resize and normalize the images
    if image_type == 'lr':
        transformLR = transforms.Compose([
            transforms.Resize([ kwargs['exp']['lr_imageSize'], kwargs['exp']['lr_imageSize'] ]),
            transforms.ToTensor()
            # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        path = os.path.join(kwargs['exp']['data_path'], kwargs['exp']['lr_datapath'])
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformLR)
        split = int(kwargs['exp']['test_split'] * len(dataset) / 100)

    if image_type == 'celeba':
            transformHR = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(128),
                transforms.Resize([kwargs['exp']['hr_imageSize'], kwargs['exp']['hr_imageSize']]),
                transforms.ToTensor()
                # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])
            path = os.path.join(kwargs['exp']['data_path'], kwargs['exp']['hr_datapath'])
            # define datasets using ImageFolder
            dataset = datasets.ImageFolder(path, transformHR)
            split = int(kwargs['exp']['test_split'] * len(dataset) / 100)

        # create and return DataLoaders

    if image_type == 'hr':
        transformHR = transforms.Compose([

        transforms.Resize(256, interpolation=5),  #2: Bicubic,  #:5 downsampling
        transforms.CenterCrop(150),
        transforms.Resize(128),
        transforms.ToTensor()
        # ,AddGaussianNoise(0,0.002)
        ])
        print("FROM dl2 ~~~")
        path = os.path.join(kwargs['exp']['data_path'], kwargs['exp']['hr_datapath'])
        dataset = datasets.ImageFolder(path, transformHR)
        split = int(kwargs['exp']['test_split']*len(dataset)/100)
        # create and return DataLoaders
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-split, split])
    train_loader = DataLoader(dataset=train_set, batch_size=kwargs['exp']['batch_size'],
                              sampler=InfiniteSamplerWrapper(train_set),
                              num_workers=kwargs['exp']['num_workers'])
    test_loader = DataLoader(dataset=test_set
                             , batch_size=kwargs['exp']['batch_size']
                             ,sampler = InfiniteSamplerWrapper(test_set),
                             num_workers=kwargs['exp']['num_workers'])

    return train_loader, test_loader
