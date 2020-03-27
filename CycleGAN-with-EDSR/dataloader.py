import math
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

def get_data_loader(image_type,  **kwargs):
    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """
    assert image_type =='lr' or image_type == 'hr', "Image type should lr or hr."

    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

    # get training and test directories
    # resize and normalize the images
    if image_type == 'lr':
        transformLR = transforms.Compose([
            transforms.Resize(kwargs['exp_params']['lr_imageSize']),
            transforms.ToTensor()
        ])

        path = os.path.join(kwargs['exp_params']['data_path'], kwargs['exp_params']['lr_datapath'])
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformLR)
        split = int(kwargs['exp_params']['test_split'] * len(dataset) / 100)

        # create and return DataLoaders
        train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset) - split, split])

    if image_type == 'hr':
        transformHR = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Pad((225, 150), 0, "constant"),
            transforms.CenterCrop((500)),
            transforms.Resize(kwargs['exp_params']['hr_imageSize']),
            transforms.ToTensor()
        ])
        path = os.path.join(kwargs['exp_params']['data_path'], kwargs['exp_params']['hr_datapath'])
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformHR)
        split = int(kwargs['exp_params']['test_split']*len(dataset)/100)

        # create and return DataLoaders
        # data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-split, split])

    train_loader = DataLoader(dataset=train_set, batch_size=kwargs['exp_params']['batch_size'],
                              shuffle=kwargs['exp_params']['shuffle'], num_workers=kwargs['exp_params']['num_workers'], drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=kwargs['exp_params']['batch_size'],
                             num_workers=kwargs['exp_params']['num_workers'], drop_last=True)


    return train_loader, test_loader

