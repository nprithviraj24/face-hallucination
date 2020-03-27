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

    # resize and normalize the images
    transformLR = transforms.Compose([
        transforms.Resize(kwargs['exp_params']['lr_imageSize']),
        transforms.ToTensor()])
    # get training and test directories
    # resize and normalize the images

    # train_path = os.path.join(kwargs['exp_params']['img_'], image_type)
    # test_path = os.path.join(kwargs['exp_params']['image_path'], 'test_{}'.format(image_type))

    # if image_type == 'lr':
    #     # define datasets using ImageFolder
    #     train_dataset = datasets.ImageFolder(train_path, transformLR)
    #     test_dataset = datasets.ImageFolder(test_path, transformLR)
    #
    #     # create and return DataLoaders
    #     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    if image_type == 'lr':
        train_path = os.path.join(kwargs['exp_params']['data_path'], '')
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(kwargs['exp_params']['lr_datapath'], transformLR)
        n = len(dataset)
        # test_dataset = datasets.ImageFolder(test_path, transform2)

            # create and return DataLoaders
        # data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [1600, n-1600])


    if image_type == 'hr':
        transformHR = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Pad((225, 150), 0, "constant"),
            transforms.CenterCrop((500)),
            transforms.Resize(kwargs['exp_params']['hr_imageSize']),
            transforms.ToTensor()
        ])
        train_path = os.path.join(kwargs['exp_params']['data_path'], '')
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(kwargs['exp_params']['hr_datapath'], transformLR)
        n = len(dataset)
        # test_dataset = datasets.ImageFolder(test_path, transform2)

        # create and return DataLoaders
        # data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        train_set, val_set = torch.utils.data.random_split(dataset, [1600, n - 1600])

    train_loader = DataLoader(dataset=train_set, batch_size=kwargs['exp_params']['batch_size'], shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, drop_last=True)


    return train_loader, test_loader

