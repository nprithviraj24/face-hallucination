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

def get_data_loader(image_type, image_dir='lrtohr', image_size=64, batch_size=8, num_workers=0):
    """Returns training and test data loaders for a given image type, either 'summer' or 'winter'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """

    # resize and normalize the images
    transform1 = transforms.Compose([transforms.Resize((image_size, image_size)), # resize to 128x128
                                    transforms.ToTensor()])
    # get training and test directories
    # resize and normalize the images
    transform2 = transforms.Compose([transforms.Resize((256,256)), # resize to 128x128
                                    transforms.ToTensor()])

    image_path = './' + image_dir
    train_path = os.path.join(image_path, image_type)
    test_path = os.path.join(image_path, 'test_{}'.format(image_type))

    if image_type == 'lr':
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform1)
        test_dataset = datasets.ImageFolder(test_path, transform1)

        # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if image_type == 'hr':
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform2)
        test_dataset = datasets.ImageFolder(test_path, transform2)

            # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

