import math
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils import data
import numpy as np

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
            transforms.Resize([ kwargs['exp']['lr_imageSize'], kwargs['exp']['lr_imageSize'] ]),
            transforms.ToTensor()
            # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        path = os.path.join(kwargs['exp']['data_path'], kwargs['exp']['lr_datapath'])
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformLR)
        split = int(kwargs['exp']['test_split'] * len(dataset) / 100)

        # create and return DataLoaders

    if image_type == 'hr':
        transformHR = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Pad((225, 150), 0, "constant"),
            # transforms.CenterCrop((500)),
            transforms.Resize([ kwargs['exp']['hr_imageSize'], kwargs['exp']['hr_imageSize'] ]),
            transforms.ToTensor()
            # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        path = os.path.join(kwargs['exp']['data_path'], kwargs['exp']['hr_datapath'])
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformHR)
        split = int(kwargs['exp']['test_split']*len(dataset)/100)

        # create and return DataLoaders
        # data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-split, split])
    # print(str([len(dataset)-split, split]))
    # print(image_type+" - Test split: "+str(len(test_set)))
    train_loader = DataLoader(dataset=train_set, batch_size=kwargs['exp']['batch_size'],
                              sampler=InfiniteSamplerWrapper(train_set),
                              # shuffle=kwargs['exp']['shuffle'],
                              num_workers=kwargs['exp']['num_workers'])
                              # , drop_last=True)
    test_loader = DataLoader(dataset=test_set
                             , batch_size=kwargs['exp']['batch_size']
                             ,sampler = InfiniteSamplerWrapper(test_set),
                             num_workers=kwargs['exp']['num_workers'])


    return train_loader, test_loader

#
# config['exp']['batch_size'] = 1
# a,b = get_data_loader(image_type='hr', exp=config['exp'])
#
# for i in range(0, len(a)):
#     print(next(iter(b))[0].shape)
