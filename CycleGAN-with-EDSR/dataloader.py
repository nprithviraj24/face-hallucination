import math
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os, yaml, argparse

##ONLY for testing
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='config.yaml')

args = parser.parse_args()


with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

###

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
            # transforms.Resize(256),
            # transforms.CenterCrop(115),
            transforms.Resize([ kwargs['exp_params']['lr_imageSize'], kwargs['exp_params']['lr_imageSize'] ]),
            transforms.ToTensor()
            # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        path = os.path.join(kwargs['exp_params']['data_path'], kwargs['exp_params']['lr_datapath'])
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformLR)
        split = int(kwargs['exp_params']['test_split'] * len(dataset) / 100)

        # create and return DataLoaders

    if image_type == 'hr':
        transformHR = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Pad((225, 150), 0, "constant"),
            transforms.Resize((550,550)),
            transforms.CenterCrop((515)),
            transforms.Resize([ kwargs['exp_params']['hr_imageSize'], kwargs['exp_params']['hr_imageSize'] ]),
            transforms.ToTensor()
            # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        path = os.path.join(kwargs['exp_params']['data_path'], kwargs['exp_params']['hr_datapath'])
        # test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        dataset = datasets.ImageFolder(path, transformHR)
        split = int(kwargs['exp_params']['test_split']*len(dataset)/100)

        # create and return DataLoaders
        # data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    train_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-split, split])
    print(str([len(dataset)-split, split]))
    print(image_type+" - Test split: "+str(len(test_set)))
    train_loader = DataLoader(dataset=train_set, batch_size=kwargs['exp_params']['batch_size'],
                              shuffle=kwargs['exp_params']['shuffle'], num_workers=kwargs['exp_params']['num_workers'], drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=kwargs['exp_params']['batch_size'],
                             num_workers=kwargs['exp_params']['num_workers'], drop_last=True)


    return train_loader, test_loader

#
# config['exp_params']['batch_size'] = 1
# a,b = get_data_loader(image_type='hr', exp_params=config['exp_params'])
#
# for i in range(0, len(a)):
#     print(next(iter(b))[0].shape)
