import os
import math
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as vgg


batch_size = 8
augment_Gaussian_blur = True



def gaussian_kernel(size, sigma=2, dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.
    
      kernel_size = 2*size + 1
      kernel_size = [kernel_size] * dim
      sigma = [sigma] * dim
      kernel = 1
      meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
      
      for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
          mean = (size - 1) / 2
          kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)
  
      # Make sure sum of values in gaussian kernel equals 1.
      kernel = kernel / torch.sum(kernel)
      # Reshape to depthwise convolutional weight
      kernel = kernel.view(1, 1, *kernel.size())
      kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
      return kernel

def _gaussian_blur(x, size, Blur_sigma):
      kernel = gaussian_kernel(size=size, sigma=Blur_sigma)
      kernel_size = 2*size + 1
      x = x[None,...]
      padding = int((kernel_size - 1) / 2)
      x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
      x = torch.squeeze(F.conv2d(x, kernel, groups=3))
      return x



class NoiseAndBlur():
    """Adds gaussian noise to a tensor.

        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     Noise(0.1, 0.05)),
        >>> ])

    """
    def __init__(self, mean, stddev, image_size, applyBlur, Blur_sigma, Blur_ker_size):
        self.mean = mean
        self.stddev = stddev
        self.image_size = image_size
        self.Blur_sigma = Blur_sigma
        self.Blur_ker_size = Blur_ker_size
        self.applyBlur = applyBlur

    def __call__(self, tensor):
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        if self.applyBlur == True:
          return _gaussian_blur(tensor.add_(noise), self.Blur_ker_size, self.Blur_sigma)
        else:
          return tensor.add_(noise)




def get_data_loader(image_type, image_dir, batch_size):
    """Returns training and test data loaders for a given image type
    """
    num_workers=0
    # resize and normalize the images
    transform1 = transforms.Compose([transforms.Resize((64, 64)) # resize to 128x128
                                    ,transforms.ToTensor()
                                    # ,NoiseAndBlur(0.1, 0.05, image_size = image_size, applyBlur=augment_Gaussian_blur, Blur_sigma=0, Blur_ker_size = 4)
                                    ,transforms.RandomErasing(p=0.2, scale=(0.00002, 0.001), ratio=(0.0001, 0.0006), value=0, inplace=False)
                                    # , tensor_normalizer()
                                    ])
    # get training and test directories
    # resize and normalize the images
    transform2 = transforms.Compose([transforms.Resize((64,64)), # resize to 128x128
                                    transforms.ToTensor()
                                    # , tensor_normalizer()
                                    ])
    
    transform0 = transforms.Compose([transforms.Resize((16,16))
                                    ,transforms.ToTensor()
                                    # ,_gaussian_blur()
                                    # ,NoiseAndBlur(0.1, 0.05, image_size = image_size, applyBlur=augment_Gaussian_blur, Blur_sigma=1, Blur_ker_size = 4)
                                    ,transforms.RandomErasing(p=0.5, scale=(0.00002, 0.001), ratio=(0.0001, 0.0006), value=0, inplace=False) 
                                     # , tensor_normalizer()
                                    ])    

    if image_type == 'lr':
        image_path = image_dir+'/DIV2K/'
        dataset = datasets.ImageFolder(image_path, transform0)
        n = len(dataset)
        train_set, val_set = torch.utils.data.random_split(dataset, [1600, n-1600])
        
        # create and return DataLoaders
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if image_type == 'hr':
        image_path = image_dir + '/lrtohr/'
        train_path = os.path.join(image_path, image_type)
        test_path = os.path.join(image_path, 'test_{}'.format(image_type))
        # define datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_path, transform2)
        test_dataset = datasets.ImageFolder(test_path, transform2)

            # create and return DataLoaders
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        

    return train_loader, test_loader
