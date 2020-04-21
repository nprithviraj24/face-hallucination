import math
from torch import nn
import numpy as np
import torch

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def tensor_normalizer():
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

def imshow(img):
    npimg = img.detach().cpu().numpy()
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(1.0)

def recover_image(img):
    return ((img * np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)) ).transpose(0, 2, 3, 1) * 255. ).clip(0, 255).astype(np.uint8)


def downsample4x(x):
    #using avgpooling
    avg = nn.AvgPool2d(kernel_size=2, stride=2)
    return avg(avg(x))

def upsample(factor, data):
    return nn.Upsample(scale_factor=factor, mode='bicubic', align_corners=True)(data)


