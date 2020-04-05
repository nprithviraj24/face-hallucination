import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from models import FFDNet
from utils import batch_psnr, normalize, init_logger_ipol, \
				variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb

device = torch.device("cuda")

datasetpath = '/tmp/Datasets/DIV2k/images/'
model_fn = 'models/net_rgb.pth'
trnfms = transforms.Compose([
            transforms.Resize([64,64], interpolation=3)
            ,transforms.ToTensor()
            # ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
datafolder = torchvision.datasets.ImageFolder(datasetpath, trnfms)

loader = torch.utils.data.DataLoader(dataset=datafolder, batch_size=64, shuffle=True)

net = nn.DataParallel(FFDNet(num_input_channels=3), device_ids=[0]).cuda()
model_fn = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_fn)
net.load_state_dict( torch.load(model_fn) )

fixedData = next(iter(loader))[0].to(device)
# print(fixedData.shape)
for noise_sigma in range(4,50, 2):

    out = net(fixedData, noise_sigma )

    torchvisions.utils.save_image(
        out,
        str(noise_sigma)+".png",
        nrow=8,
        padding=1,
        normalize=True,
        range=[0,1]
    )

# print(torch.max(fixedData))
