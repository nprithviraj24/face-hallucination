
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# import save code
# from helpers import save_samples, checkpoint
%matplotlib inline
import time
import pylab as pl
from IPython import display

from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models.vgg as vgg
from tqdm import tqdm_notebook

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import namedtuple
import torch.optim as optim

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(convert_size(torch.cuda.memory_allocated(device=device)))



def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)




dataloader_X, test_iter_X = get_data_loader(image_type='lr')
dataloader_Y, test_iter_Y = get_data_loader(image_type='hr')

div2k, test_div2k = get_data_loader(image_dir='DIV2K/', image_type='div2k')



# EDSR in the paper (x4) - from EDSR (x2)
#python main.py --model EDSR --scale 4 --save edsr_x4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --reset --pre_train [pre-trained EDSR_x2 model dir]
rgb_range = 255
n_colors = 3
n_feats = 256 #initially 256
n_resblocks = 32
res_scale= 0.1
kernel_size = 3 
scale = 4


url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)

url = {
    'r16f64x2': 'EDSR_Weights/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'EDSR_Weights/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'EDSR_Weights/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'EDSR_Weights/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'EDSR_Weights/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'EDSR_Weights/edsr_x4-4f62e9ef.pt'
}


class EDSR(nn.Module):
    def __init__(self):
        super(EDSR, self).__init__()
        conv=default_conv
        act = nn.ReLU(True)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))



# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class DTwo(nn.Module):

    def __init__(self, conv_dim=64):
        super(DTwo, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 64, depth 64
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (32, 32, 128)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (16, 16, 256)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (8, 8, 512)
        self.conv5 = conv(conv_dim*8, conv_dim*16, 4) # (4, 4, 1024)
        # self.conv6 = conv(conv_dim*16, conv_dim*32, 4) # (8, 8, 2048)
        # self.conv7 = conv(conv_dim*32, conv_dim*64, 4) # (4, 4, 4096)

        # Classification layer
        self.conv8 = conv(conv_dim*16, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        # out = F.relu(self.conv6(out))
        # out = F.relu(self.conv7(out))
        # last, classification layer
        out = self.conv8(out)
        # print(type(self.conv1))
        return out


class DOne(nn.Module):

    def __init__(self, conv_dim=64):
        super(DOne, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False) # x, y = 256, depth 256
        self.conv2 = conv(conv_dim, conv_dim*2, 4) # (128, 128, 128)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4) # (64, 64, 256)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4) # (32, 32, 512)
        self.conv5 = conv(conv_dim*8, conv_dim*16, 4) # (16, 16, 1024)
        self.conv6 = conv(conv_dim*16, conv_dim*32, 4) # (8, 8, 2048)
        self.conv7 = conv(conv_dim*32, conv_dim*64, 4) # (4, 4, 4096)

        # Classification layer
        self.conv8 = conv(conv_dim*64, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        # last, classification layer
        out = self.conv8(out)
        # print(type(self.conv1))
        return out

c = 64  #initially 256 
batchnorm = True
kernels = [3,3] 
# residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim, bn=True):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3
        
        self.conv_layer1 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                                kernel_size=3, stride=1, padding=1, batch_norm=bn)
        
        self.conv_layer2 = conv(in_channels=conv_dim, out_channels=conv_dim, 
                               kernel_size=3, stride=1, padding=1, batch_norm=bn)
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out_1 = F.relu(self.conv_layer1(x))
        out_2 = x + self.conv_layer2(out_1)
        return out_2


class GTwo(nn.Module):
       def __init__(self):
         super(GTwo, self).__init__()
         self.conv1 = nn.Conv2d(3, c, kernel_size=kernels[0], stride=1, padding=1)

         self.conv2 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=1, padding=1)

         self.conv3 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

         self.residual = self.make_res_layers(ResidualBlock, 6, 64)

         self.conv4 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

         self.conv5 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=1, padding=1)

         self.conv6 = nn.Conv2d(c, 3, kernel_size=kernels[0], stride=1, padding=1)
       

       def make_res_layers(self, block, num_of_layer, conv_dim):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(conv_dim, bn=batchnorm))
        return nn.Sequential(*layers)

       def forward(self, x):
          out = F.relu(self.conv1(x))
          out = F.relu(self.conv2(out))
          out = F.relu(self.conv3(out))
          residual = self.residual(out)
          out = torch.add(out,residual)
          # out = self.upscale4x(out)
          out = F.relu(self.conv4(out))
          out = F.relu(self.conv5(out))
          out = F.relu(self.conv6(out))
          return out



def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""
    
    # Instantiate generators
    G_XtoY = EDSR()
    G_XtoY.load_state_dict(torch.load(G_XtoY.url))
    G_YtoX = GTwo()
    # Instantiate discriminators
    D_X = DTwo(64)
    D_Y = DOne(64)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

# call the function to get models
G_XtoY, G_YtoX, D_X, D_Y = create_model()


def tv_loss(batch, weight):
    return weight * (
    torch.sum(torch.abs(batch[:, :, :, :-1] - batch[:, :, :, 1:])) + 
    torch.sum(torch.abs(batch[:, :, :-1, :] - batch[:, :, 1:, :]))
)  

def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out-1)**2)

def fake_mse_loss(D_out):
    # how close is the produced output from being "fake"?
    return torch.mean(D_out**2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # calculate reconstruction loss
    # as absolute value difference between the real and reconstructed images
    reconstr_loss = torch.mean(torch.abs(real_im - reconstructed_im))
    # return weighted loss
    return lambda_weight*reconstr_loss

mse_loss = torch.nn.MSELoss()

## what worked for me
#lr=0.0000002, beta1 = 0.05, beta2 = 0.00999

# hyperparams for Adam optimizer
lr=0.0000002
beta1=0.05
beta2=0.00999 # default value

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g1_optimizer = optim.Adam(G_XtoY.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
g2_optimizer = optim.Adam(G_YtoX.parameters(), lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])



LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
    
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

vgg_model = vgg.vgg16(pretrained=True)
if torch.cuda.is_available():
  vgg_model.cuda()
loss_network = LossNetwork(vgg_model)
loss_network.eval()

print(convert_size(torch.cuda.memory_allocated(device=device)))



# train the network
#fig = plt.figure(figsize=(12, 8))

#PATH = '/content/drive/My Drive/Datasets/SR/Cycle-EDSR-W/G_XtoY_28.pth'
#checkpoint = torch.load(PATH)

pretrain_epoch = 0
#pretrain_epoch = checkpoint['epoch']

def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=1000):

    print_every=10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)
    # print("0")
    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    # make sure to scale to a range -1 to 1
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    # fixed_X = scale(fixed_X) 
    # fixed_Y = scale(fixed_Y)

    # batches per epoch
    
    # n_epochs = 2
    for epoch in range(pretrain_epoch, n_epochs+1):

      epochG_loss = 0
      runningG_loss = 0
      runningDX_loss = 0
      runningDY_loss = 0
      LOG_INTERVAL = 25

      mbps = 0 #mini batches per epoch

      for batch_id, (x, _) in tqdm_notebook(enumerate(dataloader_X), total=len(dataloader_X)):
        #  with torch.no_grad():
           mbps += 1
           y, a = next(iter(dataloader_Y))
           images_X = x # make sure to scale to a range -1 to 1
           images_Y = y
           del y
           # move images to GPU if available (otherwise stay on CPU)
           device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
           images_X = images_X.to(device)
           images_Y = images_Y.to(device)
          #  print("start:  ",convert_size(torch.cuda.memory_allocated(device=device)))        
          
           d_x_optimizer.zero_grad()
           out_x = D_X(images_X)   
           D_X_real_loss = real_mse_loss(out_x)
           fake_X = G_YtoX(images_Y)
           out_x = D_X(fake_X)   
           D_X_fake_loss = fake_mse_loss(out_x)
           d_x_loss = D_X_real_loss + D_X_fake_loss
           d_x_loss.backward()
           d_x_optimizer.step()
           d_x_loss.detach(); out_x.detach(); D_X_fake_loss.detach();
           runningDX_loss += d_x_loss 
           del D_X_fake_loss, D_X_real_loss, out_x, fake_X , d_x_loss
           torch.cuda.empty_cache()
           
          #  print("end: DX block  and start DY", convert_size(torch.cuda.memory_allocated(device=device)))

           d_y_optimizer.zero_grad()
           out_y = D_Y(images_Y)
           D_Y_real_loss = real_mse_loss(out_y)
           fake_Y = G_XtoY(images_X)
           out_y = D_Y(fake_Y)
           D_Y_fake_loss = fake_mse_loss(out_y)
           d_y_loss = D_Y_real_loss + D_Y_fake_loss
           d_y_loss.backward()
           d_y_optimizer.step()
           d_y_loss.detach()
           runningDY_loss += d_y_loss
           del D_Y_fake_loss, D_Y_real_loss, out_y, fake_Y
           torch.cuda.empty_cache()
          #  print("End: DY ",convert_size(torch.cuda.memory_allocated(device=device)))  


           g_optimizer.zero_grad()
           fake_Y = G_XtoY(images_X)
           out_y = D_Y(fake_Y)
           g_XtoY_loss = real_mse_loss(out_y)
           reconstructed_X = G_YtoX(fake_Y)
           
           reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=50)
           
           featuresY = loss_network(images_Y);           
           featuresFakeY = loss_network(fake_Y); 
           
           CONTENT_WEIGHT = 0.00001          
           contentloss = CONTENT_WEIGHT * mse_loss(featuresY[1].data, featuresFakeY[1].data)
           del featuresY, featuresFakeY; torch.cuda.empty_cache()
           
           IDENTITY_WEIGHT = 0.0001
           downsample = nn.Upsample(scale_factor=0.25, mode='bicubic')
           identity_loss = IDENTITY_WEIGHT * mse_loss(downsample(fake_Y), images_X )

           TOTAL_VARIATION_WEIGHT = 0.0001
           tvloss = TOTAL_VARIATION_WEIGHT + tv_loss(fake_Y, 0.5)
           
           g_total_loss = g_XtoY_loss + reconstructed_x_loss + identity_loss + tvloss + contentloss
          #  tvloss + content_loss_Y + identity_loss
           g_total_loss.backward()
           g_optimizer.step()
           del out_y, fake_Y, g_XtoY_loss, reconstructed_x_loss, reconstructed_X
          #  , tvloss content_loss_Y, identity_loss
          #  print("end: ", convert_size(torch.cuda.memory_allocated(device=device)))

           runningG_loss += g_total_loss
           

           if mbps % LOG_INTERVAL == 0:  
             with torch.no_grad():
              G_XtoY.eval() # set generators to eval mode for sample generation
              fakeY = G_XtoY(fixed_X.to(device))
              imshow(torchvision.utils.make_grid(fixed_X.cpu()))
              imshow(torchvision.utils.make_grid(fakeY.cpu()))
              G_XtoY.train()          
              print('Mini-batch no: {}, at epoch [{:3d}/{:3d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(mbps, epoch, n_epochs,  runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))

      with torch.no_grad():
        G_XtoY.eval() # set generators to eval mode for sample generation
        fakeY = G_XtoY(fixed_X.to(device))
        imshow(torchvision.utils.make_grid(fixed_X.cpu()))
        imshow(torchvision.utils.make_grid(fakeY.cpu()))
        G_XtoY.train()
        # print("Epoch loss:  ", epochG_loss/)
      losses.append((runningDX_loss/mbps, runningDY_loss/mbps, runningG_loss/mbps))
      print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))
              
      
    return losses


n_epochs = 200 # keep this small when testing if a model first works

losses = training_loop(div2k, dataloader_Y, test_div2k, test_iter_Y, n_epochs=n_epochs)