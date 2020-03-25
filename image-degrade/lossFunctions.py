import torch
from torch import nn

def realHingeLoss(x):
    if x == 1:
        return 0
    else:
        return x-1

def fakeHingeLoss(x):
    if x==0:
        return -1
    else:
        return -(x+1)

def GANloss(X, fX): #real, fake
    assert X.shape[0] == fX.shape[0] # same batch size
    for i in range(X.shape[0]):
        r1 += min(0, X[i]-1 )
        r2 += min(0, -fX[i]-1 )
    r1 = r1/X.shape[0]
    r2 = r2/fX.shape[0]
    return r1+r2

def pixelLoss(X, fX):  #real, fake
    return nn.MSELoss(X, fX)

# disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()

def real_bce(x):
    bce = nn.BCELoss()
    return bce(output, torch.full((x[0]), 1, device=device))


def fake_bce(x):
    bce = nn.BCELoss()
    return bce(output, torch.full((x[0]), 0, device=device))

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

from collections import namedtuple
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
