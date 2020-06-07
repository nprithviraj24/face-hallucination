from models import degrade
from models import vae_style
from torch import nn
import torch
from models.vgg import vgg



##weight initialization tooo?
class Net(nn.Module):

    def __init__(self, encoder):
        super(Net, self).__init__()
        self.vae = vae_style.se50_fc() ## LATENT 256: Hardcoded

        for child in encoder.children():
            for param in child.parameters():
                param.requires_grad = False
        enc_layers = list(encoder.children())

        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # fix the encoder
        self.degrade = degrade.Degrade()
        # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

        # extract relu4_1 from input image

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def forward(self, lr, hr):

        hr_feat = self.encode(hr)
        lr_feat, mu, logvar = self.vae(lr)
        out = self.degrade(hr_feat, lr_feat)

        return out, mu, logvar

# vgg = nn.Sequential(*list(vgg.children())[:31])
# network = Net(vgg)
# a,b,c = network(torch.zeros(1,3,64,64), torch.ones(1,3,64,64))
# print(a.shape)










