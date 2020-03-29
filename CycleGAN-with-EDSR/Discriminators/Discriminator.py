import torch
import torch.nn as nn
import torch.nn.functional as F

## NO SPECTRAL NORMALISATION!

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
        # out = F.relu(self.conv6(out))
        # out = F.relu(self.conv7(out))
        # last, classification layer
        out = self.conv8(out)
        # print(type(self.conv1))
        return out

## DTWO:


class DTwo(nn.Module):

        def __init__(self, conv_dim=64):
            super(DTwo, self).__init__()

            # Define all convolutional layers
            # Should accept an RGB image as input and output a single value

            # Convolutional layers, increasing in depth
            # first layer has *no* batchnorm
            self.conv1 = conv(3, conv_dim, 4, batch_norm=False)  # x, y = 64, depth 64
            self.conv2 = conv(conv_dim, conv_dim * 2, 4)  # (32, 32, 128)
            self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)  # (16, 16, 256)
            self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)  # (8, 8, 512)
            self.conv5 = conv(conv_dim * 8, conv_dim * 16, 4)  # (4, 4, 1024)
            # self.conv6 = conv(conv_dim*16, conv_dim*32, 4) # (8, 8, 2048)
            # self.conv7 = conv(conv_dim*32, conv_dim*64, 4) # (4, 4, 4096)

            # Classification layer
            self.conv8 = conv(conv_dim * 16, 1, 4, stride=1, batch_norm=False)

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


class DThree(nn.Module):

    def __init__(self, conv_dim=64):
        super(DThree, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value

        # Convolutional layers, increasing in depth
        # first layer has *no* batchnorm
        #in: 16,16, 3
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)  # out: (8,8,64)
        self.conv2 = conv(conv_dim, conv_dim * 2,  4, stride=1, padding=1)  #out: (8, 8, 128)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)  # out: (4, 4, 256)

        # Classification layer
        self.conv8 = conv(conv_dim * 4, 1, 4, stride=1, batch_norm=False)

    def forward(self, x):
        # relu applied to all conv layers but last
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        # out = F.relu(self.conv4(out))
        # out = F.relu(self.conv5(out))
        # out = F.relu(self.conv6(out))
        # out = F.relu(self.conv7(out))
        # last, classification layer
        out = self.conv8(out)
        # print(type(self.conv1))
        return out



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


#
# d1 = NLayerDiscriminator(input_nc=3 , n_layers=2)
# d2 = NLayerDiscriminator(input_nc=3 , n_layers=4, ndf=128)
#
#
# print(d2(torch.rand(size=[8,3,64,64])).shape)