
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
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
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