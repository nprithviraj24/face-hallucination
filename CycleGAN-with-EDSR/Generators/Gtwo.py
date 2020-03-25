import torch
from torch import nn
import torch.nn.functional as F



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


    # helper conv function
def conv( in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        """Creates a convolutional layer, with optional batch normalization.
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        layers.append(conv_layer)

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)


class GTwo(nn.Module):
    def __init__(self):
        super(GTwo, self).__init__()
        self.conv1 = nn.Conv2d(3, c, kernel_size=kernels[0], stride=1, padding=1)

        self.conv2 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=1, padding=1)

        self.conv3 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

        self.residual = self.make_res_layers(ResidualBlock, 6, 64)

        self.conv4 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

        #  self.conv7 = nn.Conv2d(c, c, kernel_size=kernels[1], stride=2, padding=1)

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
        out = torch.add(out, residual)
        # out = self.upscale4x(out)
        out = F.relu(self.conv4(out))
        # out = F.relu(self.conv7(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        return out