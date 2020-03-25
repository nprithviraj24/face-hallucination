import dataloader as dl
from Discriminators import Discriminator as D
from Generators import EDSR, GTwo
import loss
import torchvision.models.vgg as vgg
import torch
import torch.optim as optim


def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators."""

    # Instantiate generators
    G_XtoY = EDSR()
    G_XtoY.load_state_dict(torch.load(G_XtoY.url))
    G_YtoX = GTwo()
    # Instantiate discriminators
    D_X = D.DTwo(64)
    D_Y = D.DOne(64)

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

# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
# del dataloader_X, test_dataloader_X
# del dataloader_Y, test_dataloader_Y
_, _, _, _ = create_model()
dataloader_X, test_iter_X = dl.get_data_loader(image_type='lr')
dataloader_Y, test_iter_Y = dl.get_data_loader(image_type='hr')

# next(iter(dataloader_X))[0][0]

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


c = 64  # initially 256
batchnorm = True
kernels = [3, 3]

## what worked for me
#lr=0.0000002, beta1 = 0.05, beta2 = 0.00999

# hyperparams for Adam optimizer
lr=0.000002
beta1=0.5
beta2=0.99 # default value

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g1_optimizer = optim.Adam(G_XtoY.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
g2_optimizer = optim.Adam(G_YtoX.parameters(), lr, [beta1, beta2])
d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

vgg_model = vgg.vgg16(pretrained=True)
if torch.cuda.is_available():
  vgg_model.cuda()
loss_network = loss.LossNetwork(vgg_model)
loss_network.eval()