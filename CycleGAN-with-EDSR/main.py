import dataloader as dl
from Discriminators import Discriminator as D
from Generators import EDSR as G1
from Generators import Gtwo as G2
import loss

import yaml, argparse
import torchvision.models.vgg as vgg
import torch
from torch import nn
import torch.optim as optim


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='config.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
# A = args['EDSR']
# print(type(args))
def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6, **kwargs):
    """Builds the generators and discriminators."""

    url = {
        'r16f64x2': '/tmp/EDSR_Weights/edsr_baseline_x2-1bc95232.pt',
        'r16f64x3': '/tmp/EDSR_Weights/edsr_baseline_x3-abf2a44e.pt',
        'r16f64x4': '/tmp/EDSR_Weights/edsr_baseline_x4-6b446fab.pt',
        'r32f256x2': '/tmp/EDSR_Weights/edsr_x2-0edfb8a3.pt',
        'r32f256x3': '/tmp/EDSR_Weights/edsr_x3-ea3ef2c6.pt',
        'r32f256x4': '/tmp/EDSR_Weights/edsr_x4-4f62e9ef.pt'
    }

    pretrained_Weights = 'r{}f{}x{}'.format(kwargs['EDSR']['n_resblocks'], kwargs['EDSR']['n_feats'], kwargs['EDSR']['scale'])

    # Instantiate generators

    device = torch.device('cpu')
    if torch.cuda.is_available(): device = torch.device("cuda")

    print(device)
    G_XtoY = G1.EDSR(EDSR=kwargs['EDSR']).to(device)
    # print(G_XtoY.url)
    # kwargs['EDSR']
    G_XtoY.load_state_dict(torch.load(url[G_XtoY.url]))

    G_YtoX = G2.GTwo(Gtwo=kwargs['Gtwo']).to(device)
    # Instantiate discriminators
    D_X = D.DTwo(64).to(device)
    D_Y = D.DOne(64).to(device)

    # move models to GPU, if available
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    #     G_XtoY.to(device)
    #     G_YtoX.to(device)
    #     D_X.to(device)
    #     D_Y.to(device)
    #     print('Models moved to GPU.')
    # else:
    #     print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
# del dataloader_X, test_dataloader_X
# del dataloader_Y, test_dataloader_Y
# print(config['EDSR'])
G_XtoY, G_YtoX, D_X, D_Y = create_model(EDSR=config['EDSR'], Gtwo=config['Gtwo'])
dataloader_X, test_iter_X = dl.get_data_loader(image_type='lr', exp_params=config['exp_params'])
dataloader_Y, test_iter_Y = dl.get_data_loader(image_type='hr', exp_params=config['exp_params'])

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

# import save code
# from helpers import save_samples, checkpoint
import time
import pylab as pl
from IPython import display

# train the network
pretrain_epoch = 0
# PATH = '/content/drive/My Drive/Datasets/SR/Cycle-EDSR-W/G_XtoY_28.pth'
# checkpoint = torch.load(PATH)

# pretrain_epoch = checkpoint['epoch']
test_iter_X = next(iter(test_iter_Y))[0]

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

    ## We can decide if we want to scale.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]


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
           D_X_real_loss = loss.real_mse_loss(out_x)
           fake_X = G_YtoX(images_Y)
           out_x = D_X(fake_X)
           D_X_fake_loss = loss.fake_mse_loss(out_x)
           d_x_loss = D_X_real_loss + D_X_fake_loss
           d_x_loss.backward()
           d_x_optimizer.step()
           d_x_loss.detach(); out_x.detach(); D_X_fake_loss.detach();
           runningDX_loss += d_x_loss
           del D_X_fake_loss, D_X_real_loss, out_x, fake_X
           torch.cuda.empty_cache()

          #  print("end: DX block  and start DY", convert_size(torch.cuda.memory_allocated(device=device)))

           d_y_optimizer.zero_grad()
           out_y = D_Y(images_Y)
           D_Y_real_loss = loss.real_mse_loss(out_y)
           fake_Y = G_XtoY(images_X)
           out_y = D_Y(fake_Y)
           D_Y_fake_loss = loss.fake_mse_loss(out_y)
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
           g_XtoY_loss = loss.real_mse_loss(out_y)
           reconstructed_X = G_YtoX(fake_Y)

           reconstructed_x_loss = loss.cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=50)

           featuresY = loss_network(images_Y);
           featuresFakeY = loss_network(fake_Y);

           CONTENT_WEIGHT = 10
           contentloss = CONTENT_WEIGHT * loss.mse_loss(featuresY[1].data, featuresFakeY[1].data)
           del featuresY, featuresFakeY; torch.cuda.empty_cache()

           IDENTITY_WEIGHT = 100000
           downsample = nn.Upsample(scale_factor=0.25, mode='bicubic')
           identity_loss = IDENTITY_WEIGHT * loss.mse_loss(downsample(fake_Y), images_X )

           TOTAL_VARIATION_WEIGHT = 0.01
           tvloss = TOTAL_VARIATION_WEIGHT * loss.tv_loss(fake_Y, 0.25)

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
              # imshow(torchvision.utils.make_grid(fixed_X.cpu()))
              G_XtoY.train()
              print('Mini-batch no: {}, at epoch [{:3d}/{:3d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f}| g_total_loss: {:6.4f}'.format(mbps, epoch, n_epochs,  d_x_loss.item() , d_y_loss.item() , g_total_loss.item() ))
              print(' TV-loss: ', tvloss.item(), '  content loss:', contentloss.item(), '  identity loss:', identity_loss.item() )

      with torch.no_grad():
        G_XtoY.eval() # set generators to eval mode for sample generation
        fakeY = G_XtoY(fixed_X.to(device))
        G_XtoY.train()
        # print("Epoch loss:  ", epochG_loss/)
      losses.append((runningDX_loss/mbps, runningDY_loss/mbps, runningG_loss/mbps))
      print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))


    return losses


n_epochs = 200 # keep this small when testing if a model first works

losses = training_loop(div2k, dataloader_Y, test_div2k, test_iter_Y, n_epochs=n_epochs)

