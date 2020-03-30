import dataloader as dl

# from Discriminators import Spectral_Normalization
from Discriminators import Discriminator as D
from Generators import EDSR as G1
from Generators import Gtwo as G2
import loss
from FID import fid_score as FID

# from test_tube import Experiment
import os, yaml, argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
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

torch.manual_seed(config["model_params"]["manual_seed"])

# A = args['EDSR']
# print(type(args))
if config["logging_params"]["log"]:

    print("------- Logging active -------- ")
    writer = SummaryWriter(config['logging_params']['save_dir'])
else:
    print("+++++++ No logger activated. +++++++ ")

device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device("cuda")

def create_model(device, g_conv_dim=64, d_conv_dim=64, n_res_blocks=6, **kwargs):
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

    G_XtoY = G1.EDSR(EDSR=kwargs['EDSR']).to(device)
    G_XtoY.load_state_dict(torch.load(url[G_XtoY.url]))

    G_YtoX = G2.GTwo(Gtwo=kwargs['Gtwo']).to(device)

    D_X = D.SpecNorm_NLayer(input_nc=3, n_layers=2).to(device)
    D_Y = D.SpecNorm_NLayer(input_nc=3, n_layers=4, ndf=128).to(device)

    return G_XtoY, G_YtoX, D_X, D_Y

# Create train and test dataloaders for images from the two domains X and Y
G_XtoY, G_YtoX, D_X, D_Y = create_model(EDSR=config['EDSR'], Gtwo=config['Gtwo'], device=device)
dataloader_X, test_iter_X = dl.get_data_loader(image_type='lr', exp_params=config['exp_params'])
dataloader_Y, test_iter_Y = dl.get_data_loader(image_type='hr', exp_params=config['exp_params'])

### FOR FID
reference_frame = '/tmp/Datasets/celeba/img_align_celeba/celeba'

transFORM = transforms.Compose([
    transforms.Resize([16, 16], interpolation=3),
    transforms.ToTensor()
])

images = datasets.ImageFolder(os.path.join(config['exp_params']['data_path'], config['exp_params']['lr_datapath']),
                              transFORM)
images = DataLoader(dataset=images, batch_size=1)

def calc_fid(model):
    model.eval()
    for batch_id, (x, _) in tqdm(enumerate(images), total=len(images)):

        vutils.save_image(
            model(x.to(device)),  # tensor
            config['EDSR']['save_dir'] + "/" + str(batch_id) + ".png"
        )
    model.train()

    return FID.calculate_fid_given_paths( [reference_frame, config['EDSR']['save_dir']],  # paths
          100,  # batch size
          True,  # cuda
          2048 ) # dims


c = 64  # initially 256
batchnorm = True
kernels = [3, 3]

## what worked for me
#lr=0.0000002, beta1 = 0.05, beta2 = 0.00999

# hyperparams for Adam optimizer
lr=0.00002
beta1=0.5
beta2=0.99 # default value

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g1_optimizer = optim.Adam(G_XtoY.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
g2_optimizer = optim.Adam(G_YtoX.parameters(), lr, [beta1, beta2])
d_x_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D_X.parameters()), lr=lr, betas=(0.0,0.9))
# d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
d_y_optimizer = optim.Adam(filter(lambda p: p.requires_grad, D_Y.parameters()), lr=lr, betas=(0.0,0.9))
# d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])

vgg_model = vgg.vgg16(pretrained=True)
if torch.cuda.is_available():
  vgg_model.cuda()
loss_network = loss.LossNetwork(vgg_model)
loss_network.eval()

# train the network
pretrain_epoch = 0
# PATH = '/content/drive/My Drive/Datasets/SR/Cycle-EDSR-W/G_XtoY_28.pth'
# checkpoint = torch.load(PATH)
mse_loss = torch.nn.MSELoss()
# pretrain_epoch = checkpoint['epoch']
test_iter_X = next(iter(test_iter_Y))[0]

def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=1000):

    print_every=10

    # keep track of losses over time
    losses = []

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    # make sure to scale to a range -1 to 1

    fixed_Y = next(iter(dataloader_Y))[0] ## shape: [batchsize, channels, height, width]
    fixed_X = next(iter(dataloader_X))[0]  ## shape: [batchsize, channels, height, width]

    for epoch in range(pretrain_epoch, config['hyperparams']['epochs']+1):
      print("inside")
      epochG_loss = 0
      runningG_loss = 0
      runningDX_loss = 0
      runningDY_loss = 0
      mbps = 0 #mini batches per epoch

      for batch_id, (x, _) in tqdm(enumerate(dataloader_X), total=len(dataloader_X)):
        #  with torch.no_grad():
           mbps += 1
           y, _ = next(iter(dataloader_Y))
           images_X = x # make sure to scale to a range -1 to 1
           images_Y = y
           del y
           # move images to GPU if available (otherwise stay on CPU)
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

           reconstructed_x_loss = loss.cycle_consistency_loss(images_X, reconstructed_X,
                                                              lambda_weight= config['hyperparams']['lambda_weight'])

           featuresY = loss_network(images_Y)
           featuresFakeY = loss_network(fake_Y)

           # print("\nFake Y: ", fake_Y.shape, "   imagesY: ", images_Y.shape,"\n",featuresY[1].data.shape, "   ", featuresFakeY[1].data.shape)
           # exit()
           CONTENT_WEIGHT = config['hyperparams']['Content_Weight']
           contentloss = CONTENT_WEIGHT * mse_loss(featuresY[1].data, featuresFakeY[1].data)
           del featuresY, featuresFakeY; torch.cuda.empty_cache()

           IDENTITY_WEIGHT = config['hyperparams']['Identity_Weight']

           downsample = nn.Upsample(scale_factor=0.25, mode='bicubic')
           identity_loss = IDENTITY_WEIGHT * mse_loss(downsample(fake_Y), images_X )

           TOTAL_VARIATION_WEIGHT = config['hyperparams']['TotalVariation_Weight']
           tvloss = TOTAL_VARIATION_WEIGHT * loss.tv_loss(fake_Y, 0.25)

           g_total_loss = g_XtoY_loss + reconstructed_x_loss + identity_loss + tvloss + contentloss
          #  tvloss + content_loss_Y + identity_loss
           g_total_loss.backward()
           g_optimizer.step()
           del out_y, fake_Y, g_XtoY_loss, reconstructed_x_loss, reconstructed_X
          #  , tvloss content_loss_Y, identity_loss
          #  print("end: ", convert_size(torch.cuda.memory_allocated(device=device)))

           runningG_loss += g_total_loss


           if config["logging_params"]["log"] and mbps % config["logging_params"]["log_interval"] == 0:

              if mbps%config["logging_params"]["image_log"] ==0:
                 with torch.no_grad():
                   G_XtoY.eval()
                   writer.add_image(tag=str(epoch)+'/'+str(mbps), img_tensor=vutils.make_grid( G_XtoY(fixed_X.to(device)), normalize=True,
                                                      pad_value=1, nrow=8), global_step=epoch)
                   G_XtoY.train()

              writer.add_scalars('D', {'Y': d_y_loss.item(), 'X': d_x_loss.item()}, epoch)
              writer.add_scalars('G', {'TV': tvloss.item(),
                                       'Content': contentloss.item(),
                                       'Identity': identity_loss.item()}, epoch)


           print('Mini-batch no: {}, at epoch [{:3d}/{:3d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f}| g_total_loss: {:6.4f}'
                    .format(mbps, epoch, n_epochs,  d_x_loss.item() , d_y_loss.item() , g_total_loss.item() ))
           print(' TV-loss: ', tvloss.item(), '  content loss:', contentloss.item(), '  identity loss:', identity_loss.item() )

      fid = calc_fid(G_XtoY.eval())
      G_XtoY.train()
      writer.add_scalar('FID', fid, global_step=epoch)

      losses.append((runningDX_loss/mbps, runningDY_loss/mbps, runningG_loss/mbps))
      print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))


    return losses


n_epochs = 200 # keep this small when testing if a model first works

losses = training_loop(dataloader_X, dataloader_Y, test_iter_X, test_iter_Y, n_epochs=n_epochs)

