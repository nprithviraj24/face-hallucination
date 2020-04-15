from models import rcan
# from models.net import vgg
from models import stylenet
from models import simpleSR
from models.function import adaptive_instance_normalization
import dataloader as dl
from models import Discriminator
# from Discriminators import Spectral_Normalization
# from Discriminators import Discriminator as D
# import loss
# from FID import fid_score as FID
from pytorch_lightning.logging import TestTubeLogger
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

args1 = parser.parse_args()


with open(args1.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

torch.manual_seed(config["logging"]["manual_seed"])
device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device("cuda")

def bi_scale(factor, inp): return nn.Upsample(scale_factor=factor, mode='bilinear')(inp)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = config["exp"]["lr"] / (1.0 + config["exp"]["lr_decay"] * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#### LOGGER

tt_logger = TestTubeLogger(
    save_dir=config['logging']['save_dir'],
    name=config['logging']['name'],
    debug=False,
    create_git_tag=False
)
 #########################################################

class Args():
  def __init__(self):
    super(Args, self).__init__()
    self.model = "RCAN"
    self.save = "RCAN_BIX4_G10R20P48"
    self.scale = [4]
    self.rgb_range=255
    self.n_colors=3
    self.res_scale=1
    # self.scale = list(map(lambda x: int(x), self.scale.split('+')))
    self.n_resgroups=10
    self.n_resblocks=20
    self.n_feats = 64
    self.reduction=16
    self.vgg = '/tmp/models/vgg_normalised.pth'
    self.decoder = '/tmp/models/decoder30k.pth'
    # self.reset
    # self.save_results
    # self.print_model
    self.patch_size = 64
    self.pre_train = "/tmp/pretrainedRCAN/RCAN_BIX4.pt"
args = Args()

style, style_test = dl.get_data_loader(image_type='lr', exp=config['exp'])
content, content_test = dl.get_data_loader(image_type='hr', exp=config['exp'])

## STYLE NETWORK Encoder
vgg = stylenet.vgg
vgg.load_state_dict(torch.load(config['model']['vgg']))
vgg = nn.Sequential(*list(vgg.children())[:31]).to(device)

## STYLENET pretrained decoder
decoder = stylenet.decoder
decoder.load_state_dict(torch.load(args.decoder))
decoder.to(device)
#--------------#

#STYLE NETWORK Trainable Decoder
decoder2 = decoder
decoder2.to(device)
decoder2Optim = torch.optim.Adam( decoder.parameters(),
                                  lr=config['sty']['lr'])
#--------------#

# Super-resolution Network
SRNet = rcan.make_model(args)
SRNet.load_state_dict(torch.load(args.pre_train))
# SRNet = simpleSR.Net(4)
SRNet.to(device)

#-----------------------#
''' Optimizing '''
# SROptim = torch.optim.Adam(SRNet.parameters(),lr= config['SR']['lr'])
L1_Idt = torch.nn.L1Loss()
import torch.optim as optim

# hyperparams for Adam optimizer
lr=0.0002
beta1=0.5
beta2=0.999 # default value

# g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters
g_params = SRNet.parameters()
# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr, [beta1, beta2])
# d_x_optimizer = optim.Adam(D_X.parameters(), lr, [beta1, beta2])
# d_y_optimizer = optim.Adam(D_Y.parameters(), lr, [beta1, beta2])
#-----------------------#

decoder.eval()
decoder2.train()
vgg.eval()

styletransfer = stylenet.Net(vgg, decoder)
style_iter = iter(style)
content_iter = iter(content)

s_test = iter(style_test); c_test = iter(content_test);

alpha = config['sty']['alpha']
dOne = Discriminator.define_D(3, 8, netD='pixel')
dOne.to(device)

# config['exp']['max_iter'] = 1
for i in tqdm(range(config['exp']['max_iter'] )):
    print("\n")
    adjust_learning_rate(g_optimizer, iteration_count=i)
    # adjust_learning_rate(decoder2Optim, iteration_count=i)

    content_images = next(content_iter)[0].to(device)
    style_images = next(style_iter)[0].to(device)
    with torch.no_grad():
        sty_ft = vgg(style_images)
        cont_ft = vgg(content_images)
        feat = adaptive_instance_normalization(content_feat=cont_ft, style_feat=sty_ft)

        feat = feat * alpha + cont_ft * (1 - alpha)
        cont_degrade = decoder(feat)
        del feat, sty_ft, cont_ft

    g_optimizer.zero_grad()
    content_images.detach(); style_images.detach()
    hr = SRNet( bi_scale(0.25, cont_degrade) )
    hr_IDT_loss = config['SR']['loss_weight'] * L1_Idt(content_images, hr)
    hr_IDT_loss.backward()
    g_optimizer.step()

    tt_logger.experiment.add_scalar('IDT_Loss', hr_IDT_loss, global_step=i)

    ##Logging
    if (i+1) % config['logging']['im_save_interval'] == 0:
        # print("\n .")
        out=torch.cat([content_images, bi_scale(4, style_images),cont_degrade, hr], dim=0)
        # tt_logger.experiment.add_image(config['logging']['name'],
        vutils.save_image(out, config['logging']['model_save_dir']+'/samples2/'+str(i)+'.png')

    if (i+1) % config['logging']['model_save_interval'] == 0 or (i + 5) == config['exp']['max_iter']:
            # print("\n ..")
            state_dict = SRNet.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, config["logging"]["model_save_dir"]+
                       str('network_iter_')+str(int(i+1))+str('.pth'))
    # vutils.save_image(out, 'xyz.png')
    # print("\n\n  OUT: ", out.shape)


