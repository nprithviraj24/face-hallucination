import argparse
from pathlib import Path
import dataloader as dl
import numpy as np
import torch
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
# import loss
from torchvision import transforms
from tqdm import tqdm
import yaml
from net import Net

from models.vgg import vgg
from pytorch_lightning.logging import TestTubeLogger

device = torch.device('cuda')


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

# For reproducibility
torch.manual_seed(config['logging']['manual_seed'])
np.random.seed(config['logging']['manual_seed'])

style, style_test = dl.get_data_loader(image_type='lr', exp=config['exp'])
content, content_test = dl.get_data_loader(image_type='hr', exp=config['exp'])

def loss(inp, out, mu, logvar):
    al_kld = config['loss']['kld_alpha']
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    al_rec = config['loss']['rec_alpha']
    mse = al_rec * nn.MSELoss()(inp, out)

    return al_kld*KLD + mse


vgg.load_state_dict(torch.load(config['model']['vgg']))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = Net(vgg)
network.train()
network.to(device)

style_iter = iter(style)
content_iter = iter(content)

s_test = iter(style_test); c_test = iter(content_test);

optimizer = torch.optim.Adam(network.parameters(),
                             lr= config['exp']['lr'])

# config['exp']['max_iter'] = 2
for i in tqdm(range(  config['exp']['max_iter'] )):
    adjust_learning_rate(optimizer, iteration_count=i)

    content_images = next(content_iter)[0].to(device)
    # print(content_images[1].shape ) ##  why torch[[64]]???
    style_images = next(style_iter)[0].to(device)
    out, mu, logvar = network(content_images, style_images)
    cost = loss(content_images, out, mu, logvar)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    tt_logger.experiment.add_scalar('loss', cost, i + 1)
    # tt_logger.experiment.add_scalar('loss_style', loss_s.item(), i + 1)

    if (i+1) %  config["logging"]["image_save"] :
        tt_logger.experiment.add_image(str(i),
                                       vutils.make_grid(out[0:8, :, :, :], nrow=8, padding=1, normalize=True),
                                       global_step=i)
    if (i + 1) % config["logging"]["save_model_interval"] == 0 or \
            (i + 1) == config["exp"]["max_iter"]:

            state_dict = network.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, config["model"]["save_dir"]+
                       str('network_iter_')+str(int(i+1))+str('.pth'))

tt_logger.experiment.close()
#
