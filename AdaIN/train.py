import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.logging import TestTubeLogger
import torchvision.utils as vutils
from torchvision import transforms
from tqdm import tqdm

import net
from sampler import InfiniteSamplerWrapper

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numbers
import numpy as np
from PIL import ImageFilter

class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def HRtrain_transform():
    transform_list = [
        # GaussianSmoothing([0,1.3]),
        # transforms.Resize(size=(256, 256)),
        # transforms.CenterCrop(150),
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        # AddGaussianNoise(0, 0.001)
    ]
    return transforms.Compose(transform_list)

def LRtrain_transform():
    transform_list = [
        transforms.Resize(size=(128, 128)),
        # transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)
#
# def trainCelebA_transform():
#     transform_list = [
#         transforms.Resize(size=(256, 256)),
#         transforms.CenterCrop(128),
#         transforms.Resize(size=(64, 64)),
#         transforms.ToTensor()
#     ]
#     return transforms.Compose(transform_list)



class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'


def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options

parser.add_argument('--content_dir', type=str,
                    default='/tmp/Datasets/celeba/img_align_celeba/celeba',
                    # default='/tmp/Datasets/3Dto2D/squared/variance/590',
                    help='Directory path to a batch of content images')

parser.add_argument('--style_dir', type=str,
                    # default='/tmp/Datasets/celebA-264/lr',
                    default='/tmp/Datasets/DIV2k/images/div2k',
                    help='Directory path to a batch of style images')

parser.add_argument('--vgg', type=str, default='/tmp/models/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments/div2k-celeb_20-4',
                    help='Directory to save the model')

# parser.add_argument('--log_dir', default='./logs/celeb',
#                     help='Directory to save the log')

parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=2.5)
parser.add_argument('--content_weight', type=float, default=0.75)

parser.add_argument('--n_threads', type=int, default=0)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

device = torch.device('cuda')
save_dir = Path(args.save_dir)
save_dir.mkdir(exist_ok=True, parents=True)
# log_dir = Path(args.log_dir)
# log_dir.mkdir(exist_ok=True, parents=True)
# writer = SummaryWriter(log_dir=str(log_dir))
config_name='random-alpha'

tt_logger = TestTubeLogger(
    save_dir=args.save_dir,
    name=config_name,
    debug=False,
    create_git_tag=False
)

### load deocer weights too
decoder = net.decoder

vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)
network.train()
network.to(device)


''' DATALOADERs '''
content_tf = HRtrain_transform()
# celebtf = trainCelebA_transform()
style_tf = LRtrain_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
'''-----------------'''

optimizer = torch.optim.Adam(network.decoder.parameters(), lr=args.lr)

import random
for i in tqdm(range(args.max_iter)):

    alpha = random.uniform(0,1)

    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    out, loss_c, loss_s = network(content_images, style_images, alpha= alpha)
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # out = torch.cat([content_images[0:7], style_images[0:7], out[0:7]])
    tt_logger.experiment.add_scalar('loss_content', loss_c.item(), i + 1)
    tt_logger.experiment.add_scalar('loss_style', loss_s.item(), i + 1)
    tt_logger.experiment.add_scalar('alpha', alpha, i+1)

    if (i+1) % 500== 0 or i==1:
        out = torch.cat([content_images[0:8], style_images[0:8], out[0:8]])
        tt_logger.experiment.add_image(str(i)+'/alpha_'+str(alpha),
                         vutils.make_grid(out, nrow=8, padding=1, normalize=True),
                                       global_step=i)


    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = net.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, save_dir /
                   'randomAlpha_decoder_iter_{:d}.pth.tar'.format(i + 1))
writer.close()
