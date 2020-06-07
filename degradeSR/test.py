from models import rcan
import dl2 as dl
import torch
import os, yaml, argparse
import torchvision

weights =("/tmp/face-hallucination/degradeSR/experiments/network_iter_10000.pth")
device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device("cuda")
def bi_scale(factor, inp): return torch.nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)(inp)


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
    self.patch_size = 64

args = Args()
#mmodel
SRNet = rcan.make_model(args)
SRNet.to(device); SRNet.load_state_dict(torch.load( weights ))
SRNet.eval()

config['exp']['batch_size'] = 4
config['exp']['lr_imageSize'] = 32  ## 16 -> 64 (4x)
config['exp']['data_path'] = "/tmp/Datasets/"
config['exp']['lr_datapath'] = "DIV2k/sty"
# config['exp']['lr_datapath'] = "celebA-264/noisy"
# config['exp']['lr_datapath'] = "celebA-264/noisy"

lr, _ = dl.get_data_loader(image_type='hr', exp=config['exp'])

lr_iter = iter(lr)
inp = next(lr_iter)[0].to(device)
hr = SRNet( inp )

out = torch.cat([ bi_scale(4, inp), hr], dim=0)
import datetime
res = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
torchvision.utils.save_image(out, 'test_results/test-celeba1_'+str(res)+'.png', nrow=4, padding=2,
                             normalize=False, range=None, scale_each=False, pad_value=0)
