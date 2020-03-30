from Generators import EDSR as G1
import loss
from FID import fid_score as FID

# from test_tube import Experiment
import os, yaml, argparse
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from tqdm import tqdm
import torch
from torch import nn



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
    # print(G_XtoY.url)
    # kwargs['EDSR']
    G_XtoY.load_state_dict(torch.load(url[G_XtoY.url]))

    return G_XtoY

# Create train and test dataloaders for div2k from the two domains X and Y

G_XtoY = create_model(device=device, EDSR=config['EDSR'])

### This needs complete path, i.e needs folder name where images are save. Numpy processes this.
reference_frame = os.path.join(config['exp_params']['data_path'],
                               config['exp_params']['hr_datapath'],
                                 'celeba')

transFORM = transforms.Compose([
    transforms.Resize([16, 16], interpolation=3),
    transforms.ToTensor()
])

div2k = datasets.ImageFolder(os.path.join(config['exp_params']['data_path'],
                                          config['exp_params']['lr_datapath']), transFORM)
div2k = DataLoader(dataset=div2k, batch_size=1)

def calc_fid():
    G_XtoY.eval()
    for batch_id, (x, _) in tqdm(enumerate(div2k), total=len(div2k)):

        vutils.save_image(
            G_XtoY(x.to(device)),  # tensor
            config['EDSR']['edsr_save'] + "/" + str(batch_id) + ".png"
        )
    return FID.calculate_fid_given_paths( [reference_frame, config['EDSR']['edsr_save']],  # paths
          100,  # batch size
          True,  # cuda
          2048 ) # dims

print(calc_fid())