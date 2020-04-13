from models import rdb
import torchvision.models as models
import os
from torch import nn


os.environ['http_proxy'] = 'http://192.41.170.23:3128'
os.environ['https_proxy'] = 'http://192.41.170.23:3128'

densenet = models.densenet161(pretrained=True)
net = nn.Sequential(*list(densenet.children())[:31])
print(net)