import torch, torchvision
from pytorch_lightning.logging import TestTubeLogger
import argparse, time
import dataloader
import utilities as utility
from High2Low import generator as G_H2L
from High2Low import discriminator as D_H2L
from torch import optim, nn
from Low2High import GEN_DEEP as G_L2H
from Low2High import discriminator as D_L2H
from tqdm import tqdm, trange
import lossFunctions as LossF

log_flag=True

torch.manual_seed(499)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)  ## LEARNING RATE
parser.add_argument('--save_dir', type=str, default='experiments/')
parser.add_argument('--name', type=str, default='no_noiseV')

parser.add_argument('--pixelWeight', type=float, default=0.75)
parser.add_argument('--ganWeight', type=float, default=0.45)
parser.add_argument('--cycleWeight', type=float, default=0.50)

parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--image_dirLR', type=str, default="/tmp/Datasets/DIV2k/images")
parser.add_argument('--image_dirHR', type=str, default="/tmp/Datasets/3Dto2D/squared/variance")
parser.add_argument('--lr_imageSize', type=str, default=32)
parser.add_argument('--hr_imageSize', type=str, default=128)

# parser.add_argument('--loss', type=str, default='hinge')
# parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
args = parser.parse_args()

#### LOGGER

tt_logger = TestTubeLogger(
    save_dir=args.save_dir,
    name=args.name,
    debug=False,
    create_git_tag=False
)
 #########################################################

## ------------- DATALOADERS -------------##
high_res_loader, high_res_loader_Test, HRimages = dataloader.get_data_loader(image_type='hr', exp=args)
low_res_loader, low_res_loader_Test, LRImages = dataloader.get_data_loader(image_type='lr', exp=args)
# print(len(high_res_loader))
# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        m.bias.data.fill_(0.01)

downsampleG = G_H2L.Generator(device=None); downsampleG.cuda()
downsampleG.apply(init_weights)

lowresD = D_H2L.Discriminator(); lowresD.cuda()
# lowresD.apply(init_weights)

highresD = D_L2H.Discriminator(); highresD.cuda()
# highresD.apply(init_weights)

upsampleG = ; upsampleG.cuda()

# upsampleG.load_state_dict(torch.load('model.pkl'))

x=0
for child in upsampleG.children():
    x = x+1
    if x < 5:
        for param in child.parameters():
            param.requires_grad = False

# print(x)
# '''
# -----------HYPERPARAMETERS FOR OPTIMIZATION---------

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_lowresD = optim.Adam(filter(lambda p: p.requires_grad, lowresD.parameters()), lr=args.lr, betas=(0.0, 0.9))
optim_highresD = optim.Adam(filter(lambda p: p.requires_grad, highresD.parameters()), lr=args.lr, betas=(0.0, 0.9))
g_params = list(downsampleG.parameters()) + list(upsampleG.parameters())
optim_gen = optim.Adam(g_params, lr=args.lr, betas=(0.0, 0.9))
optim_genL2H = optim.Adam(upsampleG.parameters(), lr=args.lr, betas=(0.0, 0.9))
optim_genH2L = optim.Adam(downsampleG.parameters(), lr=args.lr, betas=(0.0, 0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_lowresD, gamma=0.99)
scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_highresD, gamma=0.99)

scheduler_g1 = optim.lr_scheduler.ExponentialLR(optim_genH2L, gamma=0.99)
scheduler_g2 = optim.lr_scheduler.ExponentialLR(optim_genL2H, gamma=0.99)
# scheduler_d = optim.lr_scheduler.ExponentialLR(, gamma=0.99)

# pretrain epochs
pretrain_epoch = 0

# labels
real = 1
fake = 0
# number of updates to discriminator for every update to generator
disc_iters = 3

def bce(real, fake): return nn.BCELoss()( real, fake).mean()


def training_loop(high_res_loader, low_res_loader, test_high_res_loader, test_low_res_loader,
                  n_epochs=200):
    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_highres = iter(test_high_res_loader).next()[0].cuda()
    fixed_lowres = iter(test_low_res_loader).next()[0].cuda()


    # batches per epoch
    global_step = 0
    for epoch in trange(pretrain_epoch, n_epochs + 1):
        d1_epoch=0; d2_epoch=0;
        downsample_epoch =0; upsample_epoch=0; cycle_epoch=0

        for batch_id in trange(int(HRimages/args.batch_size)):
            #  with torch.no_grad():
            global_step = global_step + 1
            high_res = next(iter(high_res_loader))[0].cuda()
            low_res= next(iter(low_res_loader))[0].cuda()
            # move images to GPU if available (otherwise stay on CPU)

            # disc_iter:1 ratio
            # discriminator updates
            disc1 = 0; disc2 = 0
            for _ in range(disc_iters):

                with torch.no_grad():
                    fLR = downsampleG(high_res)
                    fHR = upsampleG(low_res)

                real_label = torch.full((args.batch_size,), real);
                fake_label = real_label.fill_(fake);
                optim_lowresD.zero_grad()
                # lowresD_loss = bce(lowresD(high_res.cuda()), real_label.cuda()) + \
                #                bce(lowresD(downsampleG(low_res.cuda())),fake_label.cuda())
                lowresD_loss =  lowresD(low_res).mean() - lowresD(fLR).mean()
                lowresD_loss.backward()
                optim_lowresD.step()
                disc1 += lowresD_loss

                optim_highresD.zero_grad()
                # highresD_loss = bce(highresD(low_res), real_label.cuda()) + bce(highresD(upsampleG(high_res)), fake_label.cuda())

                highresD_loss =  (highresD(high_res).mean() - highresD(fHR).mean())
                highresD_loss.backward()
                optim_highresD.step()
                disc2 += highresD_loss

            d1_epoch += disc1 ;  d2_epoch  += disc2

            # generator updates

            # high to low
            optim_gen.zero_grad()
            optim_genH2L.zero_grad()

            fake_lowres = downsampleG(high_res)
            # print("fakeY ::: ", fakeY.shape)
            fakeLR_d = lowresD(fake_lowres)
            gan1Loss = LossF.GANloss(lowresD(low_res), fakeLR_d)  # real, fake
            pixel1Loss = LossF.pixelLoss(utility.downsample4x(high_res), fake_lowres)
            lossH2L = args.ganWeight * gan1Loss + args.pixelWeight * pixel1Loss
            downsample_epoch += lossH2L
            lossH2L.backward(retain_graph=True)

            # low to high
            fake_highres = upsampleG(fake_lowres)
            fakeHR_d = highresD(fake_highres)
            gan2Loss = LossF.GANloss(highresD(high_res), fakeHR_d)
            pixel2Loss = LossF.pixelLoss(high_res, fake_highres)
            lossL2H = args.ganWeight * gan2Loss + 2 * pixel2Loss
            upsample_epoch += lossL2H
            lossL2H.backward(retain_graph=True)

            #Cycle
            cycle = args.cycleWeight * (lossH2L + lossL2H)
            cycle.backward()
            optim_gen.step()

            if log_flag:
                tt_logger.experiment.add_scalar('D/D1', disc1/disc_iters, global_step=global_step)
                tt_logger.experiment.add_scalar('D/D2', disc2/disc_iters, global_step=global_step)

                tt_logger.experiment.add_scalar('G/G1/GAN', args.ganWeight * gan1Loss, global_step=global_step)
                tt_logger.experiment.add_scalar('G/G1/Pixel', args.pixelWeight * pixel1Loss, global_step=global_step)

                tt_logger.experiment.add_scalar('G/G2/Gan', args.ganWeight * gan2Loss, global_step=global_step)
                tt_logger.experiment.add_scalar('G/G2/Pixel', 2 * pixel2Loss, global_step=global_step)

                tt_logger.experiment.add_scalar('G/Cycle', cycle, global_step=global_step)

            cycle_epoch += cycle
        with torch.no_grad():
            downsampleG.eval(); upsampleG.eval()  # set generators to eval mode for sample generation
            degradedHR = downsampleG(fixed_highres)
            rectifiedHR = upsampleG(degradedHR)
            out = torch.cat([fixed_highres[0:8], utility.upsample(4, degradedHR)[0:8], rectifiedHR[0:8]], dim=0)
            # print("shapes: ", fixed_highres.shape, degradedHR.shape, fixed_highres.shape, fixed_lowres.shape)
            if log_flag: tt_logger.experiment.add_image('Forward/Epoch_'+str(epoch),
                                           torchvision.utils.make_grid(out, nrow=8)
                                           ,global_step=epoch)
            downsampleG.train(); upsampleG.train()

                    #   # set generators to eval mode for sample generation
                    # fakeX = upsampleG(fakeY.cuda())
                    # # utility.imshow(torchvision.utils.make_grid(fakeX.cpu()))
                    # upsampleG.train()
                    # print('high->low: {:6.4f},  low->high: {:6.4f}'.format(lg / bn, hg / bn))
        if log_flag:
            tt_logger.experiment.add_scalar('Epoch/D2', d2_epoch, global_step=epoch)
            tt_logger.experiment.add_scalar('Epoch/G1', downsample_epoch, global_step=epoch)
            tt_logger.experiment.add_scalar('Epoch/D1', d1_epoch, global_step=epoch)
            tt_logger.experiment.add_scalar('Epoch/G2', upsample_epoch, global_step=epoch)
            tt_logger.experiment.add_scalar('Epoch/Cycle', cycle_epoch, global_step=epoch)

        if (epoch + 1) % 25 == 0 or (epoch + 1) == args.n_epochs:
            state_dict = downsampleG.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict,  'saved_models/'+
                       str('apr27downsampleG_iter_') + str(int(epoch + 1)) + str('.pth.tar'))

        scheduler_d.step()
        scheduler_g.step()
        time.sleep(0.01)
    return
    # return losses


# keep epochs small when testing if a model first works
training_loop(high_res_loader, low_res_loader, high_res_loader_Test,  low_res_loader_Test, n_epochs=args.n_epochs)
