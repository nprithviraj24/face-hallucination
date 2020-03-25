device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)  ## LEARNING RATE
parser.add_argument('--tensorboard', type=str, default='runs/')
parser.add_argument('--pixelWeight', type=float, default=1.00)
parser.add_argument('--ganWeight', type=float, default=0.05)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--image_dir', type=str)
# parser.add_argument('--loss', type=str, default='hinge')
# parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
args = parser.parse_args()

## ------------- DATALOADERS -------------##
xTrain, xTest = dataloader.get_data_loader(image_type='hr', image_dir=args.image_dir, batch_size=args.batch_size)
yTrain, yTest = dataloader.get_data_loader(image_type='lr', image_dir=args.image_dir, batch_size=args.batch_size)

# discriminator = torch.nn.DataParallel(Discriminator()).cuda() # TODO: try out multi-gpu training

h2l_g = G_H2L.Generator(device=None).to(device)
h2l_d = D_H2L.Discriminator().to(device)

l2h_d = D_L2H.Discriminator().to(device)
l2h_g = G_L2H.GEN_DEEP().to(device)
l2h_g.load_state_dict(torch.load('model.pkl'))
# net_G_low2high = net_G_low2high.eval()

# -----------HYPERPARAMETERS FOR OPTIMIZATION---------

# because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to 
# optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
# TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
optim_discH2L = optim.Adam(filter(lambda p: p.requires_grad, h2l_d.parameters()), lr=args.lr, betas=(0.0, 0.9))
optim_discL2H = optim.Adam(filter(lambda p: p.requires_grad, l2h_d.parameters()), lr=args.lr, betas=(0.0, 0.9))

optim_genH2L = optim.Adam(h2l_g.parameters(), lr=args.lr, betas=(0.0, 0.9))
optim_genL2H = optim.Adam(l2h_g.parameters(), lr=args.lr, betas=(0.0, 0.9))

# use an exponentially decaying learning rate
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_discH2L, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_genH2L, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_discL2H, gamma=0.99)
scheduler_d = optim.lr_scheduler.ExponentialLR(optim_genL2H, gamma=0.99)

# pretrain epochs
pretrain_epoch = 0

# labels
real = 1
fake = 0
# number of updates to discriminator for every update to generator
disc_iters = 5


def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=200):
    print_every = 10

    # keep track of losses over time
    losses = []

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    # make sure to scale to a range -1 to 1
    # fixed_X = scale(fixed_X) 
    # fixed_Y = scale(fixed_Y)

    # batches per epoch

    # n_epochs = 2
    for epoch in range(pretrain_epoch, n_epochs + 1):

        lowres = 0
        highres = 0
        lowresD = 0
        highresD = 0

        bn = 0  # mini batches per epoch
        firstbn = True
        for batch_id, (images_X, _) in tqdm_notebook(enumerate(dataloader_X), total=len(dataloader_X)):
            #  with torch.no_grad():
            bn += 1
            images_Y, a = next(iter(dataloader_Y))
            # move images to GPU if available (otherwise stay on CPU)

            images_X = images_X.to(device)
            images_Y = images_Y.to(device)

            # 5:1 ratio
            # discriminator updates
            d1 = 0
            d2 = 0
            for _ in range(disc_iters):
                real_label = torch.full((args.batch_size,), real, device=device)
                fake_label = real_label.fill_(fake)

                optim_discH2L.zero_grad()
                discH2L_loss = nn.BCELoss()(h2l_d(images_Y), real_label).mean() + nn.BCELoss()(h2l_d(h2l_g(images_X)),
                                                                                               fake_label).mean()
                discH2L_loss.backward()
                optim_discH2L.step()
                d1 += discH2L_loss

                optim_discL2H.zero_grad()
                discL2H_loss = nn.BCELoss()(l2h_d(images_X), real_label).mean() + nn.BCELoss()(l2h_d(l2h_g(images_Y)),
                                                                                               fake_label).mean()
                discL2H_loss.backward()
                optim_discL2H.step()
                d2 += discL2H_loss

            lowresD += d1 / disc_iters
            highresD += d2 / disc_iters

            # generator updates

            # high to low
            optim_genH2L.zero_grad()

            fakeY = h2l_g(images_X)
            fakeY_d = h2l_d(fakeY)
            ganLoss = LossF.GANloss(h2l_d(images_Y), fakeY_d)  # real, fake
            pixelLoss = LossF.pixelLoss(images_X, utility.downsample4x(fakeY))

            lossH2L = args.ganWeight * ganLoss + args.pixelWeight * pixelLoss
            lowres += lossH2L
            lossH2L.backward()
            optim_genH2L.step()
            del fakeY_d

            # low to high
            optim_genL2H.zero_grad()

            fakeX = l2h_g(fakeY)
            fakeX_d = l2h_d(fakeX)
            ganLoss = LossF.GANloss(l2h_d(images_X), fakeX_d)
            pixelLoss = LossF.pixelLoss(images_X, fakeX)

            lossL2H = args.ganWeight * ganLoss + args.pixelWeight * pixelLoss
            highres += lossL2H

            lossH2L.backward()
            optim_genH2L.step()
            del fakeX_d, fakeX, fakeY

            if bn % LOG_INTERVAL == 0:
                if firstbn:
                    ld = lowresD;
                    hd = highresD;
                    lg = lowres;
                    hg = highres
                    firstbn = False
                else:
                    ld = lowresD - ld;
                    hd = highresD;
                    lg = lowres;
                    hg = highres

                print('Mini-batch no: {}, at epoch [{:3d}/{:3d}] | D-low: {:6.4f} | D-high: {:6.4f}'.format(bn, epoch,
                                                                                                            n_epochs,
                                                                                                            ld / bn,
                                                                                                            hd / bn),
                      end=' ')
                with torch.no_grad():
                    h2l_g.eval()  # set generators to eval mode for sample generation
                    fakeY = h2l_g(fixed_X.to(device))
                    utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
                    # utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
                    h2l_g.train()

                    l2h_g.eval()  # set generators to eval mode for sample generation
                    fakeX = l2h_g(fakeY.to(device))
                    utility.imshow(torchvision.utils.make_grid(fakeX.cpu()))
                    l2h_g.train()
                    print('high->low: {:6.4f},  low->high: {:6.4f}'.format(lg / bn, hg / bn))

        with torch.no_grad():
            h2l_g.eval()  # set generators to eval mode for sample generation
            fakeY = h2l_g(fixed_X.to(device))
            utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
            # utility.imshow(torchvision.utils.make_grid(fakeY.cpu()))
            h2l_g.train()

            l2h_g.eval()  # set generators to eval mode for sample generation
            fakeX = l2h_g(fakeY.to(device))
            utility.imshow(torchvision.utils.make_grid(fakeX.cpu()))
            l2h_g.train()
            # print("Epoch loss:  ", epochG_loss/)
        losses.append((lowresD / bn, highresD / bn, lowres / bn, highres / D))
        print('Epoch [{:5d}/{:5d}] | D-low: {:6.4f} | D-high: {:6.4f} | low->high: {:6.4f} | high->low: {:6.4f}'.format(
            epoch, n_epochs, lowresD / bn, highresD / bn, lowres / bn, highres / bn))
        scheduler_d.step()
        scheduler_g.step()

    return losses


# keep epochs small when testing if a model first works

losses = training_loop(xTrain, yTrain, xTest, yTest, n_epochs=args.n_epochs)
