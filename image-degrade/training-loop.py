
import time
import pylab as pl
from IPython import display
from torch import nn



def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  n_epochs=1000):

    print_every=10

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
           D_X_real_loss = real_mse_loss(out_x)
           fake_X = G_YtoX(images_Y)
           out_x = D_X(fake_X)   
           D_X_fake_loss = fake_mse_loss(out_x)
           d_x_loss = D_X_real_loss + D_X_fake_loss
           d_x_loss.backward()
           d_x_optimizer.step()
           d_x_loss.detach(); out_x.detach(); D_X_fake_loss.detach();
           runningDX_loss += d_x_loss 
           del D_X_fake_loss, D_X_real_loss, out_x, fake_X , d_x_loss
           torch.cuda.empty_cache()
           
          #  print("end: DX block  and start DY", convert_size(torch.cuda.memory_allocated(device=device)))

           d_y_optimizer.zero_grad()
           out_y = D_Y(images_Y)
           D_Y_real_loss = real_mse_loss(out_y)
           fake_Y = G_XtoY(images_X)
           out_y = D_Y(fake_Y)
           D_Y_fake_loss = fake_mse_loss(out_y)
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
           g_XtoY_loss = real_mse_loss(out_y)
           reconstructed_X = G_YtoX(fake_Y)
           
           reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=50)
           
           featuresY = loss_network(images_Y);           
           featuresFakeY = loss_network(fake_Y); 
           
           CONTENT_WEIGHT = 0.00001          
           contentloss = CONTENT_WEIGHT * mse_loss(featuresY[1].data, featuresFakeY[1].data)
           del featuresY, featuresFakeY; torch.cuda.empty_cache()
           
           IDENTITY_WEIGHT = 0.0001
           downsample = nn.Upsample(scale_factor=0.25, mode='bicubic')
           identity_loss = IDENTITY_WEIGHT * mse_loss(downsample(fake_Y), images_X )

           TOTAL_VARIATION_WEIGHT = 0.0001
           tvloss = TOTAL_VARIATION_WEIGHT + tv_loss(fake_Y, 0.5)
           
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
              imshow(torchvision.utils.make_grid(fixed_X.cpu()))
              imshow(torchvision.utils.make_grid(fakeY.cpu()))
              G_XtoY.train()          
              print('Mini-batch no: {}, at epoch [{:3d}/{:3d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(mbps, epoch, n_epochs,  runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))

      with torch.no_grad():
        G_XtoY.eval() # set generators to eval mode for sample generation
        fakeY = G_XtoY(fixed_X.to(device))
        imshow(torchvision.utils.make_grid(fixed_X.cpu()))
        imshow(torchvision.utils.make_grid(fakeY.cpu()))
        G_XtoY.train()
        # print("Epoch loss:  ", epochG_loss/)
      losses.append((runningDX_loss/mbps, runningDY_loss/mbps, runningG_loss/mbps))
      print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(epoch, n_epochs, runningDX_loss/mbps ,  runningDY_loss/mbps,  runningG_loss/mbps ))
              
      
    return losses