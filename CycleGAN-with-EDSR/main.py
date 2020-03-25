import dataloader as dl


# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
# del dataloader_X, test_dataloader_X
# del dataloader_Y, test_dataloader_Y

dataloader_X, test_iter_X = dl.get_data_loader(image_type='lr')
dataloader_Y, test_iter_Y = dl.get_data_loader(image_type='hr')

# next(iter(dataloader_X))[0][0]

rgb_range = 255
n_colors = 3
n_feats = 256 #initially 256
n_resblocks = 32
res_scale= 0.1
kernel_size = 3
scale = 4


url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)

url = {
    'r16f64x2': 'EDSR_Weights/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'EDSR_Weights/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'EDSR_Weights/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'EDSR_Weights/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'EDSR_Weights/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'EDSR_Weights/edsr_x4-4f62e9ef.pt'
}