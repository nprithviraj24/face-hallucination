import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
try:
    import fid_score as FID
except:
    pass

#are the images saved already?
interpolation_done = True
lr_path = "/tmp/Datasets/DIV2k"

#Comment the line which you dont want to calculate.
_pil_interpolation_to_str = {
    'NEAREST': 0, #'PIL.Image.NEAREST',
    'BILINEAR': 1, #'PIL.Image.BILINEAR',
    'BICUBIC': 2, #'PIL.Image.BICUBIC',
    'LANCZOS': 3, #'PIL.Image.LANCZOS',
    'HAMMING': 4, #'PIL.Image.HAMMING',
    'BOX': 5, #'PIL.Image.BOX',
}

# reference_frame = '/tmp/Datasets/celeba/img_align_celeba/celeba'
reference_frame = '/tmp/Datasets/3Dto2D/squared/variance/590'

for key, value in _pil_interpolation_to_str.items():

    ##Numpy read directory. Directory with images should be mentioned.
    lr_folder = "/tmp/Datasets/div2k-interpolation/"+key

    if not interpolation_done:
        transFORM = transforms.Compose([
            transforms.Resize([64,64], interpolation=_pil_interpolation_to_str[key] ),
            transforms.ToTensor()
        ])

        images = datasets.ImageFolder(lr_path, transFORM)
        images = DataLoader(dataset=images, batch_size=1)
        for batch_id, (x, _) in tqdm(enumerate(images), total=len(images)):

            vutils.save_image(
                x, #tensor
                lr_folder+"/"+str(batch_id)+".png"
                )
    score = FID.calculate_fid_given_paths(
        [reference_frame, lr_folder],  # paths
        8,  # batch size
        True,  # cuda
        2048  # dims/tmp/Datasets/3Dto2D/squared/variance/590
    )
    # print("Reference frame: "+reference_frame)
    print("\n\n"+key+": "+str(score))