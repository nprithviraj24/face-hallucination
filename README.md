

### Face Hallucination
##### This repo is built on grounds of exploring and developing various deep generative models to perform super-resolution (SR) on limited set of unaligned/unpaired face images. 

Following models are part of my Masters' thesis ([pdf](https://www.cs.ait.ac.th/xmlui/handle/123456789/968))
1. [CycleGAN-with-EDSR](https://github.com/nprithviraj24/face-hallucination/tree/master/CycleGAN-with-EDSR)
2. [Image-Degrade](https://github.com/nprithviraj24/face-hallucination/tree/master/image-degrade)
    - [To learn image super-resolution, use a GAN to learn how to do image degradation first](https://arxiv.org/pdf/1807.11458.pdf)
3. AdaIN for degradation
    - Based on Adaptive Instance normalisation from style transfer literature
4. degradeSR
    - Degrade an image using AdaIN and then perform super-resolution.


### Models to be examined in near future:

`` Map two different domains and generate high-quality images.``


- [ ] <strong>CYCADA:</strong> Cycle consistent Adversarial Domain Adaptation
        
     - Adversarial adaptation models: Feature spaces discover domain invariant representation, but are different to visualise and sometimes fail to capture pixel-level and low-level domain shifts.
     - Alignment typically involves minimizing some measure of distance between the source and target distributions such as
            
            - Maximum mean discrepancy
            - Correlation distance
            - Adversarial discriminator accuracy
     This method is only for domain adaptation, however the <strong>conjecture</strong> is that we can do super-resolution along with domain adaptation.

- [ ] <strong>Attribute-Guided Face Generation Using Condition CycleGAN</strong>
 

- [x] <strong>StyleVAE: Style basedVAE for Real-World SR</strong> [StyleVAE + SR Network](https://arxiv.org/abs/1912.10227)

** Yet to Come **
### Comparison
```FIDs scores between model and interpolation technique```

<strong> How can we evaluate our model if we don't have groud-truth high resolution images? </strong>

We can have a decent workaround for this problem, by calculating the FID of our generated images with <i>frame of reference</i> dataset such as Celeb-A, FFHQ, AIT3D etc.
Idea is that we calculate FID between the interpolated high-res and <i>frame of reference</i> dataset. We compare this value with FID between synthesized high res image from our model and <i>frame of reference</i> dataset

#### Interpolating based upsampling process from ``PIL``
|   Reference Dataset	|  Input Dataset	| Upsampling Process  	|  FID 
| --- |---	|---	|---	
| celebA  	|   DIV2k-faces	| NEAREST  	|   221.990232219738	
| celebA  	|  DIV2k-faces 	| BILINEAR  |  231.9694814498381
| celebA  	|  DIV2k-faces 	| BICUBIC  	|  252.76578951346124   	
| celebA  	|   DIV2k-faces	| LANCZOS  	|   242.29334933169804	
| celebA  	|  DIV2k-faces 	| HAMMING  	|  221.990232219738
| celebA  	|  DIV2k-faces 	| BOX   	|  199.87696132053242

#### FID of Generated Images

Reference Dataset ```CelebA``` (from Kaggle)

Upsample | Input | FID | Experiment Notes                              
--- | --- | --- | --- |
EDSR* | DIV2k-faces |  315.56615386029983|  Scale: 4x, lr: 3x16x16,  maybe normalization?
EDSR* | DIV2k-faces |  310.02485583684836|  Scale: 4x, lr: 3x16x16,  Scaled
EDSR* | DIV2k-faces |  307.22565249983984|  Scale: 4x, lr: 3x16x16,  ImageNet normalization
CycleGAN (G1: EDSR*) | DIV2k-faces | 252.2 | Epoch:31, D from PatchGAN, SpectralNorm, ForwardCycle
 
<!--  #Uncomment below two to  when model is ready
ImageDegradation | AIT3D | --  
StyleVAE | AIT3D | -- 
-->

*With official pretrained weights. 