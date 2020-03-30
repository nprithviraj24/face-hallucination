

### Face Hallucination
##### This repo is built on grounds of developing different generative models to perform super-resolution (SR) on face images with unaligned pairs between domain X and domain Y. 


1. Image-Degrade: [To learn image super-resolution, use a GAN to learn how to do image degradation first](https://arxiv.org/pdf/1807.11458.pdf)
2. CycleGAN with EDSR. [CycleGAN](https://junyanz.github.io/CycleGAN/) and [EDSR](https://arxiv.org/pdf/1707.02921.pdf)



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
 

- [ ] <strong>StyleVAE: Style basedVAE for Real-World SR</strong> [StyleVAE + SR Network](https://arxiv.org/abs/1912.10227)

** Yet to Come **
### Comparison
```FIDs scores between model and interpolation technique```

<strong> How can we evaluate our model if we don't have groud-truth high resolution images? </strong>

We can have a decent workaround for this problem, by calculating the FID of our generated images with <i>frame of reference</i> dataset such as Celeb-A, FFHQ, AIT3D etc.
Idea is that we calculate FID between the interpolated high-res and <i>frame of reference</i> dataset. We compare this value with FID between synthesized high res image from our model and <i>frame of reference</i> dataset

#### Interpolating based upsampling process from ``PIL``
|   Reference Dataset	|  Input Dataset	| Upsampling Process  	|  FID 
| --- |---	|---	|---	
| celebA  	|   DIV2k	| NEAREST  	|   221.990232219738	
| celebA  	|  DIV2k 	| BILINEAR  |  231.9694814498381
| celebA  	|  DIV2k 	| BICUBIC  	|  252.76578951346124   	
| celebA  	|   DIV2k	| LANCZOS  	|   242.29334933169804	
| celebA  	|  DIV2k 	| HAMMING  	|  221.990232219738
| celebA  	|  DIV2k 	| BOX   	|  199.87696132053242

#### FID of Generated Images

Reference Dataset ```CelebA```

Upsample | Input | FID | Experiment Notes                              
--- | --- | --- | --- |
EDSR* | DIV2k |  315.56615386029983 |  Scale: 4x, lr: 3x16x16, maybe normalization?
CycleGAN (G1: EDSR*) | DIV2k | 252.2 | Epoch:31, D from PatchGAN, SpectralNorm, ForwardCycle
 
<!--  #Uncomment below two to  when model is ready
ImageDegradation | AIT3D | --  
StyleVAE | AIT3D | -- 
-->

*With official pretrained weights. 