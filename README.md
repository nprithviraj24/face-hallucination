

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
