## BE expert-level gland grading with Neural Networks
This repository contains code for development of a Neural Network that identifies dysplastic areas of interest on H&E 
data tissue samples and BE biopsies to support pathologists in their diagnostic workflow. 

### The Task:
Dysplasia in BE is assessed according to the revised Vienna criteria, which are based on the dysplasia classification in inflammatory bowel disease. Evaluation of cytological and architectural severity and invasion status leads to assignment in
one of the following categories: negative for dysplasia, indefinite for dysplasia (IND), low grade dysplasia (LGD), noninvasive high grade dysplasia (HGD), and invasive neoplasia. Key characteristics used to assess the
dysplasia grade in BE are surface maturation, glandular architecture, and cytonuclear changes. [[1]](#1)


![](images/examples_grading_BE.png)


### Outline of the project:

- [ ] Create a clean archive dataset:
    * Individual directories for each dataset: ASL, Bolero, LANS, RBE.
        * containing all the tiffs and xmls (converted in the same fashion)
        * a csv file with case or biopsy level annotations
    * Remove polygon annotations with <3 coordinates.
      * Gland annotations that include non tissue => mask out non tissue (Otsu)
    * Move data to the research network.
- [ ] Split data for training, evaluation and testing (keep Bolero apart for testing).
- [ ] Train a baseline segmentation pipeline for grading into: NDBE, LGD, HGD.
- [ ] Add data augmentations 
    * Manual
      - HookNet: spatial, color, noise and stain augmentation [[1]](#1). 
      - RaeNet: gamma transform, random flipping, Gaussian blur, affine translation and colour distortion.
    * Trivial Augment: https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html
    * HE Auto augment: https://github.com/DIAGNijmegen/pathology-he-auto-augment
    * Spatial augmentations need to be applied on both image and segmentation
    * Stain-Transforming Cycle-Consistent GAN [[4]](#4).
- [ ] Use a pretrained ImageNet encoder.
- [ ] Include extra classes: stroma, squamous epithelium, intestinal metaplasia (IM).

### Segmentation pipeline for gland grading into: (NDBE, LGD, HGD)
* Networks architectures to consider: UNet, Hooknet [[2]](#2), RAENet [[3]](#3)
* For on fly patch extraction we use: https://github.com/DIAGNijmegen/pathology-whole-slide-data
  * Includes configuration for patch extraction such as batch size, patch size, spacing.
  * Includes different strategies to sample patches from the WSI (balanced, random, slidingwindow).
  * Configuration stored in (`configs/unet_training_config.yml`)
* Evaluation:
  * Quantitative with DICE/F1
  * Qualitative by Sybren

### Datasets 
Below a summary is shown of all the data available for this project. Gland level annotations were provided by Sybren Meijer. 
Case or slide level annotations were not provided yet. The relevant datasets on AMC servers (/data/archief/AMC-data/Barrett/) are listed below:

| Dataset   | Cases | Biopsies | Diagnosis Available | Raters |
|-----------|:-----:|:--------:|:-------------------:|:------:|
| ASL       |  36   |   139    |          ?          |   ?    |
| Bolero    |  51   |   193    |         yes         |   4    |
| LANS      |  34   |   104    |         yes         |        |
| RBE       |  191  |   473    |          ?          ||
| **Total** |  312  |   909    |         N/A         ||


### Results
Table with results...

## References
<a id="1">[1]</a> 
M.J. van der Wel, (2019). 
What makes an expert Barrett’s pathologist? Concordance and pathologist expertise within a digital review panel
PhD thesis, Faculty of Medicine (AMC-UvA), December 2019.

<a id="2">[2]</a> 
Tellez et. al, (2018). 
Whole-Slide Mitosis Detection in H&E Breast Histology Using PHH3 as a Reference to Train Distilled Stain-Invariant Convolutional Networks.
IEEE Transactions on Medical Imaging, Volume 37, Issue 9, September 2018.

<a id="3">[3]</a> 
Rijthoven et. al, (2020). 
HookNet: multi-resolution convolutional neural networks for semantic segmentation in histopathology whole-slide images. 
Medical Image Analysis, Volume 68, February 2021.

<a id="4">[4]</a> 
Patel et. al, (2022). 
RAE-Net: a deep learning system for staging of estrous cycle. 
Proceedings of SPIE, Volume 1203, May 2022.

<a id="5">[5]</a> 
De Bel et. al, (2019). 
Stain-Transforming Cycle-Consistent Generative Adversarial Networks for Improved Segmentation of Renal Histopathology.
Proceedings of The 2nd International Conference on Medical Imaging with Deep Learning, PMLR 102:151-163, 2019.
