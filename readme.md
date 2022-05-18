## BE expert-level gland grading with Neural Networks
This repository contains code for development of a Neural Network that support pathologists with identifying dysplastic 
areas of interest on H&E tissue samples from patients with Barrett's Esophagus (BE). 

### The Task:
BE is associated with an increased risk of developing esophageal cancer. Regular check-ups and pathological assessment of biopsy material are crucial for identifying BE patients at risk.
Dysplasia in BE is assessed according to the revised Vienna criteria, which are based on the dysplasia classification in inflammatory bowel disease. Evaluation of cytological and architectural severity and invasion status leads to assignment in
one of the following categories: non-dysplastic Barrett's esophagus (NDBE), indefinite for dysplasia (IND), low grade dysplasia (LGD) and high grade dysplasia (HGD). Key characteristics used to assess the
dysplasia grade in BE are surface maturation, glandular architecture, and cytonuclear changes. Description and figure from [[1]](#1).


![](images/examples_grading_BE.png)


### Outline of the project:

- [x] Create a clean archive dataset:
    * Individual directories for each dataset: ASL, Bolero, LANS and RBE.
        * Containing all the image files (.tiff converted in the same fashion) and annotation (.xml) files.
        * A csv file with case or biopsy level diagnosis.
    * Remove polygon annotations with <3 coordinates.
    * Gland annotations that include non tissue => mask out non tissue.
  
    **Comments**: 
     * Working with decent clean data on the server for now (/data/archief/AMC-data/Barrett/). Clean up of the original archive on (L:) will be done together with Onno.
     * Masking out non tissue for loose segmentations around the border didn't work out yet with Otsu and morphology. We can also do it manually.
- [x] Split data for training, evaluation and testing. We use Bolero as internal test set.
- [x] Train a standard UNet as baseline segmentation pipeline for grading into: NDBE vs Dysplasia (LGD and HGD).

    **Comments**:
  * Results look good (spacing: 1, 512x512). Dice on validation: weighted: ~0.87-0.88, seperate: 0.93, 0.78, 0.83.
  * Produces mixed glands. We might want to postprocess and assign the most common prediction to the whole gland.
  * Not tested on Bolero yet.
- [ ] Data augmentations for segmentation in pathology:
    * Spatial augmentations need to be applied on both image and segmentation.
    * Manual:
      - HookNet: spatial, color, noise and stain augmentation [[2]](#2). 
      - RaeNet: gamma transform, random flipping, Gaussian blur, affine translation and colour distortion.
    * Stain-Transforming Cycle-Consistent GAN [[5]](#5).
    * Trivial Augment: https://pytorch.org/vision/main/generated/torchvision.transforms.TrivialAugmentWide.html.
    * HE Auto augment: https://github.com/DIAGNijmegen/pathology-he-auto-augment.
- [ ] Visualization and evaluation on slide level, preferably in a notebook.
- [ ] Experiments:
  * Find well working combinations of batch size, patch size and spacing so that it fits on GPU.
  * Context aggregation networks for segmentation in pathology: HistNet [[7]](#7), HookNet [[3]](#3), RAENet [[4]](#4).
  * ImageNet pretrained encoder.
  * Roto-Translation Equivariant CNN's [[6]](#6).
  
### Segmentation pipeline for gland grading into: NDBE, LGD or HGD
* Networks architectures to consider next: HistNet [[7]](#7), HookNet [[3]](#3), RAENet [[4]](#4).
* For on fly patch extraction we use: https://github.com/DIAGNijmegen/pathology-whole-slide-data.
  * Includes configuration for patch extraction such as batch size, patch size, spacing.
  * Includes different strategies to sample patches from the WSI (balanced, random, slidingwindow).
  * Configuration stored in (`configs/unet_training_config.yml`).

### Datasets 
Below a summary is shown of all the data available for this project. Gland level annotations were provided by Sybren Meijer for the categories: NDBE, LGD and HGD.
Case or slide level annotations were not provided yet. The relevant datasets on AMC servers (/data/archief/AMC-data/Barrett/) are listed below:

| Dataset   | Cases | Biopsies | Diagnosis Available<br/>(biopsy/case level) | Raters<br/>(biopsy/case level) |
|-----------|:-----:|:--------:|:-------------------------------------------:|:------------------------------:|
| ASL       |  36   |   139    |                      ?                      |               ?                |
| Bolero    |  51   |   193    |                  should be                  |               4                |
| LANS      |  34   |   104    |                  should be                  |               14               |
| RBE       |  212  |   534    |                      ?                      |               ?                |
| **Total** |  312  |   970    |                     N/A                     |              N/A               |


### Results
Evaluation:
  * Quantitative with DICE/F1 compared to the ground truth annotations.
  * Qualitative assessment by Sybren.

#### Preliminary Testing
We should perform some of these experiments on an internal test set (Bolero).

| Method | Batch Size | Patch Size | Spacing <br/> (mpp) |        Validation Dice        | Test <br/> (Bolero) |
|:------:|:----------:|:----------:|:-------------------:|:-----------------------------:|:-------------------:|
|  UNet  |     50     |  512x512   |          2          |              TBD              |         TBD         |
|  UNet  |     50     |  512x512   |          1          |        0.88  <br/> (0.93, 0.78, 0.83)         |         TBD         | 
|  UNet  |     50     |  512x512   |         0.5         |              TBD              |         TBD         |
|  UNet  |     10     | 1024x1024  |          2          |              TBD              |         TBD         |
|  UNet  |     10     | 1024x1024  |          1          | 0.91 <br/> (0.96, 0.79, 0.81) |         TBD         |
|  UNet  |     10     | 1024x1024  |         0.5         |              TBD              |         TBD         |


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

<a id="6">[6]</a> 
Lafarge et. al, (2020). 
Roto-Translation Equivariant Convolutional Networks: Application to Histopathology Image Analysis
Medical Image Analysis, Volume 68, Article 101849, Oct 2020.

<a id="7">[7]</a> 
Samanta et. al, (2021). 
Context Aggregation Network For Semantic Labeling In Histopathology Images.
2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), April 2021.