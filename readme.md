## BE expert-level gland grading with Neural Networks
This repository contains code for development of a Neural Network that support pathologists with identifying dysplastic 
areas of interest on H&E tissue samples from patients with Barrett's Esophagus (BE). 

### The Task
BE is associated with an increased risk of developing esophageal cancer. Regular check-ups and pathological assessment of biopsy material are crucial for identifying BE patients at risk.
Dysplasia in BE is assessed according to the revised Vienna criteria, which are based on the dysplasia classification in inflammatory bowel disease. Evaluation of cytological and architectural severity and invasion status leads to assignment in
one of the following categories: non-dysplastic Barrett's esophagus (NDBE), indefinite for dysplasia (IND), low grade dysplasia (LGD) and high grade dysplasia (HGD). Key characteristics used to assess the
dysplasia grade in BE are surface maturation, glandular architecture, and cytonuclear changes. Description and figure from [[1]](#1).


![](images/examples_grading_BE.png)


### Outline of the project:

- [x] Create a clean archive dataset:
  - [x] Individual directories for each dataset: ASL, Bolero, LANS and RBE.
     - [x] Containing all the image files (.tiff converted in the same fashion) and annotation (.xml) files.
     - [x] A csv file with case or biopsy level diagnosis (inferred from Sybren's gland annotations).
- [x] Split data for training, evaluation and testing. We keep Bolero apart as hold out test set.
- [x] Train standard baseline segmentation models for grading into: NDBE vs Dysplasia (LGD and HGD).
- [x] Visualization and evaluation (dice and pixel level confusion matrix), preferably in a notebook.
     - [ ] Stitch results back together on slide/biopsy level (currently evaluating with sliding window over the ROI's).
- [x] Create a reader study to collect biopsy level grades from multiple pathologists:
      https://grand-challenge.org/reader-studies/barretts-grading/
     - [ ] Add P53 in the viewer.
- [ ] Create a processor: takes in WSI, outputs a mask (graded glands).
- [ ] Deploy on Grand-Challenge.
- [ ] Experiments:
  * Context aggregation networks for segmentation in pathology: HistNet [[7]](#7), HookNet [[3]](#3), RAENet [[4]](#4).
  * Roto-Translation Equivariant CNN's [[6]](#6).
  
### Segmentation pipeline for gland grading into: NDBE, DYS
**For on fly patch extraction we use:** https://github.com/DIAGNijmegen/pathology-whole-slide-data.
  * Includes configuration for patch extraction such as batch size, patch size, spacing.
  * Includes different strategies to sample patches from the WSI (balanced, random, slidingwindow).
  * Configuration stored in (`configs/base_config.yml`).

**Data Augmentation:** 
  * Rotates, flips, gamma transform and color jitter (saturation, contrast, hue)

### Datasets 
Below a summary is shown of all the data available for this project. Gland level annotations were provided by Sybren Meijer for the categories: NDBE, LGD and HGD.
Case or slide level annotations were not provided yet. The relevant datasets on AMC servers (/data/archief/AMC-data/Barrett/) are listed below:

| Dataset               | Cases | Biopsies | Diagnosis Available<br/>(biopsy/case level) | Raters<br/>(biopsy/case level) | P53 <br/> Available | 
|-----------------------|:-----:|:--------:|:-------------------------------------------:|:------------------------------:|:-------------------:|
| ASL                   |  36   |   139    |                     yes                     |             Sybren             |         Yes         |    
| LANS                  |  34   |   106    |                     yes                     |             Sybren             |         Yes         |          
| RBE                   |  225  |   540    |                     yes                     |             Sybren             |         Yes         |         
| **Total Development** |  295  |   785    |                     yes                     |             Sybren             |         Yes         |  
| **Bolero**            |  51   |   193    |                     yes                     |               4                |         Yes         |   


### Experiments
We randomly split the development set of 295 WSIs in a train, validation and test set (236/29/30), in which each set roughly has the same pixel level percentages of NDBE (~75%), LGD (~12%) and HGD (~12%).

**Evaluation:**
  * Quantitative with DICE/F1 compared to the ground truth annotations.
  * Qualitative assessment/feedback from Sybren (other pathologists?).
  
|   Method   |           Encoder            | Batch Size | Patch Size | Spacing <br/> (mpp) |   Internal Test <BR> (Dice)   | External Test <br> (Dice) |
|:----------:|:----------------------------:|:----------:|:----------:|:-------------------:|:-----------------------------:|:-------------------------:|
|    UNet    |    ResNet34 <br> depth=5     |     8      |    1024    |          1          | 0.93 <br>  [0.97, 0.83, 0.86] |
|   UNet++   | EfficientNet-b4 <br> depth=5 |     8      |    1024    |          1          | 0.94 <br> [0.97, 0.86, 0.87]  |
| DeepLabV3+ |                              |            |            |                     |                               |
|  HookNet   |                              |            |            |                     |                               |
|  HistNet   |                              |            |            |                     |                               |


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