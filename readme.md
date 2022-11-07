 ## BE expert-level gland grading with Neural Networks
This repository contains code for development of a Neural Network that support pathologists with identifying dysplastic 
areas of interest on H&E tissue samples from patients with Barrett's Esophagus (BE). 

### The Task
BE is associated with an increased risk of developing esophageal cancer. Regular check-ups and pathological assessment of biopsy material are crucial for identifying BE patients at risk.
Dysplasia in BE is assessed according to the revised Vienna criteria, which are based on the dysplasia classification in inflammatory bowel disease. Evaluation of cytological and architectural severity and invasion status leads to assignment in
one of the following categories: non-dysplastic Barrett's esophagus (NDBE), indefinite for dysplasia (IND), low grade dysplasia (LGD) and high grade dysplasia (HGD). Key characteristics used to assess the
dysplasia grade in BE are surface maturation, glandular architecture, and cytonuclear changes. Description and figure from [[1]](#1).


![](images/examples_grading_BE.png)


## Outline of the project
### (1) Segmentation model for: ND vs D and NDBE vs LGD vs HGD
- [x] Split data for training, evaluation and testing. We keep Bolero apart as hold out test set.
- [x] Train standard baseline segmentation models for grading into: ND vs D and NDBE vs LGD vs HGD.
- [x] Visualization and evaluation (dice and pixel level confusion matrix), preferably in a notebook.
- [x] Assess calibration of segmentation with Brier, ECE and NLL for: 
  - Ensembles 
  - Different loss functions: Dice and/or CCE
  - **Ensemble of CE probably a good choice.**
- [x] Stain Normalization 
  - **Seems to make the grading harder. Results are significantly worse.**
- [ ] 
  
### (2) Slide-Level Aggregation
- [x] Extract tissue containing tiles in WSI's. 
- [x] Rank tiles according to segmentation probabilities and confidence for dysplasia. 
- [ ] Train aggregation models (Transformer, Attention-Pooling).

  Each slide $s$ is a sample $(\textbf{x}, y)$:
  
   * $\textbf{x}$: sequence of top N suspicious tiles for slide $s$
   * $y$: dysplasia label of slide $s$
  - [x] GRU (with ordered tiles)
  - [x] Set Transformer
  - [ ] DeepMIL [[2]](#1) (with all tiles)
- [ ] Visualization

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

**Evaluation on slide level:**
  * Quadratic weighted kappa grade + confusion matrix
  * Multiclass AUC

[//]: # (#### ND vs D)

[//]: # (|   Method   |           Encoder            | Batch Size | Patch Size |          Spacing <br/> &#40;mpp&#41;           |                              Internal Test <BR> &#40;Dice&#41; <br> [BG, NDBE, DYS]                              | External Test <br> &#40;Dice&#41; |)

[//]: # (|:----------:|:----------------------------:|:----------:|:----------:|:--------------------------------------:|:--------------------------------------------------------------------------------------------------------:|:-------------------------:|)

[//]: # (|    UNet    |    ResNet34 <br> depth=5     |     8      |    1024    |                   1                    |                                      0.93 <br>  [0.97, 0.83, 0.86]                                       |                           |)

[//]: # (|   UNet++   | EfficientNet-b4 <br> depth=5 |     8      |    1024    | 2 <br> <br> <br> 1 <br> <br> <br>  0.5 | 0.96 <br> [0.98 0.83 0.86] <br> <br>  0.94 <br> [0.97, 0.86, 0.87] <br> <br>  0.93 <br> [0.96 0.86 0.89] |                           |)

[//]: # ()
[//]: # ()
[//]: # (#### NDBE vs LGD vs HGD)

[//]: # (| Method |           Encoder            | Batch Size | Patch Size | Spacing <br/> &#40;mpp&#41; | Internal Test <BR> &#40;Dice&#41; <br> [BG, NDBE, LGD, HGD] | External Test <br> &#40;Dice&#41; |)

[//]: # (|:------:|:----------------------------:|:----------:|:----------:|:-------------------:|:---------------------------------------------------:|:-------------------------:|)

[//]: # ( | UNet++ | EfficientNet-b4 <br> depth=5 |     8      |    1024    |                     |                                                     |                           |)

[//]: # (                                                                                                                                                                                                                                      )

## References
<a id="1">[1]</a> 
M.J. van der Wel, (2019). 
PhD thesis, Faculty of Medicine (AMC-UvA), December 2019.

<a id="2">[2]</a> 
Ilse et. al, (2018).
Attention-based Deep Multiple Instance Learning
Proceedings of the 35th International Conference on MachineLearning, Stockholm, Sweden, PMLR 80, 2018.