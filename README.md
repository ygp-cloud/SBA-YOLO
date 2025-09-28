# SBA-YOLO
This repository contains the code and data associated with our paper titled “ Multi-scale clothing image instance segmentation method based on improved YOLOv8-seg.”
---
## The main research content includes
YOLOv8-seg is achieved by modifying the Head network structure of YOLOv8. Based on this, this study proposes the following three optimizations:
- Feature extraction network optimization based on cross-stage rapid spatial pyramid pooling
- Feature fusion network optimization based on the boundary refinement module
- Output optimization of the boundary refinement module based on the dual attention module
---
## Requirements
my computer platform Ubuntu 20.04
- PyTorch   
- CPU：Intel Core i5-9400F  
- GPU：RTX 3060  
- CUDA：v11.1  
- Python：3.7  
All models were tested under a unified experimental environment and conditions.
---
## Datasets
 The experiment was conducted based on the Modanet 、DeepFashion2 dataset.
 
---
## Experimental Design
### 1. Qualitative Experiment
Conducted comparative experiments with multiple state-of-the-art methods:
- **Compared Methods**: Mask R-CNN, Mask Scoring R-CNN, Cascade Mask R-CNN, SOLO, SOLOv2, YOLACT, and YOLOv8-seg
- **Purpose**: Visual comparison of segmentation quality and performance
### 2. Quantitative Experiment
Comprehensive performance evaluation on the Modanet dataset:
- **Dataset**: Modanet
- **Evaluation**: Quantitative performance comparison with all aforementioned methods
- **Metrics**: Standard segmentation metrics (mAP, Precision, Recall, etc.)
### 3. Ablation Experiment
Systematic validation of the proposed Dual Attention Module (DAM):
- **Primary Objective**: Verify the impact of DAM on YOLOv8-seg instance segmentation performance
- **Comparison Modules**: SE, ECA, CBAM, CA, GAM, EMA, and the proposed DAM
- **Focus**: Isolated analysis of attention mechanism effectiveness
### 4. Generalization Experiment
Cross-dataset validation to assess model robustness:
- **Models Tested**: YOLOv8-seg and SBA-YOLO
- **Datasets**: Modanet and DeepFashion2
- **Objective**: Verify generalization performance across different fashion datasets
---
## Usage

### 1. block.py
This is the model component library of YOLOv8, and its functions are:

- **Provides building blocks** ：for constructing the YOLOv8 network
- **Includes various components**: convolution, CSP structure, pooling, GhostNet, Transformer, segmentation Proto, etc.
- **Configuration method**: Configure by layer in the YAML file, and finally assemble into a complete YOLO model

### 2. conv.py
Constructing the backbone network and detection head of YOLOv8:

- **Convolution operators**: Various convolution operators (normal convolution, depthwise separable convolution, lightweight convolution, etc.)
- **Special structures**: Improved convolution modules (Focus, GhostConv, RepConv, etc.) for enhancing speed or accuracy
- **Feature fusion modules**: Integration modules (SimFusion, IFM, etc.) to combine features from different layers
- **Attention mechanisms**: Attention modules (SE, CBAM, CoordAtt, etc.) to enable the network to "focus" on important feature regions

### 3. head.py
The "Head Module" implementation of YOLOv8. It has implemented 5 main Heads:

- **Detect** → Object detection (bounding box + category probability)
- **Segment** → Image segmentation (adding mask prototype + coefficients on the basis of Detect)
- **Pose** → Pose estimation (keypoint prediction)
- **Classify** → Image classification
- **RTDETRDecoder** → Decoder for detection using Deformable Transformer

### 4. utils.py
A part of the utils module in the YOLOv8 model, mainly involving weight initialization and multi-scale deformable attention mechanism implementation.

**Processing procedure:**
1. Divide the feature values of different scales according to the `value_spatial_shapes`
2. Convert the sampling point coordinates (`sampling_locations`) to the range [-1, 1] required by `grid_sample`
3. Extract corresponding point features on each scale feature map using `F.grid_sample`
4. Weight and sum the features extracted from different scales and sampling points (weights derived from `attention_weights`)
5. Output the fused result
