# Oracle Imitating Selection

## Framework Overview

![Framework](framework.png)

## Purpose

To develop more effective Active Learning (AL) patch recommendations

## Methods

1. **Building Oracle Selection Classification Model (Version 1)**
   - Construct a simple classification model for predicting between two classes: oracle selection or not

2. **Enhanced Model (Version 2)**
   - Extract features from images
   - Concatenate image features with meta-information:
     - Whole Slide Image (WSI) ground truth
     - Patch prediction from DenseNet201 pretrained model
     - Patch confidence score from DenseNet201 pretrained model

3. **Unified Model (Version 3)**
   - Similar to Version 2, with the exception that the feature extractor and imitation prediction (simple model) are integrated into a single model
   - Downsampled validation dataset (balanced) is not fixed. This change is intended to test the model's generalizability more comprehensively

## Active Learning Strategy

Implements an AL strategy that combines the uncertainty from the AL process and the probability of oracle selection from the oracle-imitating selection model
