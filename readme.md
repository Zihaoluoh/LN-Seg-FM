# Implementation of LN-Seg-FM

This repository contains the implementation of **Dynamic Gradient Sparsification Training (DGST)** for few-shot fine-tuning of a lymph node (LN) segmentation foundation model (LN-Seg-FM) using nnUNetv2. The approach introduces a novel method that balances model stability and flexibility for LN segmentation tasks, which is critical for clinical applications with limited annotated data.

## Overview

We utilize nnUNetv2, a popular framework for medical image segmentation, to implement a series of `nnUNetTrainer` scripts. The specific trainer used in this implementation is **nnUNetTrainer_DGST.py**. This implementation demonstrates the effectiveness of the DGST approach in improving segmentation performance with minimal labeled data.

## Installation

1. Clone this repository:

Ensure that you have nnUNetv2 installed. You can follow the installation instructions from the [nnUNetv2](https://github.com/MIC-DKFZ/nnUNet) repository.

##Usage

Transfer the custom trainer files into the nnUNetv2/training/nnUNetTrainer directory:
bash
mv nnUNetTrainer_DGST.py /path/to/nnUNetv2/training/nnUNetTrainer/
Train the model using the nnUNetv2_train -tr command, specifying the trainer:
bash nnUNetv2_train -tr nnUNetTrainer_DGST ...

##Dataset

The dataset used for training and evaluation consists of 36,106 annotated visible lymph nodes (LNs) across 3,346 publicly available head-and-neck CT scans[RADCURE](https://www.cancerimagingarchive.net/collection/radcure). The annotated visible lymph nodes mask will be publicly available upon acceptance.

