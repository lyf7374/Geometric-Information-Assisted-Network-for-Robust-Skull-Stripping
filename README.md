# Geometric-Information-Assisted Network for Robust Skull-Stripping

## Abstract

Skull stripping, the process of isolating brain tissue from non-brain structures in MRI data, is a critical step in neuroimaging workflows. While deep learning–based methods have shown promise in brain extraction, they often struggle with generalization across diverse datasets due to patient variability, dataset differences, and the presence of pathological conditions. This limitation arises from their reliance on local texture patterns, which makes them sensitive to input variations.

In this work, we present a Geometric-Information-Assisted Network (GINet) that improves skull stripping by fusing geometric information—a texture-invariant prior—into the segmentation pipeline, mitigating variability across subjects and datasets. Specifically, GINet consists of three main components:

GI filtering module
Efficiently extracts lightweight geometric priors from input scans.

GI adaptation module
Adapts a standard brain template to input-specific GI while preserving structural integrity.

Deep GI-fused segmentation module
Hierarchically integrates GI with image features for robust segmentation.

Additionally, we propose a multi-depth (MD) loss function to hierarchically refine feature maps with adaptive GI. Despite being trained solely on healthy data, GINet generalizes effectively to various unseen healthy and pathological datasets. Our results demonstrate substantial improvements in robustness and accuracy over several state-of-the-art skull-stripping methods.
## Project Structure

The directory structure is organized as follows:

- `datasets/` — GI support generation utilities  
  - `GI_generate_support.py` — Script to create geometric priors
- `model/` — Model architecture definitions  
  - Contains core GINet architecture components
- `utils/` — Utility functions and helper scripts
- `GINet_train.py` — Training script for GINet
- `GI_extration.ipynb` — Jupyter notebook for GI extraction
- `evaluation.py` — Evaluation script on multiple datasets
- `MNI152_T1_1mm_brain.nii.gz` — Standard MNI brain template
- `mni_icbm152_t1_*.nii` — Sample image and mask files
- `README.md` — This documentation file

## Usage

### 1. Extract Geometric Information (GI)

Run the notebook:

```bash
GI_extration.ipynb
This demonstrates how to extract geometric priors using the MNI template and the support logic found in:
datasets/GI_generate_support.py

### 2. Train GINet
python GINet_train.py

3. Evaluate the Model
python evaluation.py

