## CADSM: Crossing-Attention Dual-Scanning Mamba For Multimodal Remote Sensing Image Classification

## Abstract

Precise categorization of multimodal remote-sensing images is essential for land-cover mapping. Although Vision Transformers perform well in capturing global contextual information, their quadratic computational burden restricts the fusion of high-resolution hyperspectral, LiDAR, and SAR data; Mamba offers linear complexity but its one-directional scanning mechanism constrains the integration of spatial context. To address this limitation, we propose a novel Mamba-based architecture that integrates attention mechanisms and U-Net techniques for multimodal remote sensing image classification. Specifically, we introduce two key modules: (1) a Dual-Scanning Mamba (DSM) module that models bidirectional context (forward-backward) with linear complexity, eliminating unidirectional bias while enabling efficient long-range dependency capture; and (2) a Crossing-Attention module that jointly attends to horizontal and vertical spatial orientations, extending receptive fields and enabling adaptive feature refinement crucial for complex boundaries. The proposed architecture, named CADSM (Crossing-Attention Dual-Scanning Mamba), was evaluated on the Muufl, Houston University, and Augsburg datasets, achieving state-of-the-art (SOTA) accuracies of 96.10%, 99.84%, and 97.24%, respectively. Additionally, tests on the Indian Pines hyperspectral dataset and the San Francisco PolSAR dataset demonstrated that CADSM significantly outperforms baseline models, highlighting its remarkable generalization capability. 

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1

## Usage:

python main.py

