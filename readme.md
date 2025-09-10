## MAMED: Multi-Attention Mamba Encoder-Decoder for Multimodal Remote Sensing Image Classification

## Abstract

Reliable interpretation of multimodal remote sensing data plays a critical role in improving the accuracy of land-cover mapping and broader geospatial intelligence. Despite their strengths, existing methods face significant challenges: Vision Transformers suffer from quadratic computational complexity, limiting their scalability with high-resolution inputs, while Mamba-based models, though efficient, lack the capacity to capture bidirectional spatial context due to unidirectional state transitions. To overcome these limitations, we propose MAMED (Multi-Attention Mamba Encoder–Decoder), a novel architecture that integrates Mamba’s selective state-space mechanisms with multi-directional self-attention within a U-shaped encoder–decoder framework. The model employs a lightweight Mamba-Transformer mixer to capture long-range dependencies linearly and refines local features through cross-scale skip connections. Extensive experiments on three multimodal benchmarks (Muufl, Houston, and Augsburg) demonstrate that MAMED outperforms state-of-the-art CNN, Transformer, and Mamba models, achieving overall accuracies of 96.25%, 99.80%, and 97.74%, respectively. Cross-domain evaluations on Indian Pines (hyperspectral) and Flevoland (PolSAR) further confirm its strong generalization and applicability across diverse remote sensing tasks.   

## Requirements:

- Python 3.7
- PyTorch >= 1.12.1

## Usage:

python main.py

