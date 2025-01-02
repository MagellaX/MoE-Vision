# Vision-Language Model (VLM)

## Overview
This project implements a Vision-Language Model (VLM) that combines a Vision Transformer (ViT) as the vision encoder and a transformer-based decoder for text generation. The model is designed to process multimodal data, such as images and text, to generate meaningful outputs.

### Key Features
- **Vision Transformer (ViT):** Extracts features from images using patch embeddings and self-attention.
- **Multimodal Projector:** Projects image embeddings to align with text embeddings.
- **Transformer Decoder:** Generates text conditioned on image features.
- **Training Support:** Includes data preprocessing, batching, and training loops for multimodal datasets.
- **Token Generation:** Implements autoregressive text generation conditioned on image embeddings.

---

## Installation

### Requirements
- Python 3.7+
- PyTorch (tested on GPU for optimal performance)
- pandas
- Pillow
- torchvision

Install the required libraries:
```bash
pip install torch torchvision pandas pillow
