# Vision-Language Model (VLM)

## Overview
This project implements a Mixture-of-experts Vision-Language Model (VLM) that combines a Vision Transformer (ViT) as the vision encoder and a transformer-based decoder for text generation. The model is designed to process multimodal data, such as images and text, to generate meaningful outputs.

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


Please note that in this simple example(MoE-Vision.ipynb), we are training the entire system end to end, much like Kosmos-1 from Microsoft Research. I left it at this for convenience. In practice, the commonly observed sequence is:

Get pretrained vision encoder from SigLIP or CLIP (both come in different sizes). Freeze weights (i.e. don’t update during backward pass in training)

Get a decoder-only language model, e.g., all the way from TinyLLaMA, Phi-2, etc., to Llama 3 (or even much bigger in the case of GPT-4 and Grok 1.5, etc.). Freeze weights.

Implement a projection module and train a VLM module much like what we have here, but only updating the weights of this projection module. This would effectively be the pretraining phase.

Then, during the instruction finetuning, keep both the projection module and the decoder language model unfrozen and update their weights in the backward pass.

I developed this on lightning AI  using a single T4 and MLFlow for tracking loss (during the training process). I wanted to set this up this way so that I can scale up to a GPU cluster of any size I want quite easily on Databricks, should I decide to adapt this to a more performance-oriented implementation. However, you can run this anywhere, with or without a GPU. Please note that even the toy training loop with 90 samples will be painfully slow on a CPU.
