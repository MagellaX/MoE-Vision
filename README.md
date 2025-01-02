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
```
(seemore_from_Scratch.ipynb walks through the intuition for the entire model architecture and how everything comes together. I recommend starting here...)

Please note that in this simple example(MoE-Vision.ipynb), we are training the entire system end to end, much like Kosmos-1 from Microsoft Research. I left it at this for convenience. In practice, the commonly observed sequence is:

Get pretrained vision encoder from SigLIP or CLIP (both come in different sizes). Freeze weights (i.e. don’t update during backward pass in training)

Get a decoder-only language model, e.g., all the way from TinyLLaMA, Phi-2, etc., to Llama 3 (or even much bigger in the case of GPT-4 and Grok 1.5, etc.). Freeze weights.

Implement a projection module and train a VLM module much like what we have here, but only update the weights of this projection module. This would effectively be the pretraining phase.

Then, during the instruction finetuning, keep both the projection module and the decoder language model unfrozen and update their weights in the backward pass.

I developed this on lightning AI  using a single T4 and MLFlow for tracking loss (during the training process). I wanted to set this up this way so that I can scale up to a GPU cluster of any size I want quite easily on Databricks, should I decide to adapt this to a more performance-oriented implementation. However, you can run this anywhere, with or without a GPU. Please note that even the toy training loop with 90 samples will be painfully slow on a CPU.







In this simple implementation of a vision language model (VLM), there are 3 main components.

Image Encoder to extract visual features from images. In this case, I use a from scratch implementation of the original vision transformer used in CLIP. This is actually a popular choice in many modern VLMs. The one notable exception is Fuyu series of models from Adept, which passes the patchified images directly to the projection layer.

Vision-Language Projector - Image embeddings are not of the same shape as text embeddings used by the decoder. So we need to ‘project’ i.e. change dimensionality of image features extracted by the image encoder to match what’s observed in the text embedding space. So image features become ‘visual tokens’ for the decoder. This could be a single layer or an MLP. I’ve used an MLP because it’s worth showing.

A decoder only language model. This is the component that ultimately generates text. In my implementation I’ve deviated from what you see in LLaVA etc. a bit by incorporating the projection module to my decoder. Typically this is not observed, and you leave the architecture of the decoder (which is usually an already pretrained model) untouched.




The scaled dot product self attention implementation is borrowed from Andrej Kapathy's makemore (https://github.com/karpathy/makemore). Also the decoder is an autoregressive character-level language model, just like in makemore. Now you see where the name 'seemore' came from :)

Everything is written from the ground up using pytorch. That includes the attention mechanism (both for the vision encoder and language decoder), patch creation for the vision transformer and everything else. Hope this is useful for anyone going through the repo and/ or the associated blog.


Publications heavily referenced for this implementation:

Large Multimodal Models: Notes on CVPR 2023 Tutorial: https://arxiv.org/pdf/2306.14895.pdf
Visual Instruction Tuning: https://arxiv.org/pdf/2304.08485.pdf
Language Is Not All You Need: Aligning Perception with Language Models: https://arxiv.org/pdf/2302.14045.pdf

The input.txt with tinyshakespear and the base64 encoded string representations + corresponding descriptions are in the inputs.csv file in the images directory.

Please note that the implementation emphasizes readability and hackability vs. performance, so there are many ways in which you could improve this. Please try and let me know!

Hope you find this useful. Happy hacking!!
