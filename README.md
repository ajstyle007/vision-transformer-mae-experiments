# Vision Transformer & MAE Learning Experiments

![Uploading Screenshot 2026-03-16 233144.png…]()

This repository contains my experiments while learning:

• Vision Transformers (ViT)  
• Masked Autoencoders (MAE)  
• Image embeddings for semantic image retrieval  
• Transformer-based caption generation

The goal of this repository is to understand how modern vision architectures work and how they can be used for tasks such as image similarity search and caption generation.

## What I Implemented

During this learning process I experimented with:

- Training a Vision Transformer encoder using Masked Autoencoders
- Generating image embeddings from the trained encoder
- Building a semantic image retrieval system
- Implementing a Transformer decoder for image caption generation
- Creating a small web interface to test image similarity search

## Experiments

1. MAE pretraining on Food-101
2. Image embedding generation
3. Semantic image similarity search
4. Transformer decoder caption generation

<img width="1291" height="986" alt="Screenshot 2026-03-16 233256" src="https://github.com/user-attachments/assets/d6c51ce9-9ee1-4b54-8528-07d75fae98cc" />

## Learning Outcome

Through these experiments I learned:

- How Vision Transformers process images
- How Masked Autoencoders learn visual representations
- How encoder–decoder architectures work for vision-language tasks
- How image embeddings can be used for similarity search

## Notes from My Experiments

After implementing the caption generation pipeline on top of the Vision Transformer encoder trained with **Masked Autoencoder**, I trained the decoder for about **two days**. While the training loss decreased initially, the generated captions did not improve significantly after a certain point.

Some possible reasons for this could be:

* The **MAE encoder might not have been trained long enough**, meaning the visual representations learned were still relatively basic.
* During caption generation training, I **kept the encoder frozen**, which may have limited the model’s ability to adapt visual features for the language generation task.
* Unfreezing the **last few layers of the Vision Transformer** during caption training might have helped improve results by allowing better alignment between image features and text tokens.

Since the main goal of this repository is **learning and experimentation**, I decided to stop the caption generation experiment at this stage and move forward to exploring **CLIP-style vision–language models**.

Even though the captioning experiment did not fully succeed, the trained encoder was still useful. Using the same encoder, I built a small application that demonstrates:

* **Upload Image** – upload a food image
* **Search Similar Images** – retrieve visually similar dishes using image embeddings
* **Run MAE Reconstruction** – visualize how the encoder reconstructs masked images

This repository therefore documents my learning journey while experimenting with Vision Transformers, MAE pretraining, and early attempts at building vision–language systems.

