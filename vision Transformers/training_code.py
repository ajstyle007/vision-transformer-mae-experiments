import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import glob

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from feed_forward_nn import feedforward
from positional_encoding import Positional_Encoding
from multihead_attention import MultiHeadAttention

from tqdm import tqdm

from Other_classes import (FoodDataset, PatchEmbedding, random_masking, Encoder_block, 
TransformerEncoder, insert_mask_tokens, MAE_Decoder_Block, MAEDecoder, patchify, MultiParquetFoodDataset)

device = "cuda"

# df = pd.read_parquet("data/train-00000-of-00008-26f523e9bdcc2b9a.parquet")

# sample = df.iloc[4227]

# image_bytes = sample['image']['bytes']
# image = Image.open(io.BytesIO(image_bytes))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# dataset = FoodDataset(df, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

parquet_files = glob.glob("data/*.parquet")

dataset = MultiParquetFoodDataset(parquet_files, transform=transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

patch_embed = PatchEmbedding()
batch = next(iter(dataloader))
print("batch_shape: ", batch.shape)

tokens = patch_embed(batch)
print("tokens_shape:" , tokens.shape)


visible_tokens, mask, ids_restore = random_masking(tokens)


d_model = 768  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 196  # max input length
vocab_size = 30000

pos_encoding = Positional_Encoding(seq_len, d_model)
visible_tokens = pos_encoding(visible_tokens)

encoder = TransformerEncoder(num_layers=8, d_model=768, d_ff=2048, num_heads=8)

latent = encoder(visible_tokens)
print("latent.shape: ", latent.shape)

mask_token = nn.Parameter(torch.zeros(1, 1, 768))

decoder_input = insert_mask_tokens(latent, ids_restore, mask_token)
print("decoder_input_shape: ", decoder_input.shape)


decoder = MAEDecoder()
pred = decoder(decoder_input)
print("pred_shape: ", pred.shape)


target = patchify(batch)
print("target_shape: ", target.shape)


class MAE(nn.Module):

    def __init__(self, patch_embed, encoder, decoder, mask_token, pos_encoding):
        super().__init__()

        self.patch_embed = patch_embed
        self.encoder = encoder
        self.decoder = decoder
        self.mask_token = mask_token
        self.pos_encoding = pos_encoding

    def forward(self, images):

        tokens = self.patch_embed(images)

        visible_tokens, mask, ids_restore = random_masking(tokens)

        visible_tokens = self.pos_encoding(visible_tokens)

        latent = self.encoder(visible_tokens)

        decoder_input = insert_mask_tokens(
            latent, ids_restore, self.mask_token
        )

        pred = self.decoder(decoder_input)

        target = patchify(images)

        
        target = (target - target.mean(dim=-1, keepdim=True)) / \
                (target.std(dim=-1, keepdim=True) + 1e-6)

        return pred, target, mask


def mae_loss(pred, target, mask):

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)     # patch level loss
    
    loss = (loss * mask).sum() / mask.sum()
    
    return loss


model = MAE(patch_embed, encoder, decoder, mask_token, pos_encoding).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

epochs = 5 
for epoch in range(epochs):

    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for images in loop:

        images = images.to(device)

        optimizer.zero_grad()

        pred, target, mask = model(images)

        loss = mae_loss(pred, target, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- tqdm live loss update ----
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "mae_model.pth")
