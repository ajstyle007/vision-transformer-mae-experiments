import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import glob
import os

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR, LambdaLR
from torchvision.utils import save_image

from feed_forward_nn import feedforward
from positional_encoding import Positional_Encoding
from multihead_attention import MultiHeadAttention

from tqdm import tqdm

from Other_classes import (FoodDataset, PatchEmbedding, random_masking, Encoder_block, unpatchify, setup_logger,
TransformerEncoder, insert_mask_tokens, MAE_Decoder_Block, MAEDecoder, patchify, MultiParquetFoodDataset)

device = "cuda"

logger = setup_logger(log_dir="logs", name="mae_pretrain")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


parquet_files = glob.glob("data/*.parquet")

dataset = MultiParquetFoodDataset(parquet_files, transform=transform)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

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

        # mean = images.mean(dim=[2,3], keepdim=True)
        # std  = images.std(dim=[2,3], keepdim=True) + 1e-6
        # norm_images = (images - mean) / std

        # target = patchify(norm_images)

        target = patchify(images)

        return pred, target, mask


def mae_loss(pred, target, mask):

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)     # patch level loss
    
    loss = (loss * mask).sum() / mask.sum()
    
    return loss

def denormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1,3,1,1)
    return img * std + mean

def save_checkpoint(epoch, model, optimizer,scheduler, loss, path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "loss": loss
    }, path)


val_img = next(iter(dataloader))[:1].to(device)

@torch.no_grad()
def run_inference(model, img):
    model.eval()
    pred, target, mask = model(img)

    mask = mask.unsqueeze(-1)
    recon = target * (1 - mask) + pred * mask
    img_recon = unpatchify(recon)

    return img_recon


epochs = 200
start_epoch = 0
best_loss = float("inf")

scaler = torch.cuda.amp.GradScaler()


model = MAE(patch_embed, encoder, decoder, mask_token, pos_encoding).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)

warmup_epochs = 10

# Warmup lambda
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs  # linear from ~0 to 1
    return 1.0

warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])


os.makedirs("checkpoints", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

checkpoint_path = "checkpoints/mae_epoch_new_40.pth"

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]
    best_loss = ckpt.get("loss", float("inf"))
    # scheduler.load_state_dict(ckpt.get("scheduler", scheduler.state_dict()))
    for _ in range(start_epoch):
        scheduler.step()

    logger.info(f"Resumed training from epoch {start_epoch}")
else:
    logger.info("No checkpoint found, starting from scratch")


for epoch in range(start_epoch, epochs):

    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

    for images in loop:

        images = images.to(device)

        optimizer.zero_grad(set_to_none=True)
       
        with torch.cuda.amp.autocast(dtype=torch.float16):
            pred, target, mask = model(images)
            loss = mae_loss(pred, target, mask)

        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0) # or 1.0–5.0

        if (loop.n + 1) % 1000 == 0:          # print every 50 batches (~every 50–100 seconds)
            print(f"Batch {loop.n+1} | Grad norm: {grad_norm:.4f} | loss: {loss.item():.4f}")

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # ---- tqdm live loss update ----
        loop.set_postfix(loss=loss.item())

    scheduler.step()
    avg_loss = total_loss / len(dataloader)

    if (epoch + 1) % 1 == 0:
        ckpt_path = f"checkpoints/mae_epoch_new_{epoch+1}.pth"
        save_checkpoint(epoch+1, model, optimizer, scheduler, avg_loss, ckpt_path)

        logger.info(f"Checkpoint saved at {ckpt_path}")

        recon_img = denormalize(run_inference(model, val_img))
        save_image(recon_img.clamp(0,1), f"outputs/recon_epoch_new_new_{epoch+1}.png")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(epoch+1, model, optimizer,scheduler, avg_loss, 
            "checkpoints/mae_new_best.pth"
        )

    logger.info(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")


torch.save(model.state_dict(), "mae_model.pth")
