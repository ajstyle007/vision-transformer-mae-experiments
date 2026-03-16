import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import faiss
import numpy as np
import glob
from Other_classes import Cross_Attention
from multihead_attention import MultiHeadAttention
from feed_forward_nn import feedforward
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Other_classes import PatchEmbedding, TransformerEncoder
from positional_encoding import Positional_Encoding
from enc_deco_blocks import MAEEncoder, Decoder, generate_subsequent_mask
from Food_Caption_Dataset import FoodCaptionDataset

d_model = 768  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 196  # max input length
vocab_size = 30000

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "mae_epoch_new_69.pth"

# model = MAEEncoder()
encoder = MAEEncoder().to(device)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"]

encoder.load_state_dict(state_dict, strict=False)
encoder.eval()

print("✅ MAE Encoder loaded successfully")


tokenizer = Tokenizer.from_file("food_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
lm_head = nn.Linear(768, vocab_size).to(device)


embedding_layer = nn.Embedding(vocab_size, d_model).to(device)

pos_encoding = Positional_Encoding(seq_len, d_model).to(device)

decoder = Decoder(num_layers=6, d_model=768, d_ff=2048, num_heads=8).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # ignore [PAD]

optimizer = torch.optim.AdamW(
    list(decoder.parameters()) +
    list(lm_head.parameters()),
    lr=3e-4
)


dataset = FoodCaptionDataset(captions_path = "captions_new.parquet", images_path   = "images_new.parquet",
        max_length = 32,
    )

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)


epochs = 10
global_step = 0

for epoch in range(epochs):

    total_loss = 0
    

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in pbar:

        images = batch["image"].to(device)
        decoder_input = batch["input_ids"].to(device)
        target_tokens = batch["target_ids"].to(device)

        # ---------------------------------
        # Encoder forward
        # ---------------------------------

        enc_out = encoder(images)      # (B,196,768)

        # ---------------------------------
        # Embedding
        # ---------------------------------

        dec_embed = embedding_layer(decoder_input)
        dec_embed = pos_encoding(dec_embed)

        mask = generate_subsequent_mask(dec_embed.size(1))

        # ---------------------------------
        # Decoder forward
        # ---------------------------------

        
        mask = mask.to(dec_embed.device)

        dec_out = decoder(dec_embed, enc_out, mask)   # (B,S,768)
        # ---------------------------------
        # Vocab projection
        # ---------------------------------

        logits = lm_head(dec_out)  # (B,S,vocab)

        # reshape for CE loss

        B,S,V = logits.shape

        logits = logits.view(B*S, V)
        targets = target_tokens.reshape(B*S)

        loss = criterion(logits, targets)

        # ---------------------------------
        # Backprop
        # ---------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

        # save every 100 steps
        if global_step % 1000 == 0:
            ckpt = {
                "step": global_step,
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "lm_head": lm_head.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            torch.save(ckpt, f"decoder_checkpoint_step_{global_step}.pth")
            print(f"\n✅ Checkpoint saved at step {global_step}")

        total_loss += loss.item()

        pbar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss:", total_loss/len(train_loader))