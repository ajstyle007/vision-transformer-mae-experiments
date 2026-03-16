import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import io, math
import numpy as np

import logging
import os
from datetime import datetime 

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from feed_forward_nn import feedforward
from positional_encoding import Positional_Encoding
from multihead_attention import MultiHeadAttention


d_model = 768  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 128  # max input length
vocab_size = 30000


class FoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ----- Image Bytes → PIL -----
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ----- Transform -----
        if self.transform:
            image = self.transform(image)
        
        return image
    
    
class MultiParquetFoodDataset(Dataset):

    def __init__(self, parquet_files, transform=None):

        self.parquet_files = parquet_files
        self.transform = transform

        self.file_dfs = [pd.read_parquet(f) for f in parquet_files]

        # total indexing mapping
        self.index_map = []
        for file_idx, df in enumerate(self.file_dfs):
            for row_idx in range(len(df)):
                self.index_map.append((file_idx, row_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):

        file_idx, row_idx = self.index_map[idx]

        row = self.file_dfs[file_idx].iloc[row_idx]

        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

    

class PatchEmbedding(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Linear(
            patch_size * patch_size * in_channels,
            embed_dim
        )
        
    def forward(self, x):
        
        B, C, H, W = x.shape
        
        # ---- Image → patches ----
        x = x.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        
        # shape: B, C, num_patch_h, num_patch_w, P, P
        
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        
        x = x.permute(0, 2, 1, 3, 4)
        
        # shape: B, num_patches, C, P, P
        
        x = x.flatten(2)
        
        # shape: B, num_patches, patch_dim
        
        x = self.proj(x)
        
        return x
    

def random_masking(x, mask_ratio=0.75):
    
    """
    x: [B, N, D]
    """
    
    B, N, D = x.shape
    
    len_keep = int(N * (1 - mask_ratio))
    
    # ---- random noise generate ----
    noise = torch.rand(B, N, device=x.device)
    
    # ---- shuffle indices ----
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # ---- keep first tokens ----
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )
    
    # ---- create mask ----
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    
    # unshuffle mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore


class Encoder_block(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.ffn = feedforward(d_model, d_ff)
        self.multi_att = MultiHeadAttention(d_model, num_heads) #d_model >> embed_dim
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Self Attention
        mha_out, attn = self.multi_att(x, mask)

        # first Add & Norm (Residual connection)
        residual_1 = x + self.dropout(mha_out) 
        norm_layer1_out = self.norm_layer1(residual_1)

        # Feed Forward Network
        ffn_out = self.ffn(norm_layer1_out)

        # second Add & Norm (Residual connection)
        residual_2 = norm_layer1_out + self.dropout(ffn_out)
        norm_layer2_out = self.norm_layer2(residual_2)

        return norm_layer2_out, attn
    


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads):
        super().__init__()

        self.layers = nn.ModuleList([
            Encoder_block(d_model, d_ff, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):

        for layer in self.layers:
            x, _ = layer(x)

        return x
    

def insert_mask_tokens(x, ids_restore, mask_token):
    
    B, N_visible, D = x.shape
    N_full = ids_restore.shape[1]
    
    # ---- create mask tokens ----
    mask_tokens = mask_token.repeat(B, N_full - N_visible, 1)
    
    # ---- concatenate visible + mask ----
    x_ = torch.cat([x, mask_tokens], dim=1)
    
    # ---- restore original order ----
    x_full = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1,1,D))
    
    return x_full


class MAE_Decoder_Block(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = feedforward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # ---- Self Attention ----
        attn_out, _ = self.attn(x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # ---- Feed Forward ----
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x
    

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=8, d_ff=2048, patch_dim=768):
        
        super().__init__()

        # self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))
        self.pos_embed = Positional_Encoding(196, d_model)

        self.blocks = nn.ModuleList([
            MAE_Decoder_Block(embed_dim, num_heads, d_ff)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):

        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = self.head(x)

        return x
    

def patchify(images, patch_size=16):
    
    B, C, H, W = images.shape
    
    num_patches = H // patch_size
    
    patches = images.unfold(2, patch_size, patch_size)\
                    .unfold(3, patch_size, patch_size)
    
    patches = patches.contiguous().view(
        B, C, -1, patch_size, patch_size
    )
    
    patches = patches.permute(0, 2, 1, 3, 4)
    
    patches = patches.flatten(2)
    
    return patches

def unpatchify(patches, patch_size=16, img_size=224):
    """
    patches: [B, N, C*ps*ps]
    return:  [B, C, H, W]
    """
    B, N, D = patches.shape
    C = 3
    h = w = img_size // patch_size  # 14

    patches = patches.view(B, N, C, patch_size, patch_size)
    # [B, N, C, ps, ps]

    patches = patches.view(B, h, w, C, patch_size, patch_size)
    # [B, 14, 14, C, ps, ps]

    patches = patches.permute(0, 3, 1, 4, 2, 5)
    # [B, C, 14, ps, 14, ps]

    images = patches.reshape(B, C, img_size, img_size)
    return images


def setup_logger(log_dir="logs", name="mae"):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ---- file handler ----
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    # ---- console handler ----
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


class Cross_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  #head dim

        # Linear projections for Q, K, V
        # Linear Layers = Wq, Wk, Wv
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)

        # Final linear layer (to combine heads)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q_decoder, K_encoder, V_encoder, mask=None):
        """
        Q_input = decoder hidden state  (B, S_dec, D)
        K_input = encoder output        (B, S_enc, D)
        V_input = encoder output        (B, S_enc, D)
        """
        B, S_dec, D = Q_decoder.size()
        # print("Q_decoder:", Q_decoder.shape)

        # batch_size, seq_len, _ = x.size()
        S_enc = K_encoder.size(1)

        # If x = (batch, seq_len, embed_dim):
        # Linear projections
        Q = self.Q(Q_decoder)
        K = self.K(K_encoder)
        V = self.V(V_encoder)

        # Split into heads: [batch, seq_len, num_heads, head_dim] -> rearrange
        Q = Q.view(B, S_dec, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, S_enc, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, S_enc, self.num_heads, self.d_k).transpose(1, 2)

        # d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
                                        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate all heads and project back to embed_dim
        out = output.transpose(1,2).contiguous().view(B, S_dec, self.embed_dim)
        out = self.fc_out(out)

        return out
    


