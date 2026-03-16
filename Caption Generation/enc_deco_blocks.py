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

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Other_classes import PatchEmbedding, TransformerEncoder
from positional_encoding import Positional_Encoding

class MAEEncoder(nn.Module):
    def __init__(self, seq_len=196, embed_dim=768):
        super().__init__()

        self.patch_embed = PatchEmbedding()
        self.pos_encoding = Positional_Encoding(seq_len, embed_dim)

        self.encoder = TransformerEncoder(
            num_layers=8,
            d_model=embed_dim,
            d_ff=2048,
            num_heads=8
        )

    def forward(self, images):
        tokens = self.patch_embed(images)      # [B,196,768]
        tokens = self.pos_encoding(tokens)
        latent = self.encoder(tokens)          # [B,196,768]

        # emb = latent.mean(dim=1)               # mean pooling
        # emb = F.normalize(emb, dim=-1)

        return latent                     



device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "mae_epoch_new_69.pth"

model = MAEEncoder().to(device)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"]

model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ MAE Encoder loaded successfully")



def generate_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    mask = (~mask).unsqueeze(0).unsqueeze(1)   # (1,1,L,L)
    return mask


class Decoder_Block(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()

        self.ffn = feedforward(d_model, d_ff)
        # self.multi_att = MultiHeadAttention(d_model, num_heads) #d_model >> embed_dim
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.cross_att = Cross_Attention(d_model, num_heads)
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.norm_layer3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask=None, padding_mask=None):

        B, S, D = x.shape
        if mask is None:
            mask = generate_subsequent_mask(S)  # (1,1,S,S)
        # Masked Multi-Head Self Attention
        masked_mha_out,  _  = self.masked_mha(x, mask)

        # first Add & Norm (Residual connection)
        residual_1 = x + self.dropout(masked_mha_out)
        norm_layer1_out = self.norm_layer1(residual_1)

        # cross attention or encoder-decoder attention, this "out" is K, V coming from the encoder
        cross_att_out = self.cross_att(norm_layer1_out, enc_out, enc_out, mask=padding_mask)

        # second Add & Norm (Residual connection)
        residual_2 = norm_layer1_out + self.dropout(cross_att_out)
        norm_layer2_out = self.norm_layer2(residual_2)

        # Feed Forward Network
        ffn_out = self.ffn(norm_layer2_out)

        # third Add & Norm (Residual connection)
        residual_3 = norm_layer2_out + self.dropout(ffn_out)
        norm_layer3_out = self.norm_layer3(residual_3)

        return norm_layer3_out
    

d_model = 768  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 196  # max input length
vocab_size = 30000

embedding_layer = nn.Embedding(vocab_size, d_model)

pos_encoding = Positional_Encoding(seq_len, d_model)


def prepare_encoder_input(token_ids):
    token_ids = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

    # 1. Convert token IDs → learned embeddings
    x = embedding_layer(token_ids)                      # (1, seq_len, d_model)

    # 2. Add sinusoidal positional encoding
    x = pos_encoding(x)                                 # (1, seq_len, d_model)

    return x


def prepare_decoder_input(token_ids):
    token_ids = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)

    # 1. Convert token IDs → learned embeddings
    x = embedding_layer(token_ids)                      # (1, seq_len, d_model)

    # 2. Add sinusoidal positional encoding
    x = pos_encoding(x)                                 # (1, seq_len, d_model)

    return x



class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [Decoder_Block(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, mask=None):
        """
        x       : (B, S_dec, D)
        enc_out : (B, S_enc, D)
        tgt_mask: causal mask (1,1,S_dec,S_dec)
        """

        for layer in self.layers:
            x = layer(x, enc_out, mask)

        return self.norm(x)
    

# # enc_in = prepare_encoder_input([12,43,55,99])
# images = torch.randn(1, 3, 224, 224) 
# encoder = MAEEncoder()
# enc_out = encoder(images) # >> this out is both K, V will go to MHA attention(cross atten) of decoder

# decoder = Decoder(num_layers=6, d_model=768, d_ff=2048, num_heads=8)

# dec_inp = prepare_decoder_input([7, 1542, 98])    # (1, 3, 512)
# enc_out = enc_out                                   # (1, 4, 512)

# print("encoder shape:", enc_out.shape)

# mask = generate_subsequent_mask(dec_inp.size(1))

# out = decoder(dec_inp, enc_out, mask)
# print(out.shape)  # torch.Size([1, 3, 512])
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("food_tokenizer.json")

images = torch.randn(1, 3, 224, 224)
encoder = MAEEncoder()
enc_out = encoder(images)  # [1,196,768]

BOS_ID = 2
EOS_ID = 3

dec_inp = prepare_decoder_input([BOS_ID])

mask = generate_subsequent_mask(dec_inp.size(1))

decoder = Decoder(num_layers=6, d_model=768, d_ff=2048, num_heads=8)
out = decoder(dec_inp, enc_out, mask)  # [1,1,768]
print(out.shape)