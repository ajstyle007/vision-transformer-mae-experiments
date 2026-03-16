import torch
from torch import nn
from feed_forward_nn import feedforward
from positional_encoding import Positional_Encoding
from multihead_attention import MultiHeadAttention

d_model = 512  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 128  # max input length
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



x = prepare_encoder_input([7, 1542, 98])
print(x.shape)      # (1, 3, 512)


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