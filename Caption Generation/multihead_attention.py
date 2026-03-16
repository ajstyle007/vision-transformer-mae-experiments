import torch 
from torch import nn
import math


class MultiHeadAttention(nn.Module):
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

    def forward(self, x, mask=None):

        batch_size, seq_len, _ = x.size()

        # If x = (batch, seq_len, embed_dim):
        # Linear projections
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Split into heads: [batch, seq_len, num_heads, head_dim] -> rearrange
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # print("Q:", Q.shape)
        # print("K:", K.shape)
        # print("scores:", scores.shape)

        if mask is not None:
            # mask = mask[:, None, None, :]
            # mask = mask.to(scores.dtype)
            # scores = scores.masked_fill(mask == 0, -1e9)
            mask = mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, float("-inf")) 
                                        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # Concatenate all heads and project back to embed_dim
        out1 = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.fc_out(out1)

        return out, attention_weights
    
