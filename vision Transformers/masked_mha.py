import torch
from torch import nn
import math
from RoPE import apply_rope, build_rope_frequencies

class Masked_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads  #head dim

        self.cos, self.sin = build_rope_frequencies(seq_len=max_seq_len, head_dim=self.d_k, base=10000)

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

        # Apply RoPE only to Q and K
        Q = apply_rope(Q, self.cos[:seq_len], self.sin[:seq_len])
        K = apply_rope(K, self.cos[:seq_len], self.sin[:seq_len])

        # Scaled dot-product attention
        # d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 4. Apply mask (mask shape must be broadcastable)
        if mask is not None:
            # scores = scores.masked_fill(mask == 0, float("-inf"))
            scores = scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(attention_weights, V)

        # Concatenate all heads and project back to embed_dim
        out = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.fc_out(out)

        return out
    

def generate_subsequent_mask(size):
    # Upper triangular matrix (1 = allowed, 0 = masked)
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # Step 1 — torch.ones(size, size)
    # Creates a 5×5 matrix filled with 1’s:
    #  1 1 1 1 1
    #  1 1 1 1 1
    #  1 1 1 1 1
    #  1 1 1 1 1
    #  1 1 1 1 1

    # Step 2 — torch.triu(..., diagonal=1)
    # triu = take upper triangular part.
    # diagonal=1 means start one step above the diagonal.
    #  0 1 1 1 1
    #  0 0 1 1 1
    #  0 0 0 1 1
    #  0 0 0 0 1
    #  0 0 0 0 0

    # Step 3 — .bool()
    # Convert numbers to boolean:
    #  F T T T T
    #  F F T T T
    #  F F F T T
    #  F F F F T
    #  F F F F F
    # This matrix means:  
    # True = future token → mask it
    # False = allowed token


    mask = (~mask).unsqueeze(0).unsqueeze(1)   # (1,1,L,L)
    # Step A — ~mask
    # ~ flips booleans:
    #  T F F F F
    #  T T F F F
    #  T T T F F
    #  T T T T F
    #  T T T T T

    # Now:
    # True = allowed
    # False = not allowed
    # So now it becomes the look-ahead mask.

    # Example, row 3 (0-indexed):
    # T T T F F
    # Meaning:
    # Token at position 3 can see: 0,1,2
    # It cannot see: 3, 4 (future)

    # Step B — .unsqueeze(0)
    # Add batch dimension → shape becomes:
    # (1, 5, 5)

    # Step C — .unsqueeze(1)
    # Add head dimension:
    # Final shape:
    # (1, 1, 5, 5)
    # This is the required shape for MultiHeadAttention.

    # Final Mask (size=5)
    #  [[[[ T  F  F  F  F ],
    #     [ T  T  F  F  F ],
    #     [ T  T  T  F  F ],
    #     [ T  T  T  T  F ],
    #     [ T  T  T  T  T ]]]]

    # and then pytorch below function convert it Into 
    # attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
    #  0   -inf   -inf
    #  0     0    -inf
    #  0     0      0
    # the self attention needs this mask matrix to become masked multihead attention.
    # Softmax then kills the -inf values
    # exp(-inf) = 0
    # So future tokens get ZERO attention.

    return mask


# x = torch.randn(2, 5, 512)   # (batch=2, seq_len=5, d_model=512)
# # print("x: ", x)
# print("x shape: ", x.shape)
# mask = generate_subsequent_mask(5)  # (1,1,5,5)
# # print("mask", mask)
# print("mask shape: ", mask.shape)

# mha = Masked_MHA(512, 8)
# out, attn = mha(x, mask)
# # print("out: ", out, out.shape)
# # print("attn: ", attn, attn.shape)
# print("out shape: ", out.shape)
# print("attn shape: ", attn.shape)