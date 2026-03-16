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

# [7, 1542, 98] => "I Love you"
# token 7 - I 
# token 1542 - Love
# token 98 - you

x = prepare_encoder_input([7, 1542, 98])
print(x.shape)      # (1, 3, 512)

# Meaning:
# 1 → batch size
# 3 → sequence length (["I", "love", "you"])
# 512 → d_model (original transformer dimension)


# INPUT → EMBEDDING → POSITIONAL ENCODING → ENCODER
# Input to encoder always shape of this
# (batch_size, seq_len, d_model)

# 1️⃣ "AI is transforming the world."
# 2️⃣ "Deep learning models are powerful."
# 3️⃣ "Transformers use attention mechanisms."
# 4️⃣ "Neural networks learn patterns from data."

# We will feed all 4 sentences to the Transformer together.
# batch_size = 4
# seq_len (number of tokens per sentence)
# Maximum input length that your encoder will accept.

# Sentence 1:
# "AI is transforming the world."

# Tokens =
# ["AI", "is", "transforming", "the", "world"]
# → seq_len = 5

# Sentence 2:
# "Deep learning models are powerful."

# Tokens =
# ["Deep", "learning", "models", "are", "powerful"]
# → seq_len = 5


# Sentence 3:
# "Transformers use attention mechanisms."

# Tokens =
# ["Transformers", "use", "attention", "mechanisms"]
# → seq_len = 4
# But we need same length for all sentences in a batch → so pad it:

# ["Transformers", "use", "attention", "mechanisms", "<PAD>"]
# → seq_len = 5

# Sentence 4:
# "Neural networks learn patterns from data."
# Tokens =
# ["Neural", "networks", "learn", "patterns", "from", "data"]

# → seq_len = 6
# But max length is 6, so all others must match:
# Pad the rest:
# Sentence 1 → 5 tokens → add 1 pad
# Sentence 2 → 5 tokens → add 1 pad
# Sentence 3 → 4 tokens → add 2 pads
# Sentence 4 → 6 tokens → no pad

# seq_len = 6

# d_model (embedding dimension)
# Let’s assume each token is converted to a vector of size 10 (small example).
# d_model = 10

# Final Input Tensor to Transformer
# (batch_size, seq_len, d_model)
# [4, 6, 10]


class Encoder_block(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.1):
        super().__init__()
        self.ffn = feedforward(d_model, d_ff)
        self.multi_att = MultiHeadAttention(d_model, num_heads) #d_model >> embed_dim
        self.norm_layer1 = nn.LayerNorm(d_model)
        self.norm_layer2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # x → [MHA] → mha_out
        # x + Dropout(mha_out) → Norm → out1

        # out1 → [FFN] → ffn_out
        # out1 + Dropout(ffn_out) → Norm → out2

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
