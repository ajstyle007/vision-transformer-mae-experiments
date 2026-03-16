import torch
from torch import nn
import math


class Positional_Encoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()

        PE = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len).unsqueeze(-1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))

        PE[:, 0::2] = torch.sin(position * div_term)

        PE[:, 1::2] = torch.cos(position * div_term)

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        pe = PE.unsqueeze(0)
        # print("pe: ", pe)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        
        return x 





# --- test ---
# x = torch.tensor([[[1.2]*512, [1.3]*512, [1.4]*512]])  # shape: (1, 3, 512)
# print(x[0:2, 0:2])
# pe = Positional_Encoding(seq_len=3, d_model=512)
# print(pe)
# out = pe(x)
# print(x.shape, out.shape, x.shape)
# print(out[0, 0, :10])  # first 10 dims of first word


# tensor([[[1.2000, 1.2000, 1.2000,  ..., 1.2000, 1.2000, 1.2000],
#          [1.3000, 1.3000, 1.3000,  ..., 1.3000, 1.3000, 1.3000]]])
# pe:  tensor([[[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
#            0.0000e+00,  1.0000e+00],
#          [ 8.4147e-01,  5.4030e-01,  8.2186e-01,  ...,  1.0000e+00,
#            1.0366e-04,  1.0000e+00],
#          [ 9.0930e-01, -4.1615e-01,  9.3641e-01,  ...,  1.0000e+00,
#            2.0733e-04,  1.0000e+00]]])
# Positional_Encoding()
# torch.Size([1, 3, 512]) torch.Size([1, 3, 512]) torch.Size([1, 3, 512])
# tensor([1.2000, 2.2000, 1.2000, 2.2000, 1.2000, 2.2000, 1.2000, 2.2000, 1.2000,
#         2.2000])

# from above we can see that 
# we have x embddings and pe positinal embeddings and then we have thier sum
# out = x + pe
# x => tensor([[[1.2000, 1.2000
# pe => tensor([[[ 0.0000e+00,  1.0000e+00
# out => tensor([1.2000, 2.2000



# Assume we have 3 words → embeddings:
# z1, z2, z3 → each of size d_model (say 512).
# So your input embeddings matrix is of shape (seq_len=3, d_model=512).

def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    # We’re creating a matrix to store positional encodings for each token position.
    # (seq_len=3, d_model=512)
    # 3 positions, each with a vector of size 512.
    # tensor([[0., 0., 0.,  ..., 0., 0., 0.],
    #     [0., 0., 0.,  ..., 0., 0., 0.],
    #     [0., 0., 0.,  ..., 0., 0., 0.]])


    position = torch.arange(0, seq_len).unsqueeze(-1)
    # tensor([0, 1, 2]) before unsqeeze
    # after unsqeeze
    # tensor([[0],
        # [1],
        # [2]])
    # Shape → (3, 1)
    # Each row represents the position index of a token (z1, z2, z3).
    

    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
    # print(div_term.shape)
    # torch.Size([256])

    # 👉 d_model = 512
    # 👉 hum alternate karte hain — ek sin, ek cos
    # 👉 matlab half (512/2 = 256) frequencies lenge
    # 👉 har frequency se 2 dimension banenge → sin + cos

    # So final positional encoding ka shape hota hai [seq_len, 512],
    # jisme har even index pe sine values hoti hain,
    # aur har odd index pe cos values.

    # Ye dono milke har position ke liye ek unique pattern bana dete hain,
    # jisse model ko word order samajh me aata hai 🚀

    # torch.arange(0, d_model, 2) → [0, 2, 4, …]
    # ✅ Ye hi 2i ka kaam kar raha hai.

    # (-log(10000)/d_model) → divide by d_model and take negative log

    # Multiplying dono → 0*(-log(10000)/d_model) = 0 for i=0

    # Multiplying by 2, 4, … → automatically scale ho jata hai



    # Apply sine to even indices, cosine to odd indices
    PE[:, 0::2] = torch.sin(position * div_term)
    # PE[:, 0::2]
    # : → all rows (all positions)
    # 0::2 → start at index 0, take every 2nd column
    # Matlab even indices → [0, 2, 4, ...
    # Ye sine values ke liye reserve kiya jata hai

    PE[:, 1::2] = torch.cos(position * div_term)
    # PE[:, 1::2]
    # : → all rows (all positions)
    # 1::2 → start at index 1, take every 2nd column
    # Matlab odd indices → [1, 3, 5, ...]
    # Ye cosine values ke liye reserve kiya jata hai

    # PE, PE.shape >> torch.Size([3, 512])

    # tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
    #       0.0000e+00,  1.0000e+00],
    #     [ 8.4147e-01,  5.4030e-01,  8.2186e-01,  ...,  1.0000e+00,
    #       1.0366e-04,  1.0000e+00],
    #     [ 9.0930e-01, -4.1615e-01,  9.3641e-01,  ...,  1.0000e+00,
    #       2.0733e-04,  1.0000e+00]])


    # First row → position 0 → mostly sin(0)=0, cos(0)=1
    # Second row → position 1 → sine/cos of different frequencies, gradually changing
    # Third row → position 2 → values keep changing according to frequency scaling

    pe = PE.unsqueeze(0)            # add batch dimension
    # self.register_buffer('pe', pe)  # save as non-trainable buffer

    return  PE

# print(positional_encoding(3, 512))



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        # self.pe[:, ... , :] → first dimension me kuch change nahi (kyunki pe ke paas ek hi batch dim hai)
        # x.size(1) → ye input ka actual sequence length deta hai (yaha 50)
        # Toh self.pe[:, :x.size(1), :] ka shape hoga (1, 50, 512),
        # jo input ke seq_len ke hisab se truncate ho gaya (agar pe ka max_len 100 tha to ab 50 tak hi lena hai).

        return x
