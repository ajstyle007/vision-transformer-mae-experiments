import torch
from torch import nn

# Linear → ReLU → Linear → Dropout
# x -> Linear(512→2048) -> ReLU -> Linear(2048→512) -> Dropout

class feedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(0.1)

    
    def forward(self, x): # x shape: (batch_size, seq_len, d_model)
        out = self.fc1(x)
        out1 = self.relu(out)
        out2 = self.fc2(out1)
        out3 = self.dropout(out2)    

        return out3
    

# testing 

# ffn = feedforward(d_model=512, d_ff=2048)
# x = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)
# out = ffn(x)
# print(out.shape)  # → torch.Size([2, 10, 512])