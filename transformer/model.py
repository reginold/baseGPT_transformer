import math
import torch
import torch.nn as nn


#############################
# coding from the transformer paper: https://arxiv.org/pdf/1706.03762.pdf 
# 1. inputEmbedding
# 2. PositionalEncoding
# 3. LayerNormalization
# 4. FeedForward 


class InputEmbeddings(nn.Module):

    def __inin__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) 


    # seq_len = 5
    # tensor = torch.arange(0, seq_len, dtype=torch.float)
    # print(tensor)
    # # Output: tensor([0., 1., 2., 3., 4.])

    # tensor = tensor.unsqueeze(1)
    # print(tensor)
    # # Output: tensor([[0.],
    # #                 [1.],
    # #                 [2.],
    # #                 [3.],
    # #                 [4.]])
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create vector shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(Flase)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super.__init__()
        self.d_model = 512
        self.d_ff = 2048

        self.fc1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(self.dropout(x))
        



