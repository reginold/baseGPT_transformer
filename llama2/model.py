import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    m_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_freq(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """Add position-specific information to the input embeddings of the model

    Args:
        head_dim (int): _description_
        seq_len (int): _description_
        device (str): _description_
        theta (float, optional): _description_. Defaults to 10000.0.

    Returns:
        freq_complex(torch.Tensor): 
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2..."

    theta_nume = torch.arange(0, head_dim, 2).float()
    m = torch.arange(seq_len, device=device)
    freq = torch.outer(m, theta_nume).float()
    # ones_like -> This function creates a new tensor with the same shape as a given tensor, filled with ones
    # polar -> create a complex tensor where each element is a complex number with the corresponding magnitude and phase
    freq_complex = torch.polar(torch.ones_like(freq), freq)
    return freq_complex

    
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """Apply the freq_complex into the rotary embeddings

    Args:
        x (torch.Tensor): _description_
        freqs_complex (torch.Tensor): _description_
        device (str): _description_

    Returns:
        _type_: _description_
    """
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """Caculate the RMSNorm layer

        Args:
            dim (int): _description_
            eps (float, optional): _description_. Defaults to 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (batch, seq_len , dim) * (batch, seq_len, 1) = (batch, seq_len, dim)
        # The keepdim=True argument keeps the number of dimensions unchanged, 
        # which is useful for maintaining the shape for subsequent operations like element-wise division.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
            # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
            return self.weight * self._norm(x.float()).type_as(x)
    

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) -> same
        hidden = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)

        # (batch, seq_len, dim) + (batch, seq_len, dim) -> same
        output = hidden + self.feed_forward.forward(self.ffn_norm(hidden))
        return output


class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set.."

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embedding = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output == nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_freq(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len
            * 2,  # why we need to mutiplied by 2 to extend the length?
            device=args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_len)
        _, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed..."

        # (batch, seq_len) -> (batch, seq_len, dim)
        h = self.tok_embedding(tokens)

        # Retrieve the pairs (m, theta) to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
