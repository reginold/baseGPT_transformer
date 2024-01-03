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
    m_kv_heads: Optional[int]
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    pass


def precompute_theta_pos_freq(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    """Add position-specific information to the input embeddings of the model

    Args:
        head_dim (int): _description_
        seq_len (int): _description_
        device (str): _description_
        theta (float, optional): _description_. Defaults to 10000.0.

    Returns:
        freq_complex(tensor): 
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

class EncoderBlock(nn.Module):
    pass


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
