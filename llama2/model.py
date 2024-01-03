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


def precompute_theta_pos_freq():
    pass


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
