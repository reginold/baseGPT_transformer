import pytest
import torch
import torch.nn as nn
import math

from model import InputEmbeddings
from model import PositionalEncoding
from model import LayerNormalization
from model import FeedForwardBlock

def test_embeddings_output_shape():
    d_model = 512
    vocab_size = 10000
    batch_size = 16
    seq_length = 32
    model = InputEmbeddings(d_model, vocab_size)
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_length))
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (batch_size, seq_length, d_model)

def test_positional_encoding_shape():
    d_model = 512
    seq_len = 32
    dropout = 0.1
    model = PositionalEncoding(d_model, seq_len, dropout)
    input_tensor = torch.randn(seq_len, 16, d_model)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape

def test_layer_normalization_shape():
    eps = 1e-6
    model = LayerNormalization(eps)
    input_tensor = torch.randn(16, 32, 512)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape

def test_feed_forward_block_shape():
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    model = FeedForwardBlock(d_model, d_ff, dropout)
    input_tensor = torch.randn(16, 32, d_model)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == input_tensor.shape
