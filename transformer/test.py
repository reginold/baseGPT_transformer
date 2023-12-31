import pytest
import torch
import torch.nn as nn
import math

from model import InputEmbeddings
from model import PositionalEncoding
from model import LayerNormalization
from model import FeedForwardBlock
from model import MultiheadAttention
from model import ResidualConnection
from model import Encoder, EncoderBlock

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
    dropout =0.1
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

def test_multihead_attention_output_shape():
    d_model = 512
    num_heads = 8
    dropout_rate = 0.1
    seq_length = 10
    batch_size = 2

    multihead_attn = MultiheadAttention(d_model, num_heads, dropout_rate)

    # Dummy tensors for query, key, value, and mask
    query = torch.randn(batch_size, seq_length, d_model)
    key = torch.randn(batch_size, seq_length, d_model)
    value = torch.randn(batch_size, seq_length, d_model)
    mask = torch.randint(0, 2, (batch_size, 1, seq_length, seq_length)).bool()

    # Forward pass
    output = multihead_attn(query, key, value, mask)

    # Check if output shape is correct
    assert output.shape == (batch_size, seq_length, d_model)

def test_multihead_attention_divisible_d_model():
    # Test to check if d_model is divisible by number of heads
    with pytest.raises(AssertionError):
        MultiheadAttention(d_model=512, h=10, dropout=0.1) # d_model not divisible by h

def test_residual_connection_output_shape():
    residual_connection = ResidualConnection(dropout=0.1)
    dummy_sublayer = FeedForwardBlock(d_model=512, d_ff=2048, dropout=0.1)

    # Dummy input tensor
    input_tensor = torch.randn(5, 10, 512)

    # Forward pass through the residual connection
    output_tensor = residual_connection(input_tensor, dummy_sublayer)

    # Check if output shape is correct
    assert output_tensor.shape == input_tensor.shape

def test_encoder_block_output_shape():
    attention_block = MultiheadAttention(d_model=512, h=4, dropout=0.1)
    feed_forward_block = FeedForwardBlock(d_model=512, d_ff=2048, dropout=0.1)
    encoder_block = EncoderBlock(attention_block, feed_forward_block, dropout=0.1)

    # Dummy input tensor and mask
    input_tensor = torch.randn(2, 10, 512)  # batch_size, seq_length, d_model
    mask = torch.randint(0, 2, (2, 1, 10, 10)).bool()

    # Forward pass through the encoder block
    output_tensor = encoder_block(input_tensor, mask)

    # Check if output shape is correct
    assert output_tensor.shape == input_tensor.shape

def test_encoder_output_shape():
    attention_block = MultiheadAttention(d_model=512, h=4, dropout=0.1)
    feed_forward_block = FeedForwardBlock(d_model=512, d_ff=2048, dropout=0.1)
    encoder_layers = nn.ModuleList([EncoderBlock(attention_block, feed_forward_block, dropout=0.1) for _ in range(3)])
    encoder = Encoder(encoder_layers)

    # Dummy input tensor and mask
    input_tensor = torch.randn(2, 10, 512)  # batch_size, seq_length, d_model
    mask = torch.randint(0, 2, (2, 1, 10, 10)).bool()

    # Forward pass through the encoder
    output_tensor = encoder(input_tensor, mask)

    # Check if output shape is correct
    assert output_tensor.shape == input_tensor.shape
