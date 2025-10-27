import torch
from transformers import AutoModelForCausalLM

model_path = "/mnt/remote/checkpoints/Llama-3.2-1B-Instruct"

# load Llama-3.2-1B-Instruct model 
model = AutoModelForCausalLM.from_pretrained(
   model_path,
   device_map="auto"
)
# Print all model configuration parameters
config = model.config
print("\n=== Model Configuration Parameters ===")
# Architecture parameters
print("\nArchitecture Parameters:")
# Size of the hidden layers
print(f"Hidden size: {config.hidden_size}") 
# Number of transformer blocks
print(f"Number of layers: {config.num_hidden_layers}") 
# Number of attention heads
print(f"Number of attention heads: {config.num_attention_heads}")
# Size of the MLP intermediate layer
print(f"Intermediate size: {config.intermediate_size}")  
# Tokenizer parameters
print("\nTokenizer Parameters:")
# Size of the vocabulary
print(f"Vocabulary size: {config.vocab_size}") 
# Maximum sequence length
print(f"Maximum position embeddings:{config.max_position_embeddings}") 
# Model-specific parameters
print("\nModel-specific Parameters:")
for key, value in config.to_dict().items():
   if key not in ['architectures', 'model_type', 'torch_dtype']:
       print(f"{key}: {value}")

print("\n=== Model Structure ===")
def print_module_structure(module, prefix=''):
   for name, child in module.named_children():
       # Skip certain internal modules for clarity
       if name in ['_orig_mod', 'wrapped_model']:
           continue
       # Print the current module
       print(f"{prefix}{name}: {type(child).__name__}")
       # Check if this is an attention module (generic check for any transformer)
       if "attention" in name.lower() or "attention" in type(child).__name__.lower():
           print(f"\n{prefix}  Found attention module: {name}")
           print(f"{prefix}  Type: {type(child).__name__}")
           # Print attention-specific attributes
           if hasattr(child, 'num_heads'):
               print(f"{prefix}  Number of attention heads: {child.num_heads}")
           if hasattr(child, 'head_dim'):
               print(f"{prefix}  Head dimension: {child.head_dim}")
           if hasattr(child, 'hidden_size'):
               print(f"{prefix}  Hidden size: {child.hidden_size}")
           if hasattr(child, 'rotary_emb'):
               print(f"{prefix}  Has rotary embeddings: {child.rotary_emb is not None}")
           print()  # Add blank line after attention details
       # Recursively print children
       print_module_structure(child, prefix + '  ')

print_module_structure(model)



# Result

# === Model Configuration Parameters ===

# Architecture Parameters:
# Hidden size: 2048
# Number of layers: 16
# Number of attention heads: 32
# Intermediate size: 8192

# Tokenizer Parameters:
# Vocabulary size: 128256
# Maximum position embeddings:131072

# Model-specific Parameters:
# vocab_size: 128256
# max_position_embeddings: 131072
# hidden_size: 2048
# intermediate_size: 8192
# num_hidden_layers: 16
# num_attention_heads: 32
# num_key_value_heads: 8
# hidden_act: silu
# initializer_range: 0.02
# rms_norm_eps: 1e-05
# pretraining_tp: 1
# use_cache: True
# rope_theta: 500000.0
# rope_scaling: {'factor': 32.0, 'high_freq_factor': 4.0, 'low_freq_factor': 1.0, 'original_max_position_embeddings': 8192, 'rope_type': 'llama3'}
# attention_bias: False
# attention_dropout: 0.0
# mlp_bias: False
# head_dim: 64
# return_dict: True
# output_hidden_states: False
# output_attentions: False
# torchscript: False
# use_bfloat16: False
# tf_legacy_loss: False
# pruned_heads: {}
# tie_word_embeddings: True
# chunk_size_feed_forward: 0
# is_encoder_decoder: False
# is_decoder: False
# cross_attention_hidden_size: None
# add_cross_attention: False
# tie_encoder_decoder: False
# max_length: 20
# min_length: 0
# do_sample: False
# early_stopping: False
# num_beams: 1
# num_beam_groups: 1
# diversity_penalty: 0.0
# temperature: 1.0
# top_k: 50
# top_p: 1.0
# typical_p: 1.0
# repetition_penalty: 1.0
# length_penalty: 1.0
# no_repeat_ngram_size: 0
# encoder_no_repeat_ngram_size: 0
# bad_words_ids: None
# num_return_sequences: 1
# output_scores: False
# return_dict_in_generate: False
# forced_bos_token_id: None
# forced_eos_token_id: None
# remove_invalid_values: False
# exponential_decay_length_penalty: None
# suppress_tokens: None
# begin_suppress_tokens: None
# finetuning_task: None
# id2label: {0: 'LABEL_0', 1: 'LABEL_1'}
# label2id: {'LABEL_0': 0, 'LABEL_1': 1}
# tokenizer_class: None
# prefix: None
# bos_token_id: 128000
# pad_token_id: None
# eos_token_id: [128001, 128008, 128009]
# sep_token_id: None
# decoder_start_token_id: None
# task_specific_params: None
# problem_type: None
# _name_or_path: /mnt/remote/checkpoints/Llama-3.2-1B-Instruct
# transformers_version: 4.52.3

# === Model Structure ===
# model: LlamaModel
#   embed_tokens: Embedding
#   layers: ModuleList
#     0: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     1: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     2: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     3: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     4: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     5: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     6: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     7: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     8: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     9: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     10: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     11: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     12: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     13: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     14: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#     15: LlamaDecoderLayer
#       self_attn: LlamaAttention

#         Found attention module: self_attn
#         Type: LlamaAttention
#         Head dimension: 64

#         q_proj: Linear
#         k_proj: Linear
#         v_proj: Linear
#         o_proj: Linear
#       mlp: LlamaMLP
#         gate_proj: Linear
#         up_proj: Linear
#         down_proj: Linear
#         act_fn: SiLU
#       input_layernorm: LlamaRMSNorm
#       post_attention_layernorm: LlamaRMSNorm

#         Found attention module: post_attention_layernorm
#         Type: LlamaRMSNorm

#   norm: LlamaRMSNorm
#   rotary_emb: LlamaRotaryEmbedding
# lm_head: Linear