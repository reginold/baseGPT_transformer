import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Local checkpoint path
model_path = "/mnt/remote/checkpoints/Llama-3.3-70B-Instruct"  

# 2. Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto")

# 3. (Optional but important for LLaMA or similar models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

# 4. Prepare your prompt
prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

# 5. Tokenize
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

start = time.time()

# 6. Generate
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    use_cache=False # Disable KV cache
)

end = time.time()
print(f"⏱ Generation took {end - start:.3f} seconds")

# 7. Decode and print
output_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(output_text)


# 8. Measure generation time with KV cache enabled
start_kv = time.time()
generation_output_kv = model.generate(
    input_ids=input_ids,
    max_new_tokens=100,
    use_cache=True  # Enable KV cache
)
end_kv = time.time()
print(f"⏱ Generation with KV cache took {end_kv - start_kv:.3f} seconds")
print("Generated Output with KV Cache:")
output_text_kv = tokenizer.decode(generation_output_kv[0], skip_special_tokens=True)
print(output_text_kv)


# Result
# Example Output:

# Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [01:03<00:00,  2.11s/it]
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# ⏱ Generation took 32.851 seconds
# Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. In the situation
# Subject: Gardening situation at the garden and explain the situation at home
# Dear Sarah,
# I am writing to apologize for the situation at home
# I am writing to you to explain the situation
# Dear Sarah,
# I am writing to you a tragic
# Dear Sarah,
# I am writing to explain the situation at home, apologize for the
# Dear Sarah for the
# Dear Sarah, I am writing to explain the situation at home,
# I am writing to
# I apologize to you, I
# The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
# Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
# ⏱ Generation with KV cache took 8.455 seconds
# Generated Output with KV Cache:
# Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. I couldn't make a mistake
# I apologize for the mistake
# I'm writing this email to Sarah for the tragic event of the gardening. Here is not being late to explain what happened to you, I apologize for mistake for. I apologize for the mistake in my to make a gardening, and that I apologize for the email to you for the event that I'm sorry to explain. I'm a mistake to make a mistake for the gardening for that I apologize. I to you for my to explain