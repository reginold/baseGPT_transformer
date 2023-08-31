# read the data
with open("input.txt", 'r', encoding="utf-8") as f:
    text = f.read()

print("length of dataset in chars: ", len(text))

# analyze char data
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# create a mapping from chars to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take string, output list of integer
decode = lambda l : ''.join([itos[i] for i in l]) # decoder: tale a list of integers, output a string

print(encode("hello world"))
print(decode(encode("hello world")))

# encode the all chars and store it into a torch.Tensor
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100]) # the sentences in txt will be looked like the tensor

# split data into train and test data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
# print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

