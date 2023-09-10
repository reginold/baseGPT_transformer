# read the data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset in chars: ", len(text))

# analyze char data
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

# create a mapping from chars to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take string, output list of integer
decode = lambda l: "".join([itos[i] for i in l])  # decoder: tale a list of integers, output a string

print(encode("hello world"))
print(decode(encode("hello world")))

# encode the all chars and store it into a torch.Tensor
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])  # the sentences in txt will be looked like the tensor

# split data into train and test data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
# print(train_data[:block_size+1])

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

torch.manual_seed(1337)  # to keep the training data same,reproduce the model
batch_size = 4  # how many indepent sequences will process in parallel
block_size = 8  # what is the maximum context length for prediction


def get_batch(split):
    # generate a small batch of data of input x and target y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("inputs: ")
print(xb.shape)
print(xb)
print("targets: ")
print(yb.shape)
print(yb)

print("-------")

for b in range(batch_size):  # batch dimension
    for t in range(block_size):  # time dimension
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"when input is {context.tolist()} the target: {target}")

print(xb)  # output to the transformer
print("########################")

################################
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both(B, T) tensor of integers, C is the prediction of (B, T)
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # how well we predict the num

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
            # print(f"logits shape: {logits.shape}")  # Debugging print
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # print(f"probs shape: {probs.shape}")
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel(vocab_size)
out, loss = model(xb, yb)
print(out.shape)
print(loss)
# create the random model
print(decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for steps in range(100):
    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
# create the  model
print(
    decode(model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist())
)  # get the better model than before, but it is also not good model


# The mathematical trick in self-attention
# consider the following toy example:

torch.manual_seed(1337)
B, T, C = 4, 8, 2  # batch, time, channels
x = torch.randn(B, T, C)
print(x.shape)

# let x[b, t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0])
print(xbow[0])
