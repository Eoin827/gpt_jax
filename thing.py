with open("input.txt") as f:
    input_str = f.read()

input_str[:4]
chars = sorted(list(set(input_str)))
"".join((chars))


stoi = {c: i for i, c in enumerate(chars)}
itos = {c: i for i, c in stoi.items()}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([itos[c] for c in x])

import torch

# here
data = torch.tensor(encode(input_str), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
block_size = 8
batch_size = 4


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")

import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):  # ig idx is x just the normal input??
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_tokens_size):
        for _ in range(max_tokens_size):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


hello = BigramLanguageModel(vocab_size=65)
logits, loss = hello(xb, yb)
print(logits.shape)
print(loss)


decode(
    hello.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_tokens_size=100)[
        0
    ].tolist()
)

optimiser = torch.optim.AdamW(hello.parameters(), lr=1e-3)

batch_size = 32
for i in range(10000):
    xb, yb = get_batch("train")
    logits, loss = hello(xb, yb)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
print(loss.item())


print(
    decode(
        hello.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_tokens_size=100)[
            0
        ].tolist()
    )
)
