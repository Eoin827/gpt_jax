import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import nn, random, vmap

with open("input.txt") as f:
    input_str = f.read()

input_str[:4]
chars = sorted(list(set(input_str)))
"".join((chars))


stoi = {c: i for i, c in enumerate(chars)}
itos = {c: i for i, c in stoi.items()}
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([itos[c] for c in x])
eval_iters = 200
eval_interval = 300
max_iters = 3000
block_size = 8
batch_size = 4
lr = 1e-3
batch_size = 32
NUM_STEP = 3000
n_emb = 32

data = jnp.array(encode(input_str), dtype=jnp.int32)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


key = random.PRNGKey(1337)


def get_batch(split, key):
    data = train_data if split == "train" else val_data
    ix = random.randint(key, (batch_size,), 0, len(data) - block_size)
    key, subkey = random.split(key)
    x = jnp.stack([data[i : i + block_size] for i in ix])
    y = jnp.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train", key)


# @jax.jit
# can speed this up later idk
def estimate_loss(model, key):
    out = {}
    # eval model or smthn
    for split in ["train", "val"]:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, key)
            key, subkey = random.split(key)
            logits = vmap(model)(x)
            B, T, C = logits.shape
            logits = jnp.reshape(logits, (B * T, C))
            yb = jnp.reshape(y, B * T)
            loss = jnp.mean(
                optax.losses.softmax_cross_entropy_with_integer_labels(logits, yb)
            )
            losses = losses.at[k].set(loss)
        out[split] = losses.mean()
    return out


class SelfAttention(eqx.Module):
    K: eqx.nn.Linear
    Q: eqx.nn.Linear
    V: eqx.nn.Linear

    def __init__(self, head_size, key):
        key, subkey = random.split(key)
        self.K = eqx.nn.Linear(n_emb, head_size, use_bias=False, key=key)
        key, subkey = random.split(key)
        self.Q = eqx.nn.Linear(n_emb, head_size, use_bias=False, key=key)
        key, subkey = random.split(key)
        self.V = eqx.nn.Linear(n_emb, head_size, use_bias=False, key=key)

    def __call__(self, x):
        keys = self.K(x)[:, None]
        queries = self.Q(x)[:, None]
        scores = queries @ keys.T
        # print("v1", scores.shape)
        scores = scores * keys.shape[-1] ** -0.5
        # print("yo", x.shape, scores.shape, queries.shape, keys.shape)
        tril = jnp.tril(scores) + jnp.triu(jnp.full_like(scores, float("-inf")), k=1)

        softed = jax.nn.softmax(tril)
        return softed @ self.V(x)


class BigramLanguageModel(eqx.Module):
    token_embedding_table: eqx.nn.Embedding
    position_embedding_table: eqx.nn.Embedding
    linear_proj: eqx.nn.Linear
    attention: SelfAttention

    def __init__(self, vocab_size, key):
        key, subkey = random.split(key)
        self.token_embedding_table = eqx.nn.Embedding(vocab_size, n_emb, key=key)
        key, subkey = random.split(key)
        self.position_embedding_table = eqx.nn.Embedding(block_size, n_emb, key=key)
        key, subkey = random.split(key)
        self.linear_proj = eqx.nn.Linear(n_emb, vocab_size, key=key)
        self.attention = SelfAttention(n_emb, key)

    def __call__(self, idx):
        (T,) = idx.shape
        logits = vmap(self.token_embedding_table)(idx)
        pos_enc = vmap(self.position_embedding_table)(jnp.arange(0, T))
        logits = logits + pos_enc
        logits = vmap(self.attention)(logits)
        logits = vmap(self.linear_proj)(logits)
        return logits

    def generate(
        self, idx, max_tokens_size, key
    ):  # maybe should return new key or smthing idk
        for _ in range(max_tokens_size):
            logits = vmap(self)(idx[:, -block_size:])
            logits = logits[:, -1, :]
            idx_next = jax.random.categorical(key, logits, axis=-1)[:, None]
            key, subkey = random.split(key)
            idx = jnp.concat((idx, idx_next), axis=1)
        return idx


@jax.jit
@jax.grad
def loss(model, xb, yb):
    logits = vmap(model)(xb)
    B, T, C = logits.shape
    logits = jnp.reshape(logits, (B * T, C))
    yb = jnp.reshape(yb, B * T)
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits, yb))
    return loss


hello = BigramLanguageModel(vocab_size=65, key=key)
loss_val = loss(hello, xb, yb)
optimiser = optax.adam(learning_rate=lr)
opt_state = optimiser.init(hello)

for i in range(NUM_STEP):
    if i % eval_interval == 0:
        lossses = estimate_loss(hello, key)
        key, subkey = random.split(key)
        print(f"step: {i}")
        print(lossses)

    xb, yb = get_batch("train", key)
    key, subkey = random.split(key)
    grad = loss(hello, xb, yb)
    updates, opt_state = optimiser.update(grad, opt_state)
    hello = eqx.apply_updates(hello, updates)


print(
    decode(
        hello.generate(
            idx=jnp.zeros((1, 1), dtype=jnp.int32), max_tokens_size=200, key=key
        )[0].tolist()
    )
)
