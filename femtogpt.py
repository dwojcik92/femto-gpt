import os, math, random
random.seed(42)
if not os.path.exists("input.txt"):
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt", "input.txt")
docs = [l.strip() for l in open("input.txt").read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = BOS + 1
print(f"vocab size: {vocab_size}")
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad = data, 0
        self._children, self._local_grads = children, local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data ** (other - 1),))
    def log(self): return Value(math.log(self.data), (self,), (1 / self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for c in v._children: build(c)
                topo.append(v)
        build(self)
        self.grad = 1
        for v in reversed(topo):
            for c, g in zip(v._children, v._local_grads): c.grad += g * v.grad
n_embd, n_head, n_layer, block_size = 16, 4, 1, 16
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {"wte": matrix(vocab_size, n_embd), "wpe": matrix(block_size, n_embd), "lm_head": matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict.update({
        f"layer{i}.attn_wq": matrix(n_embd, n_embd),
        f"layer{i}.attn_wk": matrix(n_embd, n_embd),
        f"layer{i}.attn_wv": matrix(n_embd, n_embd),
        f"layer{i}.attn_wo": matrix(n_embd, n_embd),
        f"layer{i}.mlp_fc1": matrix(4 * n_embd, n_embd),
        f"layer{i}.mlp_fc2": matrix(n_embd, 4 * n_embd),
    })
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")
def linear(x, w): return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(logits):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]
def rmsnorm(x):
    ms = sum(v * v for v in x) / len(x)
    return [xi * (ms + 1e-5) ** -0.5 for xi in x]
def gpt(token_id, pos_id, keys, values):
    x = rmsnorm([t + p for t, p in zip(state_dict["wte"][token_id], state_dict["wpe"][pos_id])])
    for li in range(n_layer):
        xr = x
        x = rmsnorm(x)
        q, k, v = linear(x, state_dict[f"layer{li}.attn_wq"]), linear(x, state_dict[f"layer{li}.attn_wk"]), linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            qh = q[hs:hs + head_dim]
            kh = [ki[hs:hs + head_dim] for ki in keys[li]]
            vh = [vi[hs:hs + head_dim] for vi in values[li]]
            attn_logits = [sum(qh[j] * kh[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(kh))]
            aw = softmax(attn_logits)
            x_attn.extend([sum(aw[t] * vh[t][j] for t in range(len(vh))) for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f"layer{li}.attn_wo"]), xr)]
        xr = x
        x = linear(rmsnorm(x), state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = [a + b for a, b in zip(linear(x, state_dict[f"layer{li}.mlp_fc2"]), xr)]
    return linear(x, state_dict["lm_head"])
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m, v = [0.0] * len(params), [0.0] * len(params)
num_steps = 1000
for step in range(num_steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)
    keys, values, losses = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], []
    for pos_id in range(n):
        logits = gpt(tokens[pos_id], pos_id, keys, values)
        losses.append(-softmax(logits)[tokens[pos_id + 1]].log())
    loss = (1 / n) * sum(losses)
    loss.backward()
    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat, v_hat = m[i] / (1 - beta1 ** (step + 1)), v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
        p.grad = 0
    print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}")
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values, token_id, sample = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], BOS, []
    for pos_id in range(block_size):
        probs = softmax([l / temperature for l in gpt(token_id, pos_id, keys, values)])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")
