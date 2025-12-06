import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

# --- Configuration (Research Parameters) ---
@dataclass
class GPTConfig:
    block_size: int = 64       # Context window size (Time steps)
    max_iters: int = 500       # Training steps (Lite version for demo)
    learning_rate: float = 3e-4
    device: str = 'cpu'        # Optimized for standard hardware
    n_embd: int = 64           # Embedding dimension
    n_head: int = 4            # Number of Attention heads
    n_layer: int = 4           # Number of Transformer blocks
    dropout: float = 0.2
    batch_size: int = 32       # Batch size for training

config = GPTConfig()

# --- Synthetic Medical Dataset ---
# Simulating a specialized medical corpus for domain adaptation study
raw_text = """
Diabetes mellitus is a chronic metabolic disorder. Hypertension is defined as high blood pressure.
Asthma involves inflammation of the airways. Chronic Obstructive Pulmonary Disease (COPD) obstructs airflow.
Insulin resistance is a hallmark of Type 2 Diabetes. Myocardial infarction requires immediate attention.
""" * 500 

# Character-level Tokenization
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Data Split
data = torch.tensor(encode(raw_text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# --- Transformer Components (Built from Scratch) ---

class Head(nn.Module):
    """ Single Self-Attention Head: The 'Listening' Mechanism """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Registering the "Look-ahead" mask so the model can't cheat by seeing the future
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Calculate Attention Scores (Scaled Dot-Product)
        # Equation: Softmax(Q * K^T / sqrt(d_k)) * V
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Aggregation of Values
        out = wei @ self.value(x)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads running in parallel to capture different semantic features """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate heads along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module): 
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer Block: Communication (Attention) followed by Computation (FFWD) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        # Pre-Norm architecture (better stability)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual Connections (x + ...) prevent vanishing gradients
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MediGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config.n_embd, n_head=config.n_head) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # Simple generation loop
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training Loop ---
if __name__ == "__main__":
    print(f"--- Initializing Medi-GPT Lite Architecture ---")
    model = MediGPT()
    m = model.to(config.device)
    print(f"Model Parameters: {sum(p.numel() for p in m.parameters())}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print("Starting Pre-training loop...")
    
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
        x = torch.stack([data[i:i+config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
        return x.to(config.device), y.to(config.device)

    for iter in range(config.max_iters):
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter % 100 == 0:
            print(f"Step {iter}: Loss {loss.item():.4f}")
            
    print("✅ Training Complete.")
    
    # Test Generation
    print("\n--- Generating Medical Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))
