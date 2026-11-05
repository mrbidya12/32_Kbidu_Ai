# positional_encoding.py
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

    def forward(self, x, seq_len=None):
        # x: (batch, seq_len, num_heads, head_dim)
        seq_len = seq_len or x.size(1)
        head_dim = x.size(-1)
        
        # Use the actual head dimension
        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inv_freq = inv_freq.to(x.device)
        t = torch.arange(seq_len, device=x.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        cos = freqs.cos().unsqueeze(1)
        sin = freqs.sin().unsqueeze(1)
        
        half = head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        x_rot = x1 * cos - x2 * sin
        x_pass = x1 * sin + x2 * cos
        return torch.cat((x_rot, x_pass), dim=-1)