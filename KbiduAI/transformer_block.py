# transformer_block.py
import torch.nn as nn
from attention import MultiHeadAttention
from feedforward import SwiGLU
from layernorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_heads=None, dropout=0.1, eps=1e-6):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, num_kv_heads, dropout)
        self.ff = SwiGLU(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model, eps)
        self.norm2 = RMSNorm(d_model, eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary=None):
        x = x + self.dropout1(self.attn(self.norm1(x), mask, rotary))
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x
