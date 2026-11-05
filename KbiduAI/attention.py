# attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, rotary=None):
        batch, seq_len, _ = x.size()

        q = self.W_q(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.W_k(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.W_v(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        if rotary is not None:
            q = rotary(q, seq_len)
            k = rotary(k, seq_len)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        # Only use is_causal if mask is None, otherwise use explicit mask
        use_causal = mask is None

        attn_output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=use_causal
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_o(attn_output)
    
    