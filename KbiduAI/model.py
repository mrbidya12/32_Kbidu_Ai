# model.py
import torch
import torch.nn as nn
from embedding import Embedding
from positional_encoding import RotaryEmbedding
from transformer_block import TransformerBlock
from layernorm import RMSNorm

class KbiduForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(config.d_model // config.num_heads, config.max_seq_len)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.num_kv_heads,
                config.dropout,
                config.layer_norm_eps
            ) for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.use_gradient_checkpointing:
            self.apply(self._enable_gradient_checkpointing)

    def _enable_gradient_checkpointing(self, module):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = True

    def forward(self, input_ids, mask=None):
        x = self.embed(input_ids)
        if mask is None:
            seq_len = input_ids.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            mask = mask.view(1, 1, seq_len, seq_len)

        for block in self.blocks:
            if self.training and self.config.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, mask, self.rotary, use_reentrant=False)
            else:
                x = block(x, mask, self.rotary)

        x = self.final_norm(x)
        return self.lm_head(x)
