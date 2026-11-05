# loss.py
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, pad_idx=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, logits, targets):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()
        return self.criterion(shift_logits.view(-1, shift_logits.size(-1)),
                              shift_targets.view(-1))
