# trainer.py
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed as dist
import os
from loss import Loss
from transformer_block import TransformerBlock

class Trainer:
    def __init__(self, model, train_loader, valid_loader, config, device, rank=0):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.device = device
        self.rank = rank

        if dist.is_initialized():
            auto_wrap_policy = transformer_auto_wrap_policy(TransformerBlock)
            self.model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
                device_id=device,
            )
        else:
            self.model.to(device)

        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=config.learning_rate,
                                     weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                              T_max=config.total_steps)
        self.criterion = Loss()
        self.scaler = GradScaler()
        self.step = 0

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            with autocast():
                logits = self.model(batch)
                loss = self.criterion(logits, batch)
                loss = loss / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.step += 1

                if self.rank == 0 and (self.step % 10 == 0 or self.step == 1):
                    current_loss = loss.item() * self.config.gradient_accumulation_steps
                    print(f"Step {self.step} | Loss: {current_loss:.4f} | LR: {self.scheduler.get_last_lr()[0]:.2e}")

                if self.rank == 0 and self.step % 50 == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    state = self.model.state_dict()
                    torch.save(state, f"checkpoints/model_step_{self.step}.pt")

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.valid_loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch)
                total_loss += loss.item()
        if self.rank == 0:
            avg_loss = total_loss / max(1, len(self.valid_loader))
            print(f"Validation Loss: {avg_loss:.4f}")
            return avg_loss
        return None
