# logging_utils.py
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time

class Logger:
    def __init__(self, log_dir="logs", use_wandb=False, project_name="Kbidu-30B"):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.writer = None
        self.step = 0

        if use_wandb:
            try:
                import wandb
                wandb.init(project=project_name)
                self.wandb = wandb
                print("✅ Weights & Biases initialized.")
            except ImportError:
                print("⚠️ wandb not installed. Falling back to TensorBoard.")
                self.use_wandb = False

        if not self.use_wandb:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

    def log_metrics(self, metrics, step=None):
        if step is None:
            step = self.step
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
        else:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def log_histogram(self, name, tensor, step=None):
        if step is None:
            step = self.step
        if self.use_wandb:
            self.wandb.log({name: self.wandb.Histogram(tensor.detach().cpu().numpy())}, step=step)
        else:
            self.writer.add_histogram(name, tensor, step)

    def log_text(self, name, text, step=None):
        if step is None:
            step = self.step
        if self.use_wandb:
            self.wandb.log({name: text}, step=step)
        else:
            self.writer.add_text(name, text, step)

    def close(self):
        if self.use_wandb:
            self.wandb.finish()
        else:
            self.writer.close()

# Usage in trainer.py:
# from logging_utils import Logger
# logger = Logger(use_wandb=True)
# logger.log_metrics({"train_loss": loss.item()}, step=self.step)