# dpo_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import random

class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    prompt, chosen, rejected = parts
                    self.data.append((prompt, chosen, rejected))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, chosen, rejected = self.data[idx]
        # Encode
        prompt_ids = self.tokenizer.encode(prompt)
        chosen_ids = prompt_ids + self.tokenizer.encode(chosen)[1:]  # remove BOS
        rejected_ids = prompt_ids + self.tokenizer.encode(rejected)[1:]
        # Pad
        if len(chosen_ids) > self.max_len:
            chosen_ids = chosen_ids[:self.max_len]
        else:
            chosen_ids += [self.tokenizer.pad_id] * (self.max_len - len(chosen_ids))
        if len(rejected_ids) > self.max_len:
            rejected_ids = rejected_ids[:self.max_len]
        else:
            rejected_ids += [self.tokenizer.pad_id] * (self.max_len - len(rejected_ids))
        return torch.tensor(chosen_ids, dtype=torch.long), torch.tensor(rejected_ids, dtype=torch.long)

class DPOTrainer:
    def __init__(self, model, ref_model, train_loader, config, device, beta=0.1, lr=1e-5):
        self.model = model
        self.ref_model = ref_model
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.beta = beta
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def compute_dpo_loss(self, chosen_logps, rejected_logps):
        # DPO loss: -log(sigmoid(beta * (chosen - rejected)))
        diff = chosen_logps - rejected_logps
        loss = -F.logsigmoid(self.beta * diff).mean()
        return loss

    def train_step(self, batch_chosen, batch_rejected):
        batch_chosen = batch_chosen.to(self.device)
        batch_rejected = batch_rejected.to(self.device)
        
        # Forward pass on chosen
        logits_chosen = self.model(batch_chosen)
        logits_ref_chosen = self.ref_model(batch_chosen)
        
        # Forward pass on rejected
        logits_rejected = self.model(batch_rejected)
        logits_ref_rejected = self.ref_model(batch_rejected)
        
        # Compute log probabilities (ignoring padding)
        logps_chosen = F.log_softmax(logits_chosen, dim=-1).gather(-1, batch_chosen.unsqueeze(-1)).squeeze(-1).sum(-1)
        logps_ref_chosen = F.log_softmax(logits_ref_chosen, dim=-1).gather(-1, batch_chosen.unsqueeze(-1)).squeeze(-1).sum(-1)
        
        logps_rejected = F.log_softmax(logits_rejected, dim=-1).gather(-1, batch_rejected.unsqueeze(-1)).squeeze(-1).sum(-1)
        logps_ref_rejected = F.log_softmax(logits_ref_rejected, dim=-1).gather(-1, batch_rejected.unsqueeze(-1)).squeeze(-1).sum(-1)
        
        # DPO loss
        loss = self.compute_dpo_loss(logps_chosen - logps_ref_chosen, logps_rejected - logps_ref_rejected)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_chosen, batch_rejected in self.train_loader:
                loss = self.train_step(batch_chosen, batch_rejected)
                total_loss += loss
            print(f"DPO Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader):.4f}")

# Usage:
# dpo_dataset = PreferenceDataset("data/preference.txt", tokenizer)
# dpo_loader = DataLoader(dpo_dataset, batch_size=4, shuffle=True)
# ref_model = KbiduForCausalLM(config)  # copy of base model (frozen)
# dpo_trainer = DPOTrainer(model, ref_model, dpo_loader, config, device)
# dpo_trainer.train(epochs=3)