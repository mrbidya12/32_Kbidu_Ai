# lora_finetune.py
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader

class LoRAFinetuner:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            target_modules=["W_q", "W_k", "W_v", "W_o", "w1", "w2", "w3"],
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()
        
    def train(self, texts, epochs=3, batch_size=4, lr=1e-4):
        # Prepare dataset
        dataset = Dataset.from_dict({"text": texts})
        
        # Tokenize (simplified)
        def tokenize(examples):
            # Assuming we have a tokenizer
            return {"input_ids": tokenizer.encode(examples["text"])}
        
        tokenized = dataset.map(tokenize, batched=False)
        tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 10)
        
        # DataLoader
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        loader = DataLoader(tokenized, batch_size=batch_size, collate_fn=collator)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            print(f"LoRA Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
        # Save adapter
        self.model.save_pretrained("checkpoints/lora_adapter")
        
# Usage:
# from peft import PeftModel
# model = KbiduForCausalLM(config).to(device)
# finetuner = LoRAFinetuner(model, config, device)
# finetuner.train(["cyber Q&A data...", ...])