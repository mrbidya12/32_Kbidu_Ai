# merge_lora.py
import torch
from peft import PeftModel
from model import KbiduForCausalLM
from config import Kbidu30BConfig
import os

def merge_lora(base_model_path, adapter_path, output_path):
    """Merge LoRA adapters into the base model."""
    config = Kbidu30BConfig()
    model = KbiduForCausalLM(config)
    
    # Load base model weights
    if os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path, map_location="cpu"))
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Merge and save
    merged_model = model.merge_and_unload()
    torch.save(merged_model.state_dict(), output_path)
    print(f"✅ Merged model saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    merge_lora(
        base_model_path="checkpoints/model_step_100000.pt",
        adapter_path="checkpoints/lora_adapter",
        output_path="checkpoints/model_merged.pt"
    )