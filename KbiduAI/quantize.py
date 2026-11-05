# quantize.py
import torch
import torch.nn as nn
import bitsandbytes as bnb
from model import KbiduForCausalLM
from config import Kbidu30BConfig
import os

def replace_linear_with_4bit(module):
    """
    Recursively replace all nn.Linear layers with 4-bit quantized versions.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # Create 4-bit linear layer
            new_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16,
                compress_statistics=True,
                quant_type="fp4",  # or "nf4" for NormalFloat4 (better for weights)
            )
            # Copy the weight from the old layer (the quantizer will compress it)
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()
            
            # Replace the child module
            setattr(module, name, new_layer)
        else:
            # Recursively process submodules
            replace_linear_with_4bit(child)

def quantize_model(checkpoint_path, output_path="checkpoints/model_4bit.pt"):
    """
    Load the 30B checkpoint, quantize all Linear layers to 4-bit, and save.
    """
    print("📦 Loading configuration...")
    config = Kbidu30BConfig()
    
    print("🧠 Initializing model on CPU...")
    model = KbiduForCausalLM(config)
    
    print(f"📥 Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    print("⚡ Quantizing Linear layers to 4-bit (NF4)...")
    # Move to CPU for quantisation (safer)
    model = model.to("cpu")
    replace_linear_with_4bit(model)
    
    print(f"💾 Saving 4-bit model to {output_path}...")
    torch.save(model.state_dict(), output_path)
    
    # Save a separate config file for the quantized model
    import json
    with open("checkpoints/model_4bit_config.json", "w") as f:
        json.dump({
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "num_kv_heads": config.num_kv_heads,
            "d_ff": config.d_ff,
            "max_seq_len": config.max_seq_len,
            "vocab_size": config.vocab_size,
            "dropout": config.dropout,
            "layer_norm_eps": config.layer_norm_eps,
            "quantized": True,
            "quant_type": "nf4"
        }, f)
    
    print("✅ Quantization complete! Model saved.")

def load_4bit_model(config_path="checkpoints/model_4bit_config.json", model_path="checkpoints/model_4bit.pt"):
    """
    Helper to load the 4-bit model for inference.
    """
    import json
    from types import SimpleNamespace
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Convert dict to object
    config = SimpleNamespace(**config_dict)
    
    model = KbiduForCausalLM(config)
    model.load_state_dict(torch.load(model_path, map_location="cuda"))
    return model.to("cuda")

if __name__ == "__main__":
    # Usage: python quantize.py
    checkpoint = "checkpoints/model_step_100000.pt"  # Your full trained checkpoint
    if not os.path.exists(checkpoint):
        print(f"⚠️ Checkpoint not found at {checkpoint}")
        print("Please update the path to your trained model.")
    else:
        quantize_model(checkpoint)