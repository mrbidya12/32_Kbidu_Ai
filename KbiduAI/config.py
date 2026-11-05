# config.py

class KbiduTestConfig:
    """Quick test mode – trains in ~2 minutes on a single GPU."""
    vocab_size = 1000
    d_model = 128
    num_layers = 4
    num_heads = 4
    num_kv_heads = 2               # Grouped‑Query Attention
    d_ff = 256                     # SwiGLU expansion factor
    max_seq_len = 64
    dropout = 0.1
    layer_norm_eps = 1e-6
    use_gradient_checkpointing = True

    batch_size = 4
    learning_rate = 3e-4
    weight_decay = 0.1
    warmup_steps = 10
    total_steps = 50
    gradient_accumulation_steps = 2

    # NEW: Use streaming for memory efficiency (recommended for large datasets)
    use_streaming = True


class Kbidu30BConfig:
    """30B parameter model – production configuration."""
    vocab_size = 50000
    d_model = 6656
    num_layers = 60
    num_heads = 52
    num_kv_heads = 8               # GQA
    d_ff = 17920                   # ~ (8/3) * d_model
    max_seq_len = 2048
    dropout = 0.1
    layer_norm_eps = 1e-6
    use_gradient_checkpointing = True

    batch_size = 2                 # per GPU
    learning_rate = 3e-4
    weight_decay = 0.1
    warmup_steps = 2000
    total_steps = 100000
    gradient_accumulation_steps = 8

    # NEW: Use streaming for large-scale training (prevents OOM)
    use_streaming = True