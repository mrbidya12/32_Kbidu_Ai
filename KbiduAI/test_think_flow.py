# test_think_flow.py
import torch
from config import KbiduTestConfig
from model import KbiduForCausalLM
from tokenizer import Tokenizer
from inference import Inference
import os

print("=" * 60)
print("🧠 KBIDU AI – THINK, VISUALIZE, VERIFY, ANSWER")
print("=" * 60)

# 1. Load the tiny test model
config = KbiduTestConfig()
device = torch.device("cpu")
tokenizer = Tokenizer("data/tokenizer/tokenizer.json")
model = KbiduForCausalLM(config)

checkpoint = "checkpoints/model_step_100.pt"
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print("✅ Loaded checkpoint")
else:
    print("⚠️ No checkpoint found – model is untrained (answers will be random)")

model.to(device)
model.eval()

# 2. Create inference engine (No RAG to avoid pandas/security blocks)
infer = Inference(model, tokenizer, device, retriever=None, max_new_tokens=200)

# 3. The Question
query = "A farmer has 17 sheep. All but 9 run away. How many are left?"

print("\n" + "=" * 60)
print("📝 STEP 1: USER INPUT")
print("=" * 60)
print(f"❓ Query: {query}")

# 4. Force "Think Before Answer" with reasoning hidden
print("\n" + "=" * 60)
print("🧠 STEP 2-4: INTERNAL THINKING, VISUALIZATION, VERIFICATION (HIDDEN)")
print("=" * 60)
print("🔇 [Reasoning is happening inside the model...]")

# 5. Generate with think_before_answer=True, show_reasoning=False
print("\n" + "=" * 60)
print("✅ STEP 5-6: EXTRACT & VERIFY FINAL ANSWER")
print("=" * 60)

answer = infer.generate(
    query,
    think_before_answer=True,  # Forces internal reasoning
    show_reasoning=False,      # Hides the reasoning
    use_cot=True,              # Enables step-by-step thinking
    temperature=0.7
)

print(f"\n💡 FINAL ANSWER (Verified):\n{answer}")

# 6. Optional: Show what the model "thought" (for debugging)
print("\n" + "=" * 60)
print("🔍 OPTIONAL: SHOW INTERNAL REASONING (For Debugging)")
print("=" * 60)
answer_with_reasoning = infer.generate(
    query,
    think_before_answer=True,
    show_reasoning=True,  # Show the full trace
    use_cot=True,
    temperature=0.7
)
print(f"\n📝 FULL RESPONSE (Reasoning + Answer):\n{answer_with_reasoning}")

print("\n" + "=" * 60)
print("✅ DEMONSTRATION COMPLETE")
print("=" * 60)
print("\n📌 How It Works:")
print("  1. You ask a question.")
print("  2. The model thinks internally (hidden).")
print("  3. It simulates/visualizes the answer (hidden).")
print("  4. It verifies the answer (hidden).")
print("  5. ONLY the final, verified answer is shown to you.")