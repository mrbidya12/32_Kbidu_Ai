# test_rag_think.py
import torch
from config import KbiduTestConfig
from model import KbiduForCausalLM
from tokenizer import Tokenizer
from inference import Inference
from retriever import Retriever
import os

print("=" * 60)
print("🧠 KBIDU AI – THINK + RAG + VERIFY")
print("=" * 60)

config = KbiduTestConfig()
device = torch.device("cpu")
tokenizer = Tokenizer("data/tokenizer/tokenizer.json")
model = KbiduForCausalLM(config)
model.load_state_dict(torch.load("checkpoints/model_step_100.pt", map_location=device))
model.to(device)
model.eval()

# Build retriever
retriever = Retriever()
if os.path.exists("faiss_index.faiss"):
    retriever.load_index("faiss_index")
else:
    print("⚠️ FAISS index not found. Create it by running main.py first.")
    retriever = None

infer = Inference(model, tokenizer, device, retriever=retriever, max_new_tokens=200)

query = "What is a zero-day vulnerability?"
answer = infer.generate(query, think_before_answer=True, show_reasoning=False)
print(f"Final answer: {answer}")