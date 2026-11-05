# test_thinking_simple.py
import torch
from config import KbiduTestConfig
from model import KbiduForCausalLM
from tokenizer import Tokenizer
from inference import Inference
import os

def test_thinking():
    print("=" * 60)
    print("🧠 KBIDU AI – THINKING TEST (No RAG)")
    print("=" * 60)

    config = KbiduTestConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = Tokenizer("data/tokenizer/tokenizer.json")
    model = KbiduForCausalLM(config)
    checkpoint = "checkpoints/model_step_100.pt"
    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"✅ Loaded checkpoint: {checkpoint}")
    else:
        print("⚠️ No checkpoint found – using untrained model (results will be random)")

    model.to(device)
    model.eval()

    # No retriever – we are testing pure reasoning
    infer = Inference(model, tokenizer, device, retriever=None, max_new_tokens=150)

    # Test 1
    print("\n" + "=" * 60)
    print("🧪 TEST 1: Step‑by‑Step Reasoning")
    print("=" * 60)
    query = "A farmer has 17 sheep. All but 9 run away. How many are left? Explain step by step."
    print(f"\n❓ Question: {query}")
    answer = infer.generate(query, use_cot=True)
    print(f"\n💡 Answer:\n{answer}")

    # Test 2
    print("\n" + "=" * 60)
    print("🧪 TEST 2: Multi‑Step Planning")
    print("=" * 60)
    query = "You are a cybersecurity analyst. A server is compromised. List the 5 steps you would take in order to contain the incident."
    print(f"\n❓ Question: {query}")
    answer = infer.generate(query, use_cot=True)
    print(f"\n💡 Answer:\n{answer}")

    # Test 3
    print("\n" + "=" * 60)
    print("🧪 TEST 3: Self‑Consistency (Order of Operations)")
    print("=" * 60)
    query = "What is 2 + 2 * 3? (Remember order of operations)"
    print(f"\n❓ Question: {query}")
    sc_answer = infer.generate_with_self_consistency(query, n=3)
    print(f"\n💡 Self‑consistent answer: {sc_answer}")

    # Test 4
    print("\n" + "=" * 60)
    print("🧪 TEST 4: Counter‑Intuitive Logic")
    print("=" * 60)
    query = "If a clock shows 3:15, what is the angle between the hour and minute hands? Explain your reasoning."
    print(f"\n❓ Question: {query}")
    answer = infer.generate(query, use_cot=True)
    print(f"\n💡 Answer:\n{answer}")

    # Test 5
    print("\n" + "=" * 60)
    print("🧪 TEST 5: Refusal / I Don't Know")
    print("=" * 60)
    query = "What is the exact chemical composition of the alien metal in the movie 'The Arrival'?"
    print(f"\n❓ Question: {query}")
    answer = infer.generate(query, use_cot=False)
    print(f"\n💡 Answer:\n{answer}")

    print("\n" + "=" * 60)
    print("✅ THINKING TESTS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_thinking()
