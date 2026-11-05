# eval_math.py
import re
from datasets import load_dataset
from inference import Inference
from tqdm import tqdm

def extract_answer(text):
    """Extract final numeric answer from text."""
    # Try to find the last number in the text
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if numbers:
        return numbers[-1]
    return None

def evaluate_gsm8k(model, tokenizer, device, num_samples=100):
    """Evaluate on GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main", split="test")
    total = 0
    correct = 0
    inference_engine = Inference(model, tokenizer, device, max_new_tokens=200)
    
    for i, example in enumerate(tqdm(dataset.select(range(num_samples)))):
        question = example["question"]
        true_answer = example["answer"].split("####")[-1].strip()
        
        # Generate with CoT
        output = inference_engine.generate(question, use_cot=True)
        pred_answer = extract_answer(output)
        
        if pred_answer and pred_answer in true_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"GSM8K Accuracy ({num_samples} samples): {accuracy:.2%}")
    return accuracy

def evaluate_math(model, tokenizer, device, num_samples=100):
    """Evaluate on MATH dataset (college math)."""
    # Similar logic for MATH dataset
    dataset = load_dataset("math_dataset", split="test")
    # ... implementation similar to above
    pass

# Usage:
# from model import KbiduForCausalLM
# model = KbiduForCausalLM(config).to(device)
# evaluate_gsm8k(model, tokenizer, device, num_samples=50)