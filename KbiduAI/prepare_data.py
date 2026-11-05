# prepare_data.py
import os
import random
from datasets import load_dataset

print("=" * 60)
print("📥 DOWNLOADING TRAINING DATA FOR KBIDU AI")
print("=" * 60)

# Create data folder
os.makedirs("data", exist_ok=True)

# ------------------------------------------------------------
# 1. DOWNLOAD CODE DATA (StarCoder - Python)
# ------------------------------------------------------------
print("\n📦 Downloading code data (StarCoder Python subset)...")
try:
    dataset = load_dataset("bigcode/starcoderdata", data_dir="data/python", split="train", trust_remote_code=True)
    # Take first 50,000 samples for testing (remove limit for full training)
    dataset = dataset.select(range(min(50000, len(dataset))))
    
    code_texts = []
    for item in dataset:
        content = item.get("content", "")
        if content and len(content) > 50:
            code_texts.append(content)
    
    random.shuffle(code_texts)
    split = int(len(code_texts) * 0.9)
    
    with open("data/code_train.txt", "w", encoding="utf-8") as f:
        for t in code_texts[:split]:
            f.write(t + "\n\n")
    
    with open("data/code_valid.txt", "w", encoding="utf-8") as f:
        for t in code_texts[split:]:
            f.write(t + "\n\n")
    
    print(f"   ✅ Code data: {len(code_texts[:split])} train, {len(code_texts[split:])} valid")
except Exception as e:
    print(f"   ⚠️ Error downloading code data: {e}")

# ------------------------------------------------------------
# 2. DOWNLOAD CYBERSECURITY DATA (Primus-Instruct)
# ------------------------------------------------------------
print("\n📦 Downloading cybersecurity data (Primus-Instruct)...")
try:
    dataset = load_dataset("TrendMicro/Primus-Instruct", split="train", trust_remote_code=True)
    dataset = dataset.select(range(min(5000, len(dataset))))  # Limit for testing
    
    cyber_texts = []
    for item in dataset:
        text = item.get("text", "")
        if text:
            cyber_texts.append(text)
    
    random.shuffle(cyber_texts)
    split = int(len(cyber_texts) * 0.9)
    
    with open("data/cyber_train.txt", "w", encoding="utf-8") as f:
        for t in cyber_texts[:split]:
            f.write(t + "\n\n")
    
    with open("data/cyber_valid.txt", "w", encoding="utf-8") as f:
        for t in cyber_texts[split:]:
            f.write(t + "\n\n")
    
    print(f"   ✅ Cyber data: {len(cyber_texts[:split])} train, {len(cyber_texts[split:])} valid")
except Exception as e:
    print(f"   ⚠️ Error downloading cybersecurity data: {e}")

# ------------------------------------------------------------
# 3. DOWNLOAD GENERAL TEXT (CC-News)
# ------------------------------------------------------------
print("\n📦 Downloading general text (CC-News)...")
try:
    dataset = load_dataset("cc_news", split="train", trust_remote_code=True)
    dataset = dataset.select(range(min(10000, len(dataset))))  # Limit for testing
    
    general_texts = []
    for item in dataset:
        text = item.get("text", "")
        if text:
            general_texts.append(text)
    
    random.shuffle(general_texts)
    split = int(len(general_texts) * 0.9)
    
    with open("data/general_train.txt", "w", encoding="utf-8") as f:
        for t in general_texts[:split]:
            f.write(t + "\n\n")
    
    with open("data/general_valid.txt", "w", encoding="utf-8") as f:
        for t in general_texts[split:]:
            f.write(t + "\n\n")
    
    print(f"   ✅ General text: {len(general_texts[:split])} train, {len(general_texts[split:])} valid")
except Exception as e:
    print(f"   ⚠️ Error downloading general text: {e}")

# ------------------------------------------------------------
# 4. COMBINE VALIDATION FILES
# ------------------------------------------------------------
print("\n📦 Combining validation files into valid.txt...")
valid_files = ["data/code_valid.txt", "data/cyber_valid.txt", "data/general_valid.txt"]
combined = []
for f in valid_files:
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as file:
            combined.extend(file.readlines())

random.shuffle(combined)
with open("data/valid.txt", "w", encoding="utf-8") as out:
    for line in combined:
        out.write(line)

print(f"   ✅ combined {len(combined)} validation samples into valid.txt")

# ------------------------------------------------------------
# 5. CREATE A SMALL TRAIN.TXT (FALLBACK)
# ------------------------------------------------------------
print("\n📦 Creating fallback train.txt from combined data...")
train_files = ["data/code_train.txt", "data/cyber_train.txt", "data/general_train.txt"]
train_combined = []
for f in train_files:
    if os.path.exists(f):
        with open(f, "r", encoding="utf-8") as file:
            train_combined.extend(file.readlines())

random.shuffle(train_combined)
with open("data/train.txt", "w", encoding="utf-8") as out:
    for line in train_combined:
        out.write(line)

print(f"   ✅ created train.txt with {len(train_combined)} samples")

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("✅ DATA PREPARATION COMPLETE!")
print("=" * 60)
print("\n📁 Your data files are in the 'data/' folder:")
print("   - code_train.txt")
print("   - cyber_train.txt")
print("   - general_train.txt (if downloaded)")
print("   - valid.txt (combined validation)")
print("   - train.txt (combined training - fallback)")
print("\n🚀 Next step: Run 'python train_tokenizer.py'")
print("=" * 60)