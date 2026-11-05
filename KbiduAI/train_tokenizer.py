# train_tokenizer.py
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import glob
import os

def train_tokenizer(corpus_files, vocab_size=50000, save_path="data/tokenizer"):
    os.makedirs(save_path, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    )
    tokenizer.train(files=corpus_files, trainer=trainer)
    tokenizer.save(f"{save_path}/tokenizer.json")
    print(f"✅ Tokenizer saved to {save_path}")

if __name__ == "__main__":
    # Find all .txt files in data/
    files = glob.glob("data/*.txt")
    if not files:
        print("⚠️ No .txt files found in data/. Please add training data first.")
    else:
        print(f"📥 Found {len(files)} files: {files}")
        # Filter out empty files (0 bytes)
        valid_files = [f for f in files if os.path.getsize(f) > 0]
        if not valid_files:
            print("⚠️ All .txt files are empty. Please add content to your data files.")
        else:
            print(f"✅ Using {len(valid_files)} files with content: {valid_files}")
            train_tokenizer(valid_files, vocab_size=50000)
