# data_preprocessor.py
import hashlib
import re
from collections import Counter

class DataPreprocessor:
    def __init__(self):
        self.seen_hashes = set()

    def deduplicate(self, texts):
        """Remove exact duplicate lines."""
        unique_texts = []
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash not in self.seen_hashes:
                self.seen_hashes.add(text_hash)
                unique_texts.append(text)
        print(f"Removed {len(texts) - len(unique_texts)} duplicates.")
        return unique_texts

    def filter_quality(self, texts, min_words=5, max_word_ratio=0.9):
        """Filter short texts and those with too many repeated words."""
        filtered = []
        for text in texts:
            words = text.split()
            if len(words) < min_words:
                continue
            # Check if > 90% of words are the same (spam)
            if len(words) > 0 and max(Counter(words).values()) / len(words) > max_word_ratio:
                continue
            filtered.append(text)
        print(f"Removed {len(texts) - len(filtered)} low-quality texts.")
        return filtered

    def remove_toxic(self, texts, toxic_keywords=None):
        """Basic toxicity filter."""
        if toxic_keywords is None:
            toxic_keywords = ["hate", "violence", "kill", "racist", "sexist", "nazi"]
        filtered = []
        for text in texts:
            lower_text = text.lower()
            if any(kw in lower_text for kw in toxic_keywords):
                continue
            filtered.append(text)
        print(f"Removed {len(texts) - len(filtered)} potentially toxic texts.")
        return filtered

    def preprocess_file(self, input_path, output_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        texts = self.deduplicate(texts)
        texts = self.filter_quality(texts)
        texts = self.remove_toxic(texts)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"Saved cleaned data to {output_path}")

# Usage:
# preprocessor = DataPreprocessor()
# preprocessor.preprocess_file("data/raw_train.txt", "data/train.txt")