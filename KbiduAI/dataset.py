# dataset.py
import torch
from torch.utils.data import Dataset, IterableDataset
import os
import random
from datasets import load_dataset as hf_load_dataset


class TextDataset(Dataset):
    """
    Simple dataset for a single text file (used for validation).
    """
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            if hasattr(self.tokenizer, 'pad_id'):
                pad_id = self.tokenizer.pad_id
            elif hasattr(self.tokenizer, 'pad_token_id'):
                pad_id = self.tokenizer.pad_token_id
            else:
                pad_id = 0
            ids = ids + [pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class MultiSourceTextDataset(Dataset):
    """
    Flexible dataset that can load from:
    - Local text files
    - Hugging Face datasets
    """
    def __init__(self, tokenizer, max_len=512, local_files=None, hf_datasets=None, shuffle=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.shuffle = shuffle
        self.data = []

        if local_files:
            for file_path in local_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        self.data.extend(lines)
                        print(f"   Loaded {len(lines)} from {file_path}")

        if hf_datasets:
            for ds_name in hf_datasets:
                try:
                    ds = hf_load_dataset(ds_name, split="train", trust_remote_code=True)
                    if "content" in ds.column_names:
                        texts = [item["content"] for item in ds]
                    elif "text" in ds.column_names:
                        texts = [item["text"] for item in ds]
                    else:
                        col = ds.column_names[0]
                        texts = [item[col] for item in ds]
                    self.data.extend(texts)
                    print(f"   Loaded {len(texts)} from {ds_name}")
                except Exception as e:
                    print(f"   ⚠️ Error loading {ds_name}: {e}")

        if not self.data:
            print("⚠️ No data loaded!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            if hasattr(self.tokenizer, 'pad_id'):
                pad_id = self.tokenizer.pad_id
            elif hasattr(self.tokenizer, 'pad_token_id'):
                pad_id = self.tokenizer.pad_token_id
            else:
                pad_id = 0
            ids = ids + [pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


class StreamingMultiSourceDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for huge datasets.
    """
    def __init__(self, tokenizer, max_len=512, file_paths=None, shuffle=True, buffer_size=10000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.file_paths = file_paths or []
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self):
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8') as f:
                buffer = []
                for line in f:
                    line = line.strip()
                    if line:
                        if self.shuffle:
                            buffer.append(line)
                            if len(buffer) >= self.buffer_size:
                                random.shuffle(buffer)
                                for item in buffer:
                                    yield self._tokenize(item)
                                buffer = []
                        else:
                            yield self._tokenize(line)
                if buffer:
                    random.shuffle(buffer)
                    for item in buffer:
                        yield self._tokenize(item)

    def _tokenize(self, text):
        if hasattr(self.tokenizer, 'encode'):
            ids = self.tokenizer.encode(text)
        else:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            if hasattr(self.tokenizer, 'pad_id'):
                pad_id = self.tokenizer.pad_id
            elif hasattr(self.tokenizer, 'pad_token_id'):
                pad_id = self.tokenizer.pad_token_id
            else:
                pad_id = 0
            ids = ids + [pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def create_mixed_dataset(tokenizer, max_len=512, config=None):
    """
    Factory function that creates a mixed dataset.
    Uses streaming if config.use_streaming is True.
    """
    local_files = []
    for f in ["data/code_train.txt", "data/cyber_train.txt", "data/general_train.txt", "data/train.txt"]:
        if os.path.exists(f):
            local_files.append(f)

    use_streaming = config and hasattr(config, 'use_streaming') and config.use_streaming

    if use_streaming:
        return StreamingMultiSourceDataset(
            tokenizer=tokenizer,
            max_len=max_len,
            file_paths=local_files,
            shuffle=True,
            buffer_size=10000
        )
    else:
        return MultiSourceTextDataset(
            tokenizer=tokenizer,
            max_len=max_len,
            local_files=local_files,
            hf_datasets=[],
            shuffle=True
        )