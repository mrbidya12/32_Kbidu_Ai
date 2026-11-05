# streaming_dataset.py
import torch
from torch.utils.data import IterableDataset
import os

class StreamingTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, max_len, shuffle=True, buffer_size=10000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.shuffle = shuffle
        self.buffer_size = buffer_size

    def __iter__(self):
        # Stream the file line by line
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # If shuffle, read a buffer, shuffle it, yield from buffer
            if self.shuffle:
                import random
                buffer = []
                for line in f:
                    if line.strip():
                        buffer.append(line)
                        if len(buffer) >= self.buffer_size:
                            random.shuffle(buffer)
                            for item in buffer:
                                yield self._tokenize(item)
                            buffer = []
                # Yield remaining
                random.shuffle(buffer)
                for item in buffer:
                    yield self._tokenize(item)
            else:
                for line in f:
                    if line.strip():
                        yield self._tokenize(line)

    def _tokenize(self, text):
        ids = self.tokenizer.encode(text.strip())
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        else:
            ids = ids + [self.tokenizer.pad_id] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

# In your main.py, replace TextDataset with:
# from streaming_dataset import StreamingTextDataset
# train_dataset = StreamingTextDataset("data/train.txt", tokenizer, config.max_seq_len)