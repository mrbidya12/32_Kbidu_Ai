# tokenizer.py
from tokenizers import Tokenizer as BpeTokenizer

class Tokenizer:
    def __init__(self, tokenizer_path="data/tokenizer/tokenizer.json"):
        self.tokenizer = BpeTokenizer.from_file(tokenizer_path)
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    def encode(self, text):
        ids = self.tokenizer.encode(text).ids
        return [self.bos_id] + ids + [self.eos_id]

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
        return self.tokenizer.decode(ids)
