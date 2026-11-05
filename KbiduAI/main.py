# main.py
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from config import KbiduTestConfig
from tokenizer import Tokenizer
from dataset import create_mixed_dataset, TextDataset
from model import KbiduForCausalLM
from trainer import Trainer
from inference import Inference
from retriever import Retriever
from utils import set_seed, count_parameters, ensure_directories
import os

def build_retriever():
    retriever = Retriever()
    texts = []
    data_files = ["data/train.txt", "data/code_train.txt", "data/cyber_train.txt"]
    for f in data_files:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as file:
                texts.append(file.read())
    retriever.build_index(texts)
    retriever.save_index("faiss_index")
    return retriever

def main():
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    set_seed(42)
    ensure_directories()

    config = KbiduTestConfig()

    # Load tokenizer
    tokenizer = Tokenizer("data/tokenizer/tokenizer.json")

    # Create mixed dataset (can be streaming or in-memory)
    train_dataset = create_mixed_dataset(
        tokenizer=tokenizer,
        max_len=config.max_seq_len,
        config=config,
    )

    # Validation dataset (standard non-iterable)
    valid_dataset = TextDataset("data/valid.txt", tokenizer, config.max_seq_len)

    # ============================================================
    # DATA LOADERS – FIXED FOR ITERABLE DATASET
    # ============================================================
    # Training loader: if train_dataset is an IterableDataset, we cannot use sampler/shuffle=True
    if isinstance(train_dataset, IterableDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,          # Must be False for IterableDataset
            num_workers=0,          # 0 is safe for streaming on Windows
            pin_memory=True,
        )
    else:
        # For in-memory datasets, we can use sampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=0,
            pin_memory=True,
        )

    # Validation loader (TextDataset is not iterable)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        shuffle=False,
        num_workers=0,
    )

    # Model
    model = KbiduForCausalLM(config)
    if rank == 0:
        print(f"Total parameters: {count_parameters(model):,}")

    trainer = Trainer(model, train_loader, valid_loader, config, device, rank)

    # Training
    for epoch in range(2):
        if rank == 0:
            print(f"\n--- Epoch {epoch+1} ---")
        if isinstance(train_dataset, IterableDataset):
            # IterableDataset does not support set_epoch, skip
            pass
        else:
            if train_sampler:
                train_sampler.set_epoch(epoch)
        trainer.train_epoch()
        trainer.validate()

    # Inference
    if rank == 0:
        retriever = None
        if not os.path.exists("faiss_index.faiss"):
            retriever = build_retriever()
        else:
            retriever = Retriever()
            retriever.load_index("faiss_index")

        infer = Inference(model, tokenizer, device, retriever=retriever, max_new_tokens=50)
        test_query = "Explain the concept of zero-day vulnerability."
        print(f"\nTest Query: {test_query}")
        answer = infer.generate(test_query, use_cot=True)
        print(f"Answer: {answer}")

        sc_answer = infer.generate_with_self_consistency(test_query, n=3)
        print(f"Self-consistent answer: {sc_answer}")

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()