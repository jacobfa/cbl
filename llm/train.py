import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

###############################################################################
# 1) CONFIGURATION
###############################################################################
BATCH_SIZE = 4
BLOCK_SIZE = 128   # Sequence length for each training example
EMBED_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
FFN_DIM = 512
DROPOUT = 0.1
LR = 3e-4
EPOCHS = 1        # Increase for real training
PRINT_EVERY = 100 # Steps between printing progress info
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"  # or "wikitext-2-v1" for smaller dataset

def main():
    ###############################################################################
    # 2) INITIALIZE DISTRIBUTED
    ###############################################################################
    # This script expects to be launched via:
    #    torchrun --nproc_per_node=<NUM_GPUS> ddp_minimal_wikitext.py
    #
    # We do NOT manually spawn processes. Instead, we read the environment variables
    # set by torchrun to initialize the distributed process group.

    dist.init_process_group(backend="nccl")  # For GPU-based training, typically NCCL
    local_rank = int(os.environ["LOCAL_RANK"])  # which GPU on this node
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    device = torch.device("cuda", local_rank)

    if global_rank == 0:
        print("==========================================")
        print(f" World size: {world_size}")
        print(f" Using GPU: local_rank={local_rank}")
        print("==========================================")

    ###############################################################################
    # 3) LOAD & PREPARE THE DATASET (only on rank 0, then broadcast if desired)
    ###############################################################################
    # For large datasets, you might want to only load data on rank 0 then distribute
    # to other ranks. For simplicity, each rank will download + load the dataset here.
    # This can be less efficient but simpler for demonstration.

    if global_rank == 0:
        print(f"Loading dataset {DATASET_NAME}/{DATASET_CONFIG}...")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    
    # Use a GPT-2 tokenizer for subword tokenization out-of-the-box.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # GPT2 tokenizer doesn't define a pad token, so re-use eos_token for padding:
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Concatenate all tokens into a single list (continuous stream).
    all_ids = []
    for row in tokenized_dataset:
        all_ids.extend(row["input_ids"])

    # Create contiguous blocks (chunks) of size BLOCK_SIZE from the token stream.
    def chunkify(lst, chunk_size):
        for i in range(0, len(lst) - chunk_size + 1, chunk_size):
            yield lst[i:i + chunk_size]

    chunks = list(chunkify(all_ids, BLOCK_SIZE))

    # Simple custom dataset for language modeling
    class WikiTextDataset(Dataset):
        def __init__(self, token_chunks):
            super().__init__()
            self.token_chunks = token_chunks

        def __len__(self):
            return len(self.token_chunks)

        def __getitem__(self, idx):
            input_ids = self.token_chunks[idx]
            return torch.tensor(input_ids, dtype=torch.long)

    train_dataset = WikiTextDataset(chunks)

    # Use DistributedSampler to split data among processes
    train_sampler = DistributedSampler(train_dataset)

    # If you use multiple nodes, pin_memory can be set to True if local_rank uses GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True,
    )

    from model import SimpleTransformerLM  # Import your model here
    
    ###############################################################################
    # 5) INITIALIZE MODEL & OPTIMIZER (DDP)
    ###############################################################################
    vocab_size = tokenizer.vocab_size
    model = SimpleTransformerLM(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        dropout=DROPOUT,
        block_size=BLOCK_SIZE
    ).to(device)

    # Wrap model with DistributedDataParallel
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    ###############################################################################
    # 6) TRAINING LOOP
    ###############################################################################
    model.train()
    for epoch in range(EPOCHS):
        # Set the epoch for the DistributedSampler so it shuffles differently each epoch
        train_sampler.set_epoch(epoch)

        # Only rank 0 will display a progress bar
        if global_rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        else:
            progress_bar = train_loader

        total_loss = 0.0
        for step, batch in enumerate(progress_bar):
            batch = batch.to(device, non_blocking=True)
            _, loss = model(batch, targets=batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if global_rank == 0 and (step + 1) % PRINT_EVERY == 0:
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Compute average loss across all processes
        avg_epoch_loss = total_loss / len(train_loader)

        # Optionally, gather loss from all ranks and average. 
        # For simplicity, we just print the rank 0 average.
        if global_rank == 0:
            print(f"[Rank 0] End of epoch {epoch+1}, average loss: {avg_epoch_loss:.4f}")

    ###############################################################################
    # 7) SAMPLE GENERATION (OPTIONAL) - only rank 0
    ###############################################################################
    if global_rank == 0:
        def generate(model, start_tokens, max_new_tokens=50):
            model.eval()
            generated = torch.tensor(start_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if generated.size(1) > BLOCK_SIZE:
                        generated = generated[:, -BLOCK_SIZE:]
                    logits, _ = model(generated)
                    # Take the last token's logits
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            return generated[0].tolist()

        prompt = "In the beginning"
        encoded_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated_tokens = generate(model, start_tokens=encoded_prompt[0], max_new_tokens=20)
        print("Generated text:", tokenizer.decode(generated_tokens))

    # Finalize DDP
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
