import os
import math
import csv
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
BLOCK_SIZE = 128
EMBED_DIM = 128
NUM_HEADS = 2
NUM_LAYERS = 2
FFN_DIM = 512
DROPOUT = 0.1
LR = 3e-4

EPOCHS = 50            # Train for 50 epochs
PRINT_EVERY = 100      # Steps between printing progress info
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"
CSV_STATS_FILE = "training_stats.csv"

# ---------------------------------------------------------------------------
# CFL-RELATED CONFIGS:
# ---------------------------------------------------------------------------
T_REFINE = 2   # Number of CFL refinement iterations (T). 2 or 3 often suffices.
CTX_DIM = 64   # Dimension of the global context vector z
# ---------------------------------------------------------------------------

def main():
    ###############################################################################
    # 2) INITIALIZE DISTRIBUTED
    ###############################################################################
    # Launch this script via:
    #   torchrun --nproc_per_node=<NUM_GPUS> ddp_minimal_wikitext_cfl.py
    #
    # We let torchrun handle process creation; we simply read the environment
    # variables to set up the distributed process group.

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", local_rank)

    if global_rank == 0:
        print("==========================================")
        print(f" World size: {world_size}")
        print(f" Using GPU: local_rank={local_rank}")
        print(f" Using {T_REFINE} CFL refinements each forward pass.")
        print("==========================================")

    ###############################################################################
    # 3) LOAD & PREPARE DATASET
    ###############################################################################
    if global_rank == 0:
        print(f"Loading dataset {DATASET_NAME}/{DATASET_CONFIG}...")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Reuse EOS token as pad for demonstration
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Concatenate all tokens into a single list
    all_ids = []
    for row in tokenized_dataset:
        all_ids.extend(row["input_ids"])

    # Chunk into BLOCK_SIZE
    def chunkify(lst, chunk_size):
        for i in range(0, len(lst) - chunk_size + 1, chunk_size):
            yield lst[i:i + chunk_size]

    chunks = list(chunkify(all_ids, BLOCK_SIZE))

    class WikiTextDataset(Dataset):
        def __init__(self, token_chunks):
            super().__init__()
            self.token_chunks = token_chunks

        def __len__(self):
            return len(self.token_chunks)

        def __getitem__(self, idx):
            return torch.tensor(self.token_chunks[idx], dtype=torch.long)

    train_dataset = WikiTextDataset(chunks)

    # Use DistributedSampler to split data among processes
    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True
    )

    from cfl_model import SimpleTransformerLM

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
        block_size=BLOCK_SIZE,
        context_dim=CTX_DIM,
        T_refine=T_REFINE
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    ###############################################################################
    # 6) PREPARE CSV STATS FILE (RANK 0)
    ###############################################################################
    if global_rank == 0:
        # If file doesn't exist, write the header
        if not os.path.exists(CSV_STATS_FILE):
            with open(CSV_STATS_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "train_ppl"])

    ###############################################################################
    # 7) TRAINING LOOP
    ###############################################################################
    model.train()
    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

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

        # Synchronize and compute global average loss
        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        global_loss_sum = total_loss_tensor.item()
        # Each rank sees the same number of steps (len(train_loader))
        global_avg_loss = global_loss_sum / (len(train_loader) * world_size)
        train_ppl = math.exp(global_avg_loss)

        # Rank 0 logs to CSV
        if global_rank == 0:
            with open(CSV_STATS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, global_avg_loss, train_ppl])

            print(f"[Rank 0] End of epoch {epoch+1}: "
                  f"avg_loss={global_avg_loss:.4f} ppl={train_ppl:.2f}")

    ###############################################################################
    # 8) SAMPLE GENERATION (OPTIONAL) - rank 0
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
                    next_token_logits = logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=1)
            return generated[0].tolist()

        prompt = "In the beginning"
        encoded_prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        generated_tokens = generate(model, start_tokens=encoded_prompt[0], max_new_tokens=20)
        print("Generated text:", tokenizer.decode(generated_tokens))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
