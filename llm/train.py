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
EPOCHS = 50            # Train for 50 epochs as requested
PRINT_EVERY = 100      # Steps between printing progress info
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-v1"  # or "wikitext-2-v1" for a smaller test
CSV_STATS_FILE = "training_stats.csv"

def main():
    ###############################################################################
    # 2) INITIALIZE DISTRIBUTED
    ###############################################################################
    # This script is intended to be launched via:
    #   torchrun --nproc_per_node=<NUM_GPUS> ddp_minimal_wikitext.py

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", local_rank)

    # Only rank 0 prints some info
    if global_rank == 0:
        print("==========================================")
        print(f" World size: {world_size}")
        print(f" Using GPU: local_rank={local_rank}")
        print("==========================================")

    ###############################################################################
    # 3) LOAD & PREPARE DATASET
    ###############################################################################
    # Each process downloads and processes the dataset for simplicity.
    # For large-scale setups, you might only load it once on rank 0 and scatter,
    # but that adds complexity.

    if global_rank == 0:
        print(f"Loading dataset {DATASET_NAME}/{DATASET_CONFIG}...")

    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Re-use EOS token as pad

    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    # Concatenate token IDs into one long stream
    all_ids = []
    for row in tokenized_dataset:
        all_ids.extend(row["input_ids"])

    # Chunk into BLOCK_SIZE
    def chunkify(lst, chunk_size):
        for i in range(0, len(lst) - chunk_size + 1, chunk_size):
            yield lst[i:i + chunk_size]

    chunks = list(chunkify(all_ids, BLOCK_SIZE))

    # Custom dataset
    class WikiTextDataset(Dataset):
        def __init__(self, token_chunks):
            super().__init__()
            self.token_chunks = token_chunks

        def __len__(self):
            return len(self.token_chunks)

        def __getitem__(self, idx):
            return torch.tensor(self.token_chunks[idx], dtype=torch.long)

    train_dataset = WikiTextDataset(chunks)

    # DistributedSampler ensures each rank sees a unique subset
    train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        pin_memory=True
    )

    ###############################################################################
    # 4) DEFINE MODEL
    ###############################################################################
    class ScaledDotProductAttention(nn.Module):
        def forward(self, Q, K, V, mask=None):
            d_k = Q.size(-1)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)
            return output, attn_weights

    class MultiHeadAttention(nn.Module):
        def __init__(self, embed_dim, num_heads, dropout=0.1):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            self.Q_proj = nn.Linear(embed_dim, embed_dim)
            self.K_proj = nn.Linear(embed_dim, embed_dim)
            self.V_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.attention = ScaledDotProductAttention()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            B, T, E = x.size()
            Q = self.Q_proj(x)
            K = self.K_proj(x)
            V = self.V_proj(x)

            # Reshape to (B, num_heads, T, head_dim)
            Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            out, _ = self.attention(Q, K, V, mask=mask)
            out = out.transpose(1, 2).contiguous().view(B, T, E)
            out = self.out_proj(out)
            out = self.dropout(out)
            return out

    class TransformerBlock(nn.Module):
        def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
            super().__init__()
            self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
            self.ln1 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout)
            )
            self.ln2 = nn.LayerNorm(embed_dim)

        def forward(self, x, mask=None):
            attn_out = self.attn(x, mask=mask)
            x = self.ln1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.ln2(x + ffn_out)
            return x

    class SimpleTransformerLM(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim, dropout=0.1, block_size=128):
            super().__init__()
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(block_size, embed_dim)
            self.layers = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
            ])
            self.ln_final = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, vocab_size, bias=False)
            self.block_size = block_size

        def forward(self, idx, targets=None):
            B, T = idx.size()
            positions = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
            tok_emb = self.token_embed(idx)
            pos_emb = self.pos_embed(positions)
            x = tok_emb + pos_emb

            # Causal mask
            mask = torch.ones((T, T), device=x.device).tril()
            mask = mask.unsqueeze(0).unsqueeze(0)

            for layer in self.layers:
                x = layer(x, mask=mask)

            x = self.ln_final(x)
            logits = self.head(x)  # (B, T, vocab_size)

            if targets is None:
                return logits, None
            else:
                # next-token prediction
                logits = logits[:, :-1, :].contiguous()
                targets = targets[:, 1:].contiguous()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                             targets.view(-1))
                return logits, loss

    ###############################################################################
    # 5) INIT MODEL & OPTIMIZER (DDP)
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

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

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
        # set_epoch for DistributedSampler to shuffle differently each epoch
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

        # Synchronize and compute global average loss across all ranks
        total_loss_tensor = torch.tensor(total_loss, dtype=torch.float, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        # total_loss_tensor now is the SUM of losses from all ranks

        # Each rank runs the same number of steps, so total steps = len(train_loader)
        # total_loss_tensor / (len(train_loader)*world_size) => global average
        global_loss_sum = total_loss_tensor.item()
        global_avg_loss = global_loss_sum / (len(train_loader) * world_size)
        train_ppl = math.exp(global_avg_loss)

        # Rank 0 logs to CSV
        if global_rank == 0:
            with open(CSV_STATS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, global_avg_loss, train_ppl])

            print(f"[Rank 0] End of epoch {epoch+1}: avg_loss={global_avg_loss:.4f} ppl={train_ppl:.2f}")

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
                    # Take the last token's logits
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
