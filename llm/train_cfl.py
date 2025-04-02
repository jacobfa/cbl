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

    ###############################################################################
    # 4) DEFINE THE CFL-ENABLED TRANSFORMER MODEL
    ###############################################################################
    #
    # The key changes vs. a standard Transformer LM:
    #   - We unroll multiple "refinement" steps per forward pass.
    #   - We define a "projector" g() that maps the final output's hidden state
    #     or logits into a global context vector z^(tau).
    #   - We define "feedback adapters" that fuse z^(tau) into each layer's hidden
    #     states for the next iteration.
    #
    # Below is a minimal gating-based implementation for demonstration.
    #
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

    # ---------------------------------------------------------------------------
    # Simple gating-based adapter (psi^{(l)}) for injecting global context z.
    # h^(l)_new = gating(l)(h^(l), z) = sigmoid(W_g^l z + b_g^l) * h^(l).
    # We also apply a small linear+nonlinearity to the fused representation
    # to keep it in the same dimension as h^(l).
    # ---------------------------------------------------------------------------
    class FeedbackAdapter(nn.Module):
        def __init__(self, embed_dim, context_dim):
            """
            Gating-based feedback adapter:
               gating_factor = sigmoid(Wg * z + bg)  -> shape (B, embed_dim)
               out = gating_factor * h
            Then optionally pass through small linear to refine dimension.

            For simplicity, we'll do:
               gating_factor = MLP(z) -> shape (B, embed_dim)
               out = sigmoid(gating_factor) * h
               out = LN( out )        # optional
            """
            super().__init__()
            self.gate = nn.Linear(context_dim, embed_dim)
            self.ln = nn.LayerNorm(embed_dim)

        def forward(self, h, z):
            """
            h: (B, T, E)
            z: (B, context_dim)  -- a global context vector
            Return shape = (B, T, E)
            """
            # Expand z over time dimension so we can do elementwise gating
            B, T, E = h.size()
            # gating_factor shape = (B, E) after linear
            gating_factor = torch.sigmoid(self.gate(z))  # (B, E)
            gating_factor = gating_factor.unsqueeze(1)   # (B, 1, E) for broadcasting over T

            out = gating_factor * h
            out = self.ln(out)
            return out

    # ---------------------------------------------------------------------------
    # Projector g(y) -> z
    # We'll create a small module that, given the final hidden states or final
    # output, produces a single global context vector of dimension CTX_DIM.
    # For LM, we might take the last layer's hidden states of shape (B, T, E),
    # mean-pool them to shape (B, E), then linear -> (B, CTX_DIM).
    # ---------------------------------------------------------------------------
    class Projector(nn.Module):
        def __init__(self, embed_dim, context_dim):
            super().__init__()
            self.linear = nn.Linear(embed_dim, context_dim)

        def forward(self, hidden_states):
            """
            hidden_states: (B, T, E)
            We'll do simple mean-pooling over T, then a linear projection
            to get z in (B, context_dim).
            """
            # mean pool over T dimension
            # shape: (B, E)
            mean_h = hidden_states.mean(dim=1)
            z = self.linear(mean_h)
            return z

    class SimpleTransformerLM(nn.Module):
        """
        A minimal Transformer LM with Contextual Feedback Loops (CFL).

        We'll do T_REFINE unrolled refinements each forward.
        1) Standard forward to get initial hidden states, h^(l)_0, final h^(L)_0
        2) For tau in [0..T_REFINE-1]:
             - z^(tau) = projector( h^(L)_tau )
             - For each layer l, refine h^(l)_tau -> h^(l)_{tau+1} by injecting z^(tau)
             - Re-run forward from layer l to L in a chain
        3) Final output is from h^(L)_{T_REFINE}, produce logits, optionally compute loss.
        """
        def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ffn_dim,
                     dropout=0.1, block_size=128, context_dim=64, T_refine=2):
            super().__init__()
            self.vocab_size = vocab_size
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.ffn_dim = ffn_dim
            self.dropout = dropout
            self.block_size = block_size
            self.context_dim = context_dim
            self.T_refine = T_refine

            # Embeddings
            self.token_embed = nn.Embedding(vocab_size, embed_dim)
            self.pos_embed = nn.Embedding(block_size, embed_dim)

            # Transformer blocks
            self.layers = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ])

            # Feedback adapters (one per layer)
            self.feedback_adapters = nn.ModuleList([
                FeedbackAdapter(embed_dim, context_dim)
                for _ in range(num_layers)
            ])

            # Final layer norm & head
            self.ln_final = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, vocab_size, bias=False)

            # Projector g(\cdot)
            self.projector = Projector(embed_dim, context_dim)

        def forward(self, idx, targets=None):
            """
            idx: (B, T) token indices
            targets: (B, T), optional for LM training
            Returns: (logits, loss) or (logits, None)
            """
            B, T = idx.size()
            if T > self.block_size:
                raise ValueError("Sequence length exceeds block_size")

            # Produce initial hidden states by standard forward pass
            positions = torch.arange(0, T, device=idx.device).unsqueeze(0)
            tok_emb = self.token_embed(idx)      # (B, T, E)
            pos_emb = self.pos_embed(positions)  # (1, T, E)
            x = tok_emb + pos_emb

            # Causal mask
            mask = torch.ones((T, T), device=x.device).tril()
            mask = mask.unsqueeze(0).unsqueeze(0)  # shape (1, 1, T, T)

            # h^(l)_0 for l=1..L: do a normal forward pass
            hidden_states = []
            cur = x
            for layer in self.layers:
                cur = layer(cur, mask=mask)
                hidden_states.append(cur)  # store h^(l)

            # We have h^(1)_0 ... h^(L)_0 in hidden_states[-1]
            # We'll do T_refine feedback refinements
            for tau in range(self.T_refine):
                # Project final layer's hidden states to global context z^(tau)
                hL_tau = hidden_states[-1]   # shape (B, T, E) from the last layer
                z_tau = self.projector(hL_tau)  # (B, context_dim)

                # For each layer l=1..L, fuse z^(tau) into h^(l)_tau
                # Then re-run forward from that layer onward to get updated states
                #   h^(l)_{tau+1} = psi^{(l)}( h^(l)_tau, z^(tau) )
                #   then pass to the block to get next hidden
                new_states = []
                # We'll do a chain update: each layer uses the updated hidden from the previous
                prev = None
                for l, layer in enumerate(self.layers):
                    # Gate old h^(l) with z
                    h_l_tau = hidden_states[l]
                    h_l_new = self.feedback_adapters[l](h_l_tau, z_tau)
                    # Re-run the block on the new hidden (plus mask)
                    if l == 0:
                        # first layer input
                        prev = layer(h_l_new, mask=mask)
                    else:
                        prev = layer(prev, mask=mask)
                    new_states.append(prev)

                hidden_states = new_states  # updated h^(l)_{tau+1} for all layers

            # After T_refine, produce final logits from h^(L)_{T_refine}
            final_h = hidden_states[-1]  # shape: (B, T, E)
            final_h = self.ln_final(final_h)  # LN
            logits = self.head(final_h)       # (B, T, vocab_size)

            if targets is None:
                return logits, None
            else:
                # Typical next-token LM: shift by 1
                # We'll ignore the last token's prediction for the loss
                logits_for_loss = logits[:, :-1, :].contiguous()
                targets_for_loss = targets[:, 1:].contiguous()
                loss = nn.CrossEntropyLoss()(
                    logits_for_loss.view(-1, logits_for_loss.size(-1)),
                    targets_for_loss.view(-1)
                )
                return logits, loss

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
