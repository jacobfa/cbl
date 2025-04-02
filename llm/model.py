import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------------------------------------------------
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
        # Causal mask (triangular) to prevent attention to future tokens
        # Shape (1, 1, T, T)
        mask = torch.ones((T, T), device=x.device).tril()
        mask = mask.unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.ln_final(x)
        logits = self.head(x)  # (B, T, vocab_size)
        if targets is None:
            return logits, None
        else:
            # Shift targets by 1: model predicts the next token
            logits = logits[:, :-1, :].contiguous()
            targets = targets[:, 1:].contiguous()
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
