import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# ---------------------------------------------------------------------------
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
