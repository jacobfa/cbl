import os
import math
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless servers
import matplotlib.pyplot as plt

# For saliency (Captum)
try:
    from captum.attr import Saliency
except ImportError:
    raise ImportError("Captum not installed. Please install via: pip install captum")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

############################
# Utility / Model Components
############################

def denormalize_for_display(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Undo ImageNet normalization for nicer viewing. shape: (B,3,H,W).
    """
    denorm = tensor.clone()
    for c in range(3):
        denorm[:, c] = denorm[:, c] * std[c] + mean[c]
    return torch.clamp(denorm, 0, 1)

class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=768, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # IMPORTANT: pass average_attn_weights=False to get shape (B,nHeads,L,S)
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # <--- crucial
        )
        # Residual
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, patch_size=16, embed_dim=192, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size   = img_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Suppose x is (B,3,224,224) and patch_size=16 => (B, embed_dim,14,14) => 196 patches.
        Flatten => (B, 196, embed_dim).
        """
        out = self.proj(x)  # => (B, E, H//patch, W//patch)
        B, E, Hp, Wp = out.shape
        out = out.flatten(2).transpose(1, 2)  # => (B, Hp*Wp, E)
        return out

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=192, num_heads=4, num_layers=4,
                 num_classes=1000, seq_len=196, dim_feedforward=768, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        # Positional embedding up to seq_len=196
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

    def forward(self, x):
        """
        x: (B, N, E)  => N=H/patch * W/patch
        """
        B, N, E = x.shape
        x = x + self.pos_embedding[:, :N, :]
        for layer in self.layers:
            x, _ = layer(x)
        x_mean = x.mean(dim=1)  # pool
        logits = self.classifier(x_mean)
        return logits

    def get_last_layer_attention(self, x):
        """
        Returns attn_weights from last layer + final hidden states.
        With average_attn_weights=False => shape is (B, nHeads, N, N).
        """
        B, N, E = x.shape
        x = x + self.pos_embedding[:, :N, :]

        for layer in self.layers[:-1]:
            x, _ = layer(x)

        x, attn_weights = self.layers[-1](x)
        return attn_weights, x

class FeedbackAdapter(nn.Module):
    """
    Gating module for feedback. Projects logits => context => gating of hidden states.
    """
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.gate = nn.Linear(context_dim, embed_dim)

    def forward(self, h_t, z_t):
        # Gate is [0..1], broadcast over patch dimension
        gate_vals = torch.sigmoid(self.gate(z_t))  # (B,E)
        gate_vals = gate_vals.unsqueeze(1)         # => (B,1,E)
        return gate_vals * h_t

class CFLTransformer(nn.Module):
    """
    Vision Transformer with gating-based feedback (Contextual Feedback Loop).
    """
    def __init__(self, patch_size=16, embed_dim=192, num_heads=4, num_layers=4,
                 num_classes=1000, img_size=224, context_dim=64,
                 seq_len=196, dim_feedforward=768, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(
            in_chans=3, patch_size=patch_size,
            embed_dim=embed_dim, img_size=img_size
        )
        self.transformer = SimpleTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=num_classes,
            seq_len=seq_len,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.projector = nn.Sequential(
            nn.Linear(num_classes, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        self.feedback_adapter = FeedbackAdapter(embed_dim, context_dim)

    def forward_once(self, x_emb):
        logits = self.transformer(x_emb)
        return logits, x_emb

    def forward_cfl(self, x_emb, T=1):
        """
        For T>0, repeatedly feed back predicted class => gating => new pass
        """
        logits, h_t = self.forward_once(x_emb)
        for _ in range(T):
            z_t = self.projector(logits)
            h_t = self.feedback_adapter(h_t, z_t)
            logits, h_t = self.forward_once(h_t)
        return logits

    def forward(self, x, T=0):
        x_emb = self.patch_embed(x)
        if T <= 0:
            return self.transformer(x_emb)
        else:
            return self.forward_cfl(x_emb, T=T)

    def get_attention_maps(self, x, T=0):
        """
        Return last-layer attn maps after T feedback steps, shape => (B, nHeads, N, N).
        """
        x_emb = self.patch_embed(x)
        if T <= 0:
            attn_weights, _ = self.transformer.get_last_layer_attention(x_emb)
            return attn_weights
        # else apply feedback T times
        logits, h_t = self.forward_once(x_emb)
        for _ in range(T):
            z_t = self.projector(logits)
            h_t = self.feedback_adapter(h_t, z_t)
            logits, h_t = self.forward_once(h_t)
        attn_weights, _ = self.transformer.get_last_layer_attention(h_t)
        return attn_weights

####################
# Testing Script
####################

def main():
    # Settings
    model_path = "best_model.pth"
    imagenet_root = "/data/jacob/ImageNet"
    num_random_samples = 5  # how many random images to visualize
    output_dir = "results_inference_T"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Create a validation dataset/loader with ImageNet
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_dataset = ImageNet(root=imagenet_root, split='val', transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 2) Build model and load weights
    model = CFLTransformer(
        patch_size=16,
        embed_dim=192,
        num_heads=4,
        num_layers=4,
        num_classes=1000,
        img_size=224,
        context_dim=64,
        seq_len=196,
        dim_feedforward=768,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3) Select random indices from val set
    total_val = len(val_dataset)
    random_indices = random.sample(range(total_val), num_random_samples)

    # 4) For each random image, compute saliency & attention for T=0,1,2,3
    with torch.no_grad():
        for idx in random_indices:
            image, label = val_dataset[idx]
            # Prepare a batch of size 1
            image_tensor = image.unsqueeze(0).to(device)

            # Denormalized version for plotting
            vis_img = denormalize_for_display(image_tensor.clone())  # (1,3,H,W)
            vis_img_np = vis_img[0].cpu().numpy().transpose(1,2,0)    # (H,W,3)

            for T_val in [0, 1, 2, 3]:
                # Forward for the given T
                logits = model(image_tensor, T=T_val)
                pred_cls = logits.argmax(dim=1).item()
                # -----------------
                # Attention
                # -----------------
                attn_maps = model.get_attention_maps(image_tensor, T=T_val)  # (B,nHeads,N,N)
                # We'll average over heads and queries to get a single 2D map
                attn_1d = attn_maps[0].mean(dim=(0,1))  # shape (N,)
                n_patches = attn_1d.shape[0]
                side = int(math.isqrt(n_patches))

                # Convert to patch grid & upsample if it's a perfect square
                if side * side == n_patches:
                    A_2d = attn_1d.detach().cpu().numpy().reshape(side, side)
                    H, W = vis_img_np.shape[:2]
                    A_full = cv2.resize(A_2d, (W, H), interpolation=cv2.INTER_LINEAR)

                    # Plot & save attention overlay
                    fig_a, ax_a = plt.subplots(figsize=(4,4))
                    ax_a.imshow(vis_img_np)
                    ax_a.imshow(A_full, cmap="jet", alpha=0.4,
                                extent=[0, W, H, 0])
                    ax_a.axis("off")
                    attn_out_path = os.path.join(
                        output_dir,
                        f"idx_{idx}_label_{label}_pred_{pred_cls}_T{T_val}_attn.png"
                    )
                    plt.savefig(attn_out_path, dpi=600, bbox_inches="tight", pad_inches=0)
                    plt.close(fig_a)
                else:
                    print(
                        f"Skipping attention overlay for idx={idx}, T={T_val} "
                        f"because {n_patches} is not a perfect square."
                    )

                print(f"Done idx={idx}, T={T_val}, true_label={label}, pred={pred_cls}")

    print(f"All done! Check the '{output_dir}' folder for T=0..3 overlays.")

if __name__ == "__main__":
    main()
