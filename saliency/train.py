import os
import cv2  # for resizing patch-level attention maps (pip install opencv-python)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# For saliency (Captum)
try:
    from captum.attr import Saliency
except ImportError:
    print("Captum not found. Please install via: pip install captum")
    exit(1)

import matplotlib.pyplot as plt
import numpy as np
import argparse


# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
def get_imagenet_loaders(
    train_dir="/data/jacob/ImageNet/train",
    val_dir="/data/jacob/ImageNet/val",
    batch_size=64,
    num_workers=4,
    distributed=False,
    world_size=1,
    rank=0,
):
    """
    Returns train and validation DataLoaders for ImageNet with standard transforms.
    Uses DistributedSampler if 'distributed=True'.
    """
    # Standard ImageNet normalization
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=transform_val)

    if distributed:
        # Each process has its own sampler, with a subset of the data
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # shuffle only if not distributed
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    # For validation, we typically don't shuffle. We can skip distributed sampler or use it for consistent batch sizing.
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


# -----------------------------------------------------------------------------
# 2. TransformerEncoderLayerWithAttn
# -----------------------------------------------------------------------------
class TransformerEncoderLayerWithAttn(nn.Module):
    """
    A drop-in replacement for nn.TransformerEncoderLayer that always returns
    (output, attn_weights) with 'need_weights=True'.
    """
    def __init__(self, d_model, nhead, dim_feedforward=768, dropout=0.1,
                 activation="relu", batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=batch_first
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
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


# -----------------------------------------------------------------------------
# 3. Simple Vision Transformer
# -----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    """
    Splits an image of size 224x224 (typical for ImageNet) into non-overlapping patches.
    E.g. patch_size=16 => 14x14=196 patches. Then projects them to embed_dim.
    """
    def __init__(self, in_chans=3, patch_size=16, embed_dim=192, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.img_size   = img_size
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        out = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        B, E, H, W = out.shape  # e.g. H=W=14 if patch_size=16
        out = out.flatten(2)    # (B, E, H*W)
        out = out.transpose(1, 2)  # (B, H*W, E)
        return out


class SimpleTransformerEncoder(nn.Module):
    """
    A minimal TransformerEncoder for ImageNet classification.
    """
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
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

    def forward(self, x):
        B, N, E = x.shape
        x = x + self.pos_embedding[:, :N, :]

        for layer in self.layers:
            x, _ = layer(x)

        x_mean = x.mean(dim=1)
        logits = self.classifier(x_mean)
        return logits

    def get_last_layer_attention(self, x):
        B, N, E = x.shape
        x = x + self.pos_embedding[:, :N, :]

        for layer in self.layers[:-1]:
            x, _ = layer(x)

        x, attn_weights = self.layers[-1](x)
        return attn_weights, x


# -----------------------------------------------------------------------------
# 4. CFL (Contextual Feedback Loops)
# -----------------------------------------------------------------------------
class FeedbackAdapter(nn.Module):
    """
    Simple gating-based feedback:
    h_{t+1} = sigma(W_g z_t) * h_t
    """
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.gate = nn.Linear(context_dim, embed_dim)

    def forward(self, h_t, z_t):
        gate_vals = torch.sigmoid(self.gate(z_t))  # (B, embed_dim)
        gate_vals = gate_vals.unsqueeze(1)         # (B,1,embed_dim)
        return gate_vals * h_t


class CFLTransformer(nn.Module):
    """
    Wrap the base Vision Transformer with:
      - a projector g() to map final outputs -> context
      - a feedback adapter for iterative refinements
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
        logits, h_t = self.forward_once(x_emb)
        for _ in range(T):
            z_t = self.projector(logits)
            h_t = self.feedback_adapter(h_t, z_t)
            logits, h_t = self.forward_once(h_t)
        return logits

    def forward(self, x, T=0):
        x_emb = self.patch_embed(x)  # [B, seq_len, embed_dim]
        if T <= 0:
            return self.transformer(x_emb)
        else:
            return self.forward_cfl(x_emb, T=T)

    def get_attention_maps(self, x, T=0):
        x_emb = self.patch_embed(x)
        if T <= 0:
            attn_weights, _ = self.transformer.get_last_layer_attention(x_emb)
            return attn_weights

        logits, h_t = self.forward_once(x_emb)
        for _ in range(T):
            z_t = self.projector(logits)
            h_t = self.feedback_adapter(h_t, z_t)
            logits, h_t = self.forward_once(h_t)
        attn_weights, _ = self.transformer.get_last_layer_attention(h_t)
        return attn_weights


# -----------------------------------------------------------------------------
# 5. Train & Evaluate
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, rank=0):
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images, T=0)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)

    avg_loss = total_loss / total_count
    if rank == 0:
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, loader, device, T=0, rank=0):
    model.eval()
    correct_top1 = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, T=T)
            preds = torch.argmax(logits, dim=1)
            correct_top1 += (preds == labels).sum().item()
            total += labels.size(0)

    top1 = 100.0 * correct_top1 / total
    if rank == 0:
        print(f"Val Accuracy @T={T}: {top1:.2f}%")
    return top1


# -----------------------------------------------------------------------------
# 6. Saliency
# -----------------------------------------------------------------------------
def compute_saliency(model, images, device, T=0):
    """
    We'll use the max logit as the target for saliency.
    The returned sal_map is shape (B,1,H,W).
    """
    def forward_fn(x):
        out = model(x, T=T)
        preds = out.argmax(dim=1)  # predicted class index
        pred_logits = out[range(len(preds)), preds]
        return pred_logits

    sal_method = Saliency(forward_fn)
    images.requires_grad = True
    attributions = sal_method.attribute(images, target=None)
    sal_map = attributions.abs().mean(dim=1, keepdim=True)  # shape (B,1,H,W)
    return sal_map.detach()


# -----------------------------------------------------------------------------
# 7. Main (DDP-Style)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="DDP local rank, set by launch utility.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (demo).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--world_size", type=int, default=8, help="Total number of processes.")
    args = parser.parse_args()

    # 1) Initialize Process Group
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 2) Set device for this rank
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    if rank == 0:
        print(f"DDP training on rank {rank}, world_size={world_size}, device={device}.")

    # 3) Create DataLoaders with DistributedSampler
    train_dir = "/data/jacob/ImageNet/train"
    val_dir   = "/data/jacob/ImageNet/val"

    train_loader, val_loader = get_imagenet_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        num_workers=8,
        distributed=True,
        world_size=world_size,
        rank=rank
    )

    # 4) Build Model and Wrap with DDP
    model = CFLTransformer(
        patch_size=16,
        embed_dim=192,
        num_heads=4,
        num_layers=4,
        num_classes=1000,
        img_size=224,
        context_dim=64,
        seq_len=14*14,
        dim_feedforward=768,
        dropout=0.1
    ).to(device)

    # Wrap the model for DDP
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 5) Training loop
    EPOCHS = args.epochs
    for epoch in range(EPOCHS):
        # IMPORTANT: If using DistributedSampler, must call .set_epoch() each epoch
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        _ = train_one_epoch(model, train_loader, optimizer, device, epoch, rank)
        _ = evaluate(model, val_loader, device, T=0, rank=rank)

    # Evaluate at T=0,1,2,3
    if rank == 0:
        print("Evaluating with T=0,1,2,3 across the entire val set:")
    for t_val in [0, 1, 2, 3]:
        _ = evaluate(model, val_loader, device, T=t_val, rank=rank)

    # ----------------------
    # 6) Overlays (Attention & Saliency) - only rank=0
    # ----------------------
    if rank == 0:
        # We'll pick a small batch from val_loader
        sample_iter = iter(val_loader)
        images_batch, labels_batch = next(sample_iter)
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        # Just take first 4 images for demonstration
        sample_images = images_batch[:4].clone()
        sample_labels = labels_batch[:4].clone()

        # We'll do T in [0,1,2,3]
        os.makedirs("results", exist_ok=True)
        ddp_model = model.module  # unwrap DDP for direct method calls

        for t_val in [0, 1, 2, 3]:
            print(f"[Rank 0] Generating overlays for T={t_val}...")

            # 1) Saliency overlay
            saliency_maps = compute_saliency(ddp_model, sample_images.clone(), device, T=t_val)
            saliency_maps = saliency_maps.cpu().numpy()  # (B,1,224,224)

            for i in range(len(sample_images)):
                img_np = sample_images[i].clone().cpu().numpy()
                # shape (3,224,224); normally we might un-normalize for display
                # For brevity, just clip to [0,1]:
                img_np = np.clip(img_np, 0, 1)
                img_np = np.transpose(img_np, (1,2,0))  # (224,224,3)

                sal_map = saliency_maps[i,0]  # (224,224)

                fig, ax = plt.subplots(figsize=(3,3))
                ax.imshow(img_np)
                ax.imshow(sal_map, cmap='bwr', alpha=0.3)
                ax.set_title(f"SAL idx={i}, T={t_val}")
                ax.axis('off')
                plt.savefig(f"results/imagenet_saliency_{i}_T{t_val}.png", dpi=600)
                plt.close(fig)

            # 2) Attention overlay
            attn_maps = ddp_model.get_attention_maps(sample_images, T=t_val)
            if attn_maps is not None:
                attn_maps_avg = attn_maps.mean(dim=1)  # (B, seq_len, seq_len)

                for i in range(len(sample_images)):
                    A = attn_maps_avg[i].detach().cpu().numpy()
                    # For patch_size=16 => seq_len=196 => shape ~ (14,14)
                    if A.ndim == 1 and A.shape[0] == 196:
                        A = A.reshape(14,14)

                    # Upsample 14x14 => 224x224
                    A_upsampled = cv2.resize(A, (224,224), interpolation=cv2.INTER_LINEAR)

                    # Same image from above
                    img_np = sample_images[i].clone().cpu().numpy()
                    img_np = np.clip(img_np, 0, 1)
                    img_np = np.transpose(img_np, (1,2,0))  # (224,224,3)

                    fig, ax = plt.subplots(figsize=(3,3))
                    ax.imshow(img_np)
                    ax.imshow(A_upsampled, cmap='bwr', alpha=0.2)
                    ax.set_title(f"ATTN idx={i}, T={t_val}")
                    ax.axis('off')
                    plt.savefig(f"results/imagenet_attention_{i}_T{t_val}.png", dpi=600)
                    plt.close(fig)

        print("[Rank 0] Done! Overlays saved in ./results")


if __name__ == "__main__":
    main()
