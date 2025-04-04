import os
import math
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageNet

# For saliency (Captum)
try:
    from captum.attr import Saliency
except ImportError:
    raise ImportError("Captum not installed. Please install via: pip install captum")

import matplotlib
matplotlib.use("Agg")  # For headless servers
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denormalize_for_display(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Undo ImageNet normalization for nicer viewing. shape: (B,3,H,W).
    """
    denorm = tensor.clone()
    for c in range(3):
        denorm[:, c] = denorm[:, c] * std[c] + mean[c]
    return torch.clamp(denorm, 0, 1)

def get_imagenet_loaders(
    root="/data/jacob/ImageNet",
    batch_size=64,
    num_workers=4,
    distributed=False,
    world_size=1,
    rank=0
):
    """
    Train/val DataLoaders for ImageNet. Expects /train and /val subfolders.
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = ImageNet(root=root, split='train', transform=transform_train)
    val_dataset   = ImageNet(root=root, split='val',   transform=transform_val)

    if distributed:
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
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader


# ---------------------------
# Transformer Building Blocks
# ---------------------------
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
        B, C, H, W = x.shape
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
        # Positional embedding is declared up to seq_len=196, but actual input might differ
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


# ---------------------------
# Training + Evaluation
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, rank=0):
    model.train()
    total_loss, total_count, correct_top1 = 0.0, 0, 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", disable=(rank!=0))
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images, T=0)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * images.size(0)
        total_count += images.size(0)
        correct_top1 += (logits.argmax(dim=1) == labels).sum().item()

    avg_loss = total_loss / total_count
    train_acc = 100.0 * correct_top1 / total_count
    if rank == 0:
        print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
    return avg_loss, train_acc

def evaluate(model, loader, device, T=0, rank=0, epoch=None):
    model.eval()
    correct_top1, total = 0, 0
    loop = tqdm(loader, desc=f"Validation T={T}", disable=(rank!=0))
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            logits = model(images, T=T)
            correct_top1 += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    top1 = 100.0 * correct_top1 / total
    if rank == 0:
        if epoch is not None:
            print(f"[Epoch {epoch+1}] Val Accuracy (T={T}): {top1:.2f}%")
        else:
            print(f"Val Accuracy (T={T}): {top1:.2f}%")
    return top1


# ---------------------------
# Saliency
# ---------------------------
def compute_saliency(model, images, device, T=0):
    """
    Return pixel-level saliency wrt each image's predicted class.
    shape => (B,1,H,W).
    """
    model.eval()

    def forward_fn(x):
        out = model(x, T=T)           # (B, #classes)
        preds = out.argmax(dim=1)     # top-1 class idx
        return out[range(len(preds)), preds]

    images_for_grad = images.clone().requires_grad_(True)
    sal_method = Saliency(forward_fn)
    attributions = sal_method.attribute(images_for_grad, target=None)
    # Combine absolute grads over RGB => single-channel
    sal_map = attributions.abs().mean(dim=1, keepdim=True)  # => (B,1,H,W)

    # Optional: normalize each map to [0,1] across the entire image
    # min_ = sal_map.flatten(1).min(dim=1)[0].view(-1,1,1,1)
    # max_ = sal_map.flatten(1).max(dim=1)[0].view(-1,1,1,1)
    # sal_map = (sal_map - min_) / (max_ - min_ + 1e-8)

    return sal_map.detach()


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--imagenet_root", type=str, default="/data/jacob/ImageNet")
    args = parser.parse_args()

    # Detect if distributed
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    else:
        rank = 0
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    distributed = (world_size > 1)
    if distributed:
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank
        )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = get_imagenet_loaders(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        num_workers=8,
        distributed=distributed,
        world_size=world_size,
        rank=rank
    )

    # Model
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
    ).to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    # Train loop
    for epoch in range(args.epochs):
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        train_one_epoch(model, train_loader, optimizer, device, epoch, rank)
        val_acc = evaluate(model, val_loader, device, T=0, rank=rank, epoch=epoch)

        # Save best
        if rank == 0 and val_acc > best_val_acc:
            best_val_acc = val_acc
            core_model = model.module if distributed else model
            torch.save(core_model.state_dict(), "best_model.pth")
            print(f"New best model saved (Val={best_val_acc:.2f}%).")

    # Evaluate with T=0..3
    if rank == 0:
        print("Evaluating with T=0,1,2,3 on val set:")
    for t_val in [0, 1, 2, 3]:
        _ = evaluate(model, val_loader, device, T=t_val, rank=rank)

    # Example: Overlays
    if rank == 0:
        os.makedirs("results", exist_ok=True)
        model_core = model.module if distributed else model
        model_core.eval()

        sample_imgs, sample_labels = next(iter(val_loader))
        sample_imgs = sample_imgs[:4].to(device)
        vis_imgs = denormalize_for_display(sample_imgs.clone())  # for display

        for T_val in [0, 1, 2, 3]:
            # Saliency
            sal_maps = compute_saliency(model_core, sample_imgs, device, T=T_val)
            sal_maps = sal_maps.cpu().numpy()  # => shape (B,1,H,W)

            # Attention
            attn_maps = model_core.get_attention_maps(sample_imgs, T=T_val)
            # => shape (B,nHeads,N,N) or None

            for i in range(len(sample_imgs)):
                # -----------
                # Saliency
                # -----------
                s_map = sal_maps[i,0]  # shape (H,W)
                img_np = vis_imgs[i].cpu().numpy().transpose(1,2,0)  # (H,W,3)

                fig, ax = plt.subplots(figsize=(4,4))
                ax.imshow(img_np)
                # Show saliency once over image
                ax.imshow(s_map, cmap="jet", alpha=0.4,
                          extent=[0, img_np.shape[1], img_np.shape[0], 0])
                ax.axis("off")
                plt.savefig(f"results/sample_{i}_T{T_val}_sal.png",
                            dpi=600, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # -----------
                # Attention
                # -----------
                if attn_maps is not None:
                    # average over heads, then average over "query" dimension => 1D per patch
                    # or pick, e.g., queries for the class token if you had a class token
                    # but here we have patch-based only, so let's just do an average.
                    attn_bnhn = attn_maps[i]  # shape (nHeads,N,N)
                    # average over heads + queries => shape (N,)
                    attn_1d = attn_bnhn.mean(dim=(0,1))  # => (N,)

                    n_patches = attn_1d.shape[0]
                    side = int(math.isqrt(n_patches))
                    if side * side != n_patches:
                        print(f"Skipping image {i} since {n_patches} isn't a perfect square.")
                        continue
                    A_2d = attn_1d.cpu().numpy().reshape(side, side)

                    # Upsample to (H,W)
                    A_full = cv2.resize(A_2d, (img_np.shape[1], img_np.shape[0]),
                                        interpolation=cv2.INTER_LINEAR)

                    fig, ax = plt.subplots(figsize=(4,4))
                    ax.imshow(img_np)
                    ax.imshow(A_full, cmap="jet", alpha=0.4,
                              extent=[0, img_np.shape[1], img_np.shape[0], 0])
                    ax.axis("off")
                    plt.savefig(f"results/sample_{i}_T{T_val}_attn.png",
                                dpi=600, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)

        print("Done! Check the ./results folder for overlay images.")

if __name__ == "__main__":
    main()
