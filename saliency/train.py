import os
import cv2  # for resizing patch-level attention maps
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
    print("Captum not found. Please install via: pip install captum")
    exit(1)

import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm


# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
def get_imagenet_loaders(
    root="/data/jacob/ImageNet",
    batch_size=64,
    num_workers=4,
    distributed=False,
    world_size=1,
    rank=0
):
    """
    Returns train and validation DataLoaders for ImageNet with standard transforms,
    using torchvision.datasets.ImageNet. Expects a directory structure:
      /data/jacob/ImageNet/train/<class>/*.JPEG
      /data/jacob/ImageNet/val/<class>/*.JPEG
    """

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

    train_dataset = ImageNet(
        root=root,
        split='train',
        transform=transform_train
    )
    val_dataset = ImageNet(
        root=root,
        split='val',
        transform=transform_val
    )

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


# -----------------------------------------------------------------------------
# 2. TransformerEncoderLayerWithAttn
# -----------------------------------------------------------------------------
class TransformerEncoderLayerWithAttn(nn.Module):
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
        out = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        B, E, H, W = out.shape
        out = out.flatten(2)      # (B, E, H*W)
        out = out.transpose(1, 2) # (B, H*W, E)
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
    def __init__(self, embed_dim, context_dim):
        super().__init__()
        self.gate = nn.Linear(context_dim, embed_dim)

    def forward(self, h_t, z_t):
        gate_vals = torch.sigmoid(self.gate(z_t))  # (B, embed_dim)
        gate_vals = gate_vals.unsqueeze(1)         # (B,1,embed_dim)
        return gate_vals * h_t


class CFLTransformer(nn.Module):
    """
    Minimal Vision Transformer with a gating-based feedback loop.
    Hard-coded num_classes=1000 for standard ImageNet-1k.
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
        x_emb = self.patch_embed(x)
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
# 5. Train & Evaluate (with tqdm)
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, epoch, rank=0):
    model.train()
    total_loss = 0.0
    total_count = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", disable=(rank!=0))
    for images, labels in loop:
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

    loop = tqdm(loader, desc=f"Validation T={T}", disable=(rank!=0))
    with torch.no_grad():
        for images, labels in loop:
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
    def forward_fn(x):
        out = model(x, T=T)
        preds = out.argmax(dim=1)
        pred_logits = out[range(len(preds)), preds]
        return pred_logits

    sal_method = Saliency(forward_fn)
    images.requires_grad = True
    attributions = sal_method.attribute(images, target=None)
    sal_map = attributions.abs().mean(dim=1, keepdim=True)
    return sal_map.detach()


# -----------------------------------------------------------------------------
# 7. Main (DDP-Style) - Updated to use environment variables
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # If your ImageNet is in /data/jacob/ImageNet with train/ and val/ subfolders,
    # we set that as the root. This code expects 1000 classes in there.
    imagenet_root = "/data/jacob/ImageNet"

    # Read local/global ranks and world_size from environment variables
    # that are set by torchrun or torch.distributed.launch
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://",
                            world_size=world_size, rank=rank)

    # Assign the current process to its own GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Build DataLoaders in distributed mode
    train_loader, val_loader = get_imagenet_loaders(
        root=imagenet_root,
        batch_size=args.batch_size,
        num_workers=8,
        distributed=True,
        world_size=world_size,
        rank=rank
    )

    # Hard-code 1000 classes for standard ImageNet
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

    # Wrap model with DDP; this process uses device = local_rank
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Main training loop
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        _ = train_one_epoch(model, train_loader, optimizer, device, epoch, rank)
        _ = evaluate(model, val_loader, device, T=0, rank=rank)

    # Evaluate with T=0..3
    if rank == 0:
        print("Evaluating with T=0,1,2,3 on val set:")
    for t_val in [0, 1, 2, 3]:
        _ = evaluate(model, val_loader, device, T=t_val, rank=rank)

    # Example: Generate attention/saliency overlays on a small batch
    if rank == 0:
        sample_iter = iter(val_loader)
        images_batch, labels_batch = next(sample_iter)
        images_batch = images_batch.to(device)

        # We'll just take the first 4 images
        sample_images = images_batch[:4].clone()

        os.makedirs("results", exist_ok=True)
        # unwrap from DDP
        ddp_core = model.module  

        for t_val in [0, 1, 2, 3]:
            saliency_maps = compute_saliency(ddp_core, sample_images.clone(), device, T=t_val)
            saliency_maps = saliency_maps.cpu().numpy()

            for i in range(len(sample_images)):
                img_np = sample_images[i].detach().cpu().numpy()
                # If images were in [0,1], ensure they’re clipped.
                # (If normalizing in transforms, you may want to de-normalize for visualization.)
                img_np = np.clip(img_np, 0, 1)
                img_np = np.transpose(img_np, (1,2,0))

                sal_map = saliency_maps[i,0]

                fig, ax = plt.subplots(figsize=(3,3))
                ax.imshow(img_np)
                ax.imshow(sal_map, cmap='bwr', alpha=0.3)
                ax.axis('off')
                plt.savefig(f"results/imagenet_saliency_{i}_T{t_val}.png", dpi=600)
                plt.close(fig)

            attn_maps = ddp_core.get_attention_maps(sample_images, T=t_val)
            if attn_maps is not None:
                attn_maps_avg = attn_maps.mean(dim=1)

                for i in range(len(sample_images)):
                    A = attn_maps_avg[i].detach().cpu().numpy()
                    if A.ndim == 1 and A.shape[0] == 196:
                        A = A.reshape(14,14)

                    A_upsampled = cv2.resize(A, (224,224), interpolation=cv2.INTER_LINEAR)

                    img_np = sample_images[i].cpu().numpy()
                    img_np = np.clip(img_np, 0, 1)
                    img_np = np.transpose(img_np, (1,2,0))

                    fig, ax = plt.subplots(figsize=(3,3))
                    ax.imshow(img_np)
                    ax.imshow(A_upsampled, cmap='bwr', alpha=0.2)
                    ax.axis('off')
                    plt.savefig(f"results/imagenet_attention_{i}_T{t_val}.png", dpi=600)
                    plt.close(fig)

        print("Done! Overlays saved in ./results")


if __name__ == "__main__":
    main()
