#!/usr/bin/env python

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

# -------------------------------------------------
#               1. HYPERPARAMETERS
# -------------------------------------------------
SEED               = 42
BATCH_SIZE         = 128
LEARNING_RATE      = 1e-3
EPOCHS             = 100
USE_LAYERNORM      = False  # Switch between BatchNorm vs LayerNorm if you like
FEEDBACK_ITERATIONS= 2      # T for CFL iterative refinement
NUM_CLASSES        = 10

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# -------------------------------------------------
# 2. NETWORK DEFINITION (Standard + CFL Models)
# -------------------------------------------------
def norm_layer(num_features, use_layernorm):
    if use_layernorm:
        return nn.LayerNorm(num_features)
    else:
        return nn.BatchNorm2d(num_features)

class StandardCNN(nn.Module):
    """
    A simple 3-layer CNN for CIFAR-10:
      - Conv -> Norm -> ReLU -> Dropout -> Pool (x3)
      - Flatten -> Linear
    """
    def __init__(self, num_classes=10, use_layernorm=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.norm1 = norm_layer(32, use_layernorm)
        self.drop1 = nn.Dropout(p=0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.norm2 = norm_layer(64, use_layernorm)
        self.drop2 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.norm3 = norm_layer(128, use_layernorm)
        self.drop3 = nn.Dropout(p=0.25)

        self.pool  = nn.MaxPool2d(2, 2)
        self.fc    = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self._apply_norm(x, self.norm1)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self._apply_norm(x, self.norm2)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self._apply_norm(x, self.norm3)
        x = F.relu(x)
        x = self.drop3(x)
        x = self.pool(x)

        # Flatten + FC
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

    def _apply_norm(self, x, norm):
        """Handle BN vs LN shapes."""
        if isinstance(norm, nn.BatchNorm2d):
            return norm(x)
        else:
            # LayerNorm
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = norm(x)
            x = x.permute(0, 3, 1, 2)
            return x

class AttentionAdapter(nn.Module):
    """
    Attention-based adapter that fuses local CNN features h (B, C, H, W)
    with a global context vector z (B, d_z).
    We'll do a simple "query = z, key = h, value = h" style attention.
    """
    def __init__(self, in_channels, d_z):
        super().__init__()
        self.in_channels = in_channels
        self.d_z = d_z

        self.W_q = nn.Linear(d_z, in_channels)
        self.W_k = nn.Linear(in_channels, in_channels)
        self.W_v = nn.Linear(in_channels, in_channels)

        self.out_proj = nn.Linear(in_channels, in_channels)

    def forward(self, h, z):
        B, C, H, W = h.shape
        # Flatten to (B, H*W, C)
        h_reshaped = h.permute(0, 2, 3, 1).reshape(B, H*W, C)

        Q = self.W_q(z).unsqueeze(1)   # (B, 1, C)
        K = self.W_k(h_reshaped)       # (B, H*W, C)
        V = self.W_v(h_reshaped)       # (B, H*W, C)

        d_k = C**0.5
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / d_k
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, H*W)

        context = torch.bmm(attn_weights, V)  # (B, 1, C)
        context = self.out_proj(context).squeeze(1)  # (B, C)

        # Residual + broadcast
        context_4d = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        refined_h = h + context_4d
        return refined_h

class CFLAttentionCNN(nn.Module):
    """
    3-layer CNN augmented with top-down feedback using an attention-based adapter.
    We'll do T iterative refinements.
    """
    def __init__(self, num_classes=10, d_z=64, T=2):
        super().__init__()
        self.T = T

        # Convolutional trunk
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)

        self.fc_in_features = 128 * 4 * 4
        self.fc   = nn.Linear(self.fc_in_features, num_classes)

        # Projector g(): maps logits to context vector z
        self.projector = nn.Sequential(
            nn.Linear(num_classes, d_z),
            nn.ReLU(),
            nn.Linear(d_z, d_z),
        )

        # Attention adapters
        self.adapter1 = AttentionAdapter(32, d_z)
        self.adapter2 = AttentionAdapter(64, d_z)
        self.adapter3 = AttentionAdapter(128, d_z)

    def forward(self, x):
        # 1) Initial forward pass
        (h1, h2, h3), logits = self.forward_once(x)

        # 2) Iterative refinement
        for _ in range(self.T):
            z = self.projector(logits)

            h1 = self.adapter1(h1, z)
            h2 = self.adapter2(h2, z)
            h3 = self.adapter3(h3, z)

            # re-run partial forward
            h2 = F.relu(self.conv2(h1))
            h2 = self.pool(h2)

            h3 = F.relu(self.conv3(h2))
            h3 = self.pool(h3)

            flat = h3.view(h3.size(0), -1)
            logits = self.fc(flat)

        return logits

    def forward_once(self, x):
        h1 = F.relu(self.conv1(x))
        h1 = self.pool(h1)

        h2 = F.relu(self.conv2(h1))
        h2 = self.pool(h2)

        h3 = F.relu(self.conv3(h2))
        h3 = self.pool(h3)

        flat   = h3.view(h3.size(0), -1)
        logits = self.fc(flat)
        return (h1, h2, h3), logits

# -------------------------------------------------
#           DDP TRAIN & EVAL FUNCTIONS
# -------------------------------------------------
def train_one_epoch(model, optimizer, loader, epoch, local_rank):
    model.train()
    # Important: set the distributed sampler epoch for data shuffling
    loader.sampler.set_epoch(epoch)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for (data, target) in loader:
        data, target = data.cuda(local_rank), target.cuda(local_rank)

        optimizer.zero_grad()
        output = model(data)
        loss   = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Local (per-GPU) stats
        total_loss    += loss.item() * data.size(0)
        _, pred        = torch.max(output, dim=1)
        total_correct += (pred == target).sum().item()
        total_samples += data.size(0)

    # All-reduce for global stats across GPUs
    total_loss_tensor    = torch.tensor([total_loss], dtype=torch.float32, device=local_rank)
    total_correct_tensor = torch.tensor([total_correct], dtype=torch.float32, device=local_rank)
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device=local_rank)

    dist.all_reduce(total_loss_tensor,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    # Compute global average loss and accuracy
    global_loss = total_loss_tensor.item()    / total_samples_tensor.item()
    global_acc  = 100.0 * total_correct_tensor.item() / total_samples_tensor.item()

    return global_loss, global_acc


def evaluate(model, loader, local_rank):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for (data, target) in loader:
            data, target = data.cuda(local_rank), target.cuda(local_rank)
            output = model(data)
            loss   = F.cross_entropy(output, target)

            total_loss    += loss.item() * data.size(0)
            _, pred        = torch.max(output, dim=1)
            total_correct += (pred == target).sum().item()
            total_samples += data.size(0)

    # All-reduce for global stats
    total_loss_tensor    = torch.tensor([total_loss], dtype=torch.float32, device=local_rank)
    total_correct_tensor = torch.tensor([total_correct], dtype=torch.float32, device=local_rank)
    total_samples_tensor = torch.tensor([total_samples], dtype=torch.float32, device=local_rank)

    dist.all_reduce(total_loss_tensor,    op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

    global_loss = total_loss_tensor.item()    / total_samples_tensor.item()
    global_acc  = 100.0 * total_correct_tensor.item() / total_samples_tensor.item()
    return global_loss, global_acc


def main():
    # -------------------------------------------------
    #   1. Initialize DDP & define local_rank/gpu
    # -------------------------------------------------
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # (Optional) Only do console prints on rank=0
    def sprint(*args, **kwargs):
        if dist.get_rank() == 0:
            print(*args, **kwargs)

    # -------------------------------------------------
    #   2. Setup CIFAR-10 + Distributed Samplers
    # -------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(), 
        shuffle=True
    )
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        num_workers=2
    )

    # -------------------------------------------------
    #  3. Instantiate and DDP-wrap both models
    # -------------------------------------------------
    model_standard = StandardCNN(num_classes=NUM_CLASSES, use_layernorm=USE_LAYERNORM).cuda(local_rank)
    model_cfl      = CFLAttentionCNN(num_classes=NUM_CLASSES, d_z=64, T=FEEDBACK_ITERATIONS).cuda(local_rank)

    ddp_model_standard = DDP(model_standard, device_ids=[local_rank], output_device=local_rank)
    ddp_model_cfl      = DDP(model_cfl,      device_ids=[local_rank], output_device=local_rank,
                            find_unused_parameters=True)

    optimizer_standard = optim.Adam(ddp_model_standard.parameters(), lr=LEARNING_RATE)
    optimizer_cfl      = optim.Adam(ddp_model_cfl.parameters(),      lr=LEARNING_RATE)

    # -------------------------------------------------
    #       4. MAIN TRAINING LOOP (COMPARISON)
    # -------------------------------------------------
    train_losses_std, train_accs_std  = [], []
    test_losses_std,  test_accs_std   = [], []

    train_losses_cfl, train_accs_cfl = [], []
    test_losses_cfl,  test_accs_cfl  = [], []

    # Only rank=0 will handle logging to file
    if dist.get_rank() == 0:
        log_file = open("log.txt", "a")

    for epoch in range(EPOCHS):
        sprint(f"\n=== Epoch [{epoch+1}/{EPOCHS}] ===")

        # --- Train standard CNN (DDP) ---
        train_loss_s, train_acc_s = train_one_epoch(ddp_model_standard, optimizer_standard, train_loader, epoch, local_rank)
        test_loss_s, test_acc_s   = evaluate(ddp_model_standard, test_loader, local_rank)

        train_losses_std.append(train_loss_s)
        train_accs_std.append(train_acc_s)
        test_losses_std.append(test_loss_s)
        test_accs_std.append(test_acc_s)

        sprint(
            f"[StdCNN]  Epoch {epoch+1} | Train Loss: {train_loss_s:.4f}, Train Acc: {train_acc_s:.2f}% "
            f"| Test Loss: {test_loss_s:.4f}, Test Acc: {test_acc_s:.2f}%"
        )

        # --- Train CFL CNN (DDP) ---
        train_loss_c, train_acc_c = train_one_epoch(ddp_model_cfl, optimizer_cfl, train_loader, epoch, local_rank)
        test_loss_c, test_acc_c   = evaluate(ddp_model_cfl, test_loader, local_rank)

        train_losses_cfl.append(train_loss_c)
        train_accs_cfl.append(train_acc_c)
        test_losses_cfl.append(test_loss_c)
        test_accs_cfl.append(test_acc_c)

        sprint(
            f"[CFL-CNN] Epoch {epoch+1} | Train Loss: {train_loss_c:.4f}, Train Acc: {train_acc_c:.2f}% "
            f"| Test Loss: {test_loss_c:.4f}, Test Acc: {test_acc_c:.2f}%"
        )

        # --- LOG TO FILE (only rank=0) ---
        if dist.get_rank() == 0:
            log_file.write(f"Epoch {epoch+1}/{EPOCHS}\n")
            log_file.write(
                f"[StdCNN]  Train Loss: {train_loss_s:.4f}, Train Acc: {train_acc_s:.2f}%, "
                f"Test Loss: {test_loss_s:.4f}, Test Acc: {test_acc_s:.2f}%\n"
            )
            log_file.write(
                f"[CFL-CNN] Train Loss: {train_loss_c:.4f}, Train Acc: {train_acc_c:.2f}%, "
                f"Test Loss: {test_loss_c:.4f}, Test Acc: {test_acc_c:.2f}%\n\n"
            )
            log_file.flush()

    if dist.get_rank() == 0:
        log_file.close()

    # -------------------------------------------------
    #    5. (Optional) PLOTTING RESULTS (rank=0)
    # -------------------------------------------------
    if dist.get_rank() == 0:
        sprint("\nPlotting results...")

        matplotlib.rcParams.update({'font.size': 12})
        plt.style.use('bmh')
        epochs_range = np.arange(1, EPOCHS+1)

        # -- Training Loss --
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, train_losses_std, label='StdCNN - Train Loss', marker='o')
        plt.plot(epochs_range, train_losses_cfl,  label='CFL-CNN - Train Loss',  marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison (CIFAR-10)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cifar10_train_loss_comparison.png', dpi=200)
        plt.close()

        # -- Test Loss --
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, test_losses_std, label='StdCNN - Test Loss', marker='o')
        plt.plot(epochs_range, test_losses_cfl,  label='CFL-CNN - Test Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Test Loss Comparison (CIFAR-10)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cifar10_test_loss_comparison.png', dpi=200)
        plt.close()

        # -- Training Accuracy --
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, train_accs_std, label='StdCNN - Train Acc', marker='o')
        plt.plot(epochs_range, train_accs_cfl, label='CFL-CNN - Train Acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy Comparison (CIFAR-10)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cifar10_train_acc_comparison.png', dpi=200)
        plt.close()

        # -- Test Accuracy --
        plt.figure(figsize=(8, 6))
        plt.plot(epochs_range, test_accs_std, label='StdCNN - Test Acc', marker='o')
        plt.plot(epochs_range, test_accs_cfl,  label='CFL-CNN - Test Acc', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy Comparison (CIFAR-10)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('cifar10_test_acc_comparison.png', dpi=200)
        plt.close()

        sprint("\nTraining complete! Plots saved to:")
        sprint("  cifar10_train_loss_comparison.png")
        sprint("  cifar10_test_loss_comparison.png")
        sprint("  cifar10_train_acc_comparison.png")
        sprint("  cifar10_test_acc_comparison.png")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
