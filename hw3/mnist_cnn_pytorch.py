import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


class MNISTCNN(nn.Module):
    """
    A simple, strong baseline:
    Conv(1, 32, 3) -> ReLU -> Conv(32, 32, 3) -> ReLU -> MaxPool(2) -> Dropout(0.25)
    Conv(32, 64, 3) -> ReLU -> Conv(64, 64, 3) -> ReLU -> MaxPool(2) -> Dropout(0.25)
    Flatten -> FC(1600, 128) -> ReLU -> Dropout(0.5) -> FC(128,10)
    """
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28x28 -> 14x14
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14x14 -> 7x7
            nn.Dropout(0.25),
        )
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dataloaders(data_dir: Path, batch_size: int, val_split: int = 5000, seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train/val/test loaders for MNIST. Downloads if not found."""
    train_tfms = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_full = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=train_tfms)
    test_set   = datasets.MNIST(root=str(data_dir), train=False, download=True, transform=test_tfms)

    # Split train into train/val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [len(train_full) - val_split, val_split], generator=g)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_acc  += (logits.argmax(1) == y).float().sum().item()
        n += y.size(0)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * y.size(0)
        total_acc  += (logits.argmax(1) == y).float().sum().item()
        n += y.size(0)
    return total_loss / n, total_acc / n


@torch.no_grad()
def make_confusion_matrix(model, loader, device, num_classes=10):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(1).cpu()
        ys.append(y)
        ps.append(pred)
    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy()
    cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
    return cm


def plot_curves(history, out_prefix="curves"):
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc  = [h["train_acc"] for h in history]
    va_acc  = [h["val_acc"] for h in history]

    plt.figure()
    plt.plot(epochs, tr_loss, label="Train Loss")
    plt.plot(epochs, va_loss, label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_loss.png", dpi=160)

    plt.figure()
    plt.plot(epochs, tr_acc, label="Train Acc")
    plt.plot(epochs, va_acc, label="Val Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_acc.png", dpi=160)


def save_samples(model, loader, device, out_path="samples.png", nrow=5):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
    preds = logits.argmax(1).cpu()
    grid = utils.make_grid(x[: nrow * nrow].cpu(), nrow=nrow, normalize=True, scale_each=True)
    plt.figure()
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("Predictions: " + " ".join(map(str, preds[: nrow * nrow].tolist())))
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def save_architecture_summary(model, out_path="architecture.txt"):
    try:
        from torchsummary import summary as torchsummary
    except Exception:
        torchsummary = None
    with open(out_path, "w") as f:
        f.write(str(model) + "\n\n")
        if torchsummary is None:
            f.write("Install torchsummary to include layer-by-layer shapes: pip install torchsummary\n")

            
    if torchsummary is not None:
        try:
            # This prints to stdout by default; capture not needed if writing text above
            torchsummary(model, (1,28,28), verbose=0)
        except Exception:
            pass


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup data
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, args.batch_size, val_split=5000, seed=args.seed)

    # Setup model
    model = MNISTCNN().to(device)
    nparams = count_params(model)
    print(model)
    print(f"Trainable parameters: {nparams:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training setup
    best_val = float("inf")
    history = []
    log_path = Path("log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })
        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f}\n")

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "model.pt")

    model.load_state_dict(torch.load("model.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

    plot_curves(history, out_prefix="curves")
    save_samples(model, test_loader, device, out_path="samples.png")
    
    try:
        cm = make_confusion_matrix(model, test_loader, device)
        import numpy as np
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        fig.savefig("confusion_matrix.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        print("Confusion matrix skipped (scikit-learn not installed).")

    save_architecture_summary(model, out_path="architecture.txt")

    with open("metrics.json", "w") as f:
        json.dump({
            "best_val_loss": best_val,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "params": nparams
        }, f, indent=2)


if __name__ == "__main__":
    main()
