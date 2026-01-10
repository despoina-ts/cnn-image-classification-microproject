#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -------------------- utils --------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def plot_class_counts(dataset, out_png, title):
    counts = {}
    for _, label in dataset.samples:
        counts[label] = counts.get(label, 0) + 1
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Images")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_confusion(cm, class_names, out_png):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------- CNN --------------------

class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),  # assumes 128x128 input
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--outdir", default="OUT")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dir(fig_dir)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_dir, "val"),   transform=transform)
    test_ds  = datasets.ImageFolder(os.path.join(args.data_dir, "test"),  transform=transform)

    plot_class_counts(train_ds, os.path.join(fig_dir, "01_train_class_counts.png"), "Train set")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(n_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ---- training ----
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} - loss {total_loss/len(train_loader):.4f}")

    # ---- evaluation ----
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=train_ds.classes)

    plot_confusion(cm, train_ds.classes, os.path.join(fig_dir, "02_confusion_matrix.png"))

    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(rep)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    torch.save(model.state_dict(), os.path.join(args.outdir, "cnn_model.pt"))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "classes": train_ds.classes,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    print("Saved outputs in:", args.outdir)


if __name__ == "__main__":
    main()
