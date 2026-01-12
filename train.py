#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data import build_transforms, create_datasets, create_dataloaders
from model import SimpleCNN
from utils import ensure_dir, plot_class_counts, plot_confusion


def main():
    """
    Main training and evaluation pipeline.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to dataset folder containing train/val/test")
    parser.add_argument("--img-size", type=int, default=128, help="Image size after resizing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--outdir", default="OUT", help="Directory to store results")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dir(fig_dir)

    # Create datasets and loaders
    transform = build_transforms(args.img_size)
    train_ds, val_ds, test_ds = create_datasets(args.data_dir, transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, args.batch_size
    )

    # Plot class distribution
    plot_class_counts(train_ds, os.path.join(fig_dir, "01_train_class_counts.png"), "Training set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN(n_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # -------- Training loop --------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs} - loss {total_loss/len(train_loader):.4f}")

    # -------- Evaluation --------
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
    report = classification_report(y_true, y_pred, target_names=train_ds.classes)

    plot_confusion(cm, train_ds.classes, os.path.join(fig_dir, "02_confusion_matrix.png"))

    # Save metrics and model
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "classes": train_ds.classes,
            "confusion_matrix": cm.tolist()
        }, f, indent=2)

    torch.save(model.state_dict(), os.path.join(args.outdir, "cnn_model.pt"))

    print("All outputs saved in:", args.outdir)


if __name__ == "__main__":
    main()
