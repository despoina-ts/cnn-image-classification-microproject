#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data import build_transforms, create_datasets, create_dataloaders
from model import SimpleCNN
from utils import ensure_dir, plot_class_counts, plot_confusion


def main() -> None:
    """
    Run the full training and evaluation pipeline.

    This function:
    - parses command-line arguments
    - loads the dataset
    - builds the model
    - trains the network
    - evaluates it on the test set
    - saves metrics, plots, and the trained model
    """
    parser = argparse.ArgumentParser(description="Train a CNN for image classification")
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Path to the dataset root directory containing train/val/test splits",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Target size to which all images will be resized",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of samples processed in each batch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="OUT",
        help="Directory where all outputs will be saved",
    )

    args = parser.parse_args()

    # Create output directories
    ensure_dir(args.outdir)
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dir(fig_dir)

    # Load data
    transform = build_transforms(args.img_size)
    train_ds, val_ds, test_ds = create_datasets(args.data_dir, transform)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, args.batch_size
    )

    # Plot class distribution for the training set
    plot_class_counts(train_ds, os.path.join(fig_dir, "01_train_class_counts.png"), "Training set")

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(n_classes=len(train_ds.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------ Training loop ------------------
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {avg_loss:.4f}")

    # ------------------ Evaluation ------------------
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(y.numpy())
            y_pred.extend(preds)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=train_ds.classes)

    plot_confusion(cm, train_ds.classes, os.path.join(fig_dir, "02_confusion_matrix.png"))

    # Save results
    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "classes": train_ds.classes,
                "confusion_matrix": cm.tolist(),
            },
            f,
            indent=2,
        )

    torch.save(model.state_dict(), os.path.join(args.outdir, "cnn_model.pt"))

    print("All outputs saved in:", args.outdir)


if __name__ == "__main__":
    main()

