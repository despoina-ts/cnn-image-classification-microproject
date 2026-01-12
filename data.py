import os
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(img_size: int) -> transforms.Compose:
    """
    Build the image preprocessing pipeline.

    All images are resized to a fixed square resolution and converted to
    PyTorch tensors.

    Parameters
    ----------
    img_size : int
        Target size (height and width) used to resize every input image.

    Returns
    -------
    torchvision.transforms.Compose
        A composition of transformations applied to every sample.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def create_datasets(
    data_dir: str,
    transform: transforms.Compose
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    """
    Create ImageFolder datasets for train/val/test splits.

    Expected directory structure:

        data_dir/
          train/<class_name>/*
          val/<class_name>/*
          test/<class_name>/*

    Parameters
    ----------
    data_dir : str
        Path to the dataset root directory.
    transform : torchvision.transforms.Compose
        Transform pipeline applied to images in all splits.

    Returns
    -------
    tuple of torchvision.datasets.ImageFolder
        (train_dataset, val_dataset, test_dataset)
    """
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    return train_ds, val_ds, test_ds


def create_dataloaders(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    test_ds: datasets.ImageFolder,
    batch_size: int,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets into PyTorch DataLoaders.

    Parameters
    ----------
    train_ds : torchvision.datasets.ImageFolder
        Training dataset.
    val_ds : torchvision.datasets.ImageFolder
        Validation dataset.
    test_ds : torchvision.datasets.ImageFolder
        Test dataset.
    batch_size : int
        Number of samples per batch.
    num_workers : int, optional
        Number of subprocesses used for data loading. Use 0 for maximum
        compatibility (default: 0).

    Returns
    -------
    tuple of torch.utils.data.DataLoader
        (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader
