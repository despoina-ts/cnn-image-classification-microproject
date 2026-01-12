import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def build_transforms(img_size: int):
    """
    Build the image preprocessing pipeline.
    All images are resized and converted to tensors.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])


def create_datasets(data_dir: str, transform):
    """
    Load train/val/test datasets from a folder structure:
        data_dir/
            train/
            val/
            test/
    """
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=transform)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=transform)
    return train_ds, val_ds, test_ds


def create_dataloaders(train_ds, val_ds, test_ds, batch_size: int):
    """
    Wrap datasets into PyTorch DataLoaders.
    """
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
