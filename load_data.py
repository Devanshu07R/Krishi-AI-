# load_data.py
"""Utility to load plant‑disease images into a PyTorch DataLoader."""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(data_root: str = "data",
                    batch_size: int = 32,
                    img_size: int = 224):
    """
    Returns DataLoader and class names for images stored in:
    data_root/train/<class_name>/*.jpg
    """
    train_dir = os.path.join(data_root, "train")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"❌ Folder '{train_dir}' not found. "
            "Make sure your images are in data/train/<class_name>/."
        )

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"✅ Found {len(train_ds)} images across {len(train_ds.classes)} classes.")
    print(f"📂 Classes: {train_ds.classes}")

    return train_loader, train_ds.classes


# -------------------------------------------------
# Quick test: run `python load_data.py`
# -------------------------------------------------
if __name__ == "__main__":
    try:
        loader, classes = get_dataloaders(data_root="data", batch_size=8)
        images, labels = next(iter(loader))
        print(f"🖼️ Loaded batch size: {images.size(0)}")
        print(f"📏 Image tensor shape: {images.shape}")  # (B, 3, 224, 224)
        print(f"🏷️ Label tensor: {labels}")
    except Exception as e:
        print(str(e))
