import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from datetime import datetime

# ✅ Set correct source directory (your Early Blight images)
SOURCE_DIR = "data/test/Potato___Early_blight"

# ✅ Destination for augmented images
DEST_DIR = "data/augmented/Potato___Early_blight"
os.makedirs(DEST_DIR, exist_ok=True)

# ✅ Define augmentation transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0))
])

# ✅ Augment each image
print("Augmenting:")
for filename in tqdm(os.listdir(SOURCE_DIR)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(SOURCE_DIR, filename)
        image = Image.open(img_path).convert("RGB")

        # Apply transform
        augmented_img = transform(image)

        # Save augmented image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        aug_filename = f"aug_{timestamp}_{filename}"
        save_path = os.path.join(DEST_DIR, aug_filename)
        augmented_img.save(save_path)
