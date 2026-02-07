import os
import shutil

SRC_DIR = "data/augmented/Potato___Early_blight"
DEST_DIR = "data/train/Potato___Early_blight"

os.makedirs(DEST_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DEST_DIR, fname)
    shutil.copy(src_path, dst_path)

print(f"✅ Copied {len(os.listdir(SRC_DIR))} files to training folder.")
