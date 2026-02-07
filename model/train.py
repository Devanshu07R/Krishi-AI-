# model/train.py
import os, json, torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# Paths & Hyperparams
DATA_DIR   = "data/train"
MODEL_PATH = "model/disease_model.pt"
CLASS_PATH = "model/classes.json"
BATCH      = 32
EPOCHS     = 6
LR         = 1e-3
IMG_SIZE   = 224
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Transforms
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Dataset & Split
full_ds = datasets.ImageFolder(DATA_DIR, transform=tfm)
train_len = int(0.8 * len(full_ds))
val_len   = len(full_ds) - train_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len])

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=BATCH)

# Model: MobileNetV2
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.last_channel, len(full_ds.classes))
model.to(DEVICE)

crit = nn.CrossEntropyLoss()
opt  = optim.Adam(model.parameters(), lr=LR)

for ep in range(EPOCHS):
    model.train()
    run_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        run_loss += loss.item()

    # Validation
    model.eval(); correct = 0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
    acc = correct / len(val_ds)
    print(f"Epoch {ep+1}/{EPOCHS}  Loss: {run_loss/len(train_dl):.4f}  Val Acc: {acc:.2%}")

# Save model and classes
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
with open(CLASS_PATH, "w") as f:
    json.dump(full_ds.classes, f)

print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Classes saved to {CLASS_PATH}")
