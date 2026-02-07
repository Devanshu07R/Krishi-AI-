import os
import sys
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────
# Add project root to Python path
# ─────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

# ─────────────────────────────────────────────
# Import model loader
# ─────────────────────────────────────────────
from model.predict import load_model

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "model", "disease_model.pt")
CLASS_PATH = os.path.join(ROOT, "model", "classes.json")
TEST_DIR = os.path.join(ROOT, "data", "test")
SAVE_DIR = os.path.join(ROOT, "evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Load model and class names (tuple unpack)
# ─────────────────────────────────────────────
model, CLASS_NAMES = load_model(MODEL_PATH, CLASS_PATH)
model.to(device)
model.eval()

# ─────────────────────────────────────────────
# Dataset & DataLoader
# ─────────────────────────────────────────────
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ─────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────
y_true, y_pred, filenames = [], [], []

with torch.no_grad():
    idx = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        outputs = model(xb)
        preds = outputs.argmax(1)

        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

        batch_files = dataset.samples[idx: idx + len(yb)]
        filenames.extend([os.path.basename(f[0]) for f in batch_files])
        idx += len(yb)

# ─────────────────────────────────────────────
# Save predictions CSV
# ─────────────────────────────────────────────
df = pd.DataFrame({
    "filename": filenames,
    "true_label": [CLASS_NAMES[i] for i in y_true],
    "predicted_label": [CLASS_NAMES[i] for i in y_pred]
})

csv_path = os.path.join(SAVE_DIR, "results.csv")
df.to_csv(csv_path, index=False)
print("✅ Saved predictions CSV ->", csv_path)

# ─────────────────────────────────────────────
# Classification Report
# ─────────────────────────────────────────────
print("\n📊 Classification Report\n")
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
)
print(report)

# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

img_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
plt.tight_layout()
plt.savefig(img_path)
plt.show()

print("✅ Saved confusion matrix ->", img_path)
