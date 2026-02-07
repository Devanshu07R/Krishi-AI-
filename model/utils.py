# model/utils.py

import torch
import json
from torchvision import models
from torch import nn

MODEL_PATH = "model/disease_model.pt"
CLASSES_PATH = "model/classes.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = models.mobilenet_v2(weights=None)
    
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_class_names():
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    return class_names

# Optional: Expose DEVICE
DEVICE = DEVICE
