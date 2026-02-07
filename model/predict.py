# model/predict.py
import torch, json
from torchvision import models
from torch import nn

def load_model(model_path, class_path):
    with open(class_path, "r") as f:
        class_names = json.load(f)
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, class_names
