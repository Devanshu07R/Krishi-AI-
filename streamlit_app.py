import streamlit as st
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import os
import cv2
import datetime
import json
from remedy_generator import get_remedy

# -------------------- PATHS --------------------
MODEL_PATH = "model/disease_model.pt"
CLASS_PATH = "model/classes.json"
LANG_PATH = "languages.json"
UPLOAD_DIR = "uploads"
HISTORY_FILE = "upload_history.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- LOAD DATA --------------------
with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

with open(LANG_PATH, "r", encoding="utf-8") as f:
    LANG_DATA = json.load(f)

# -------------------- MODEL --------------------
model = models.mobilenet_v2(weights="IMAGENET1K_V1")
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------- TRANSFORM --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------- GRAD-CAM --------------------
def get_gradcam_heatmap(img_tensor, model, target_layer="features"):
    activations, gradients = {}, {}

    def forward_hook(module, input, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    layer = dict([*model.named_modules()])[target_layer]
    fwd = layer.register_forward_hook(forward_hook)
    bwd = layer.register_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    output[0, pred_class].backward()

    act = activations["value"].detach().squeeze().cpu().numpy()
    grad = gradients["value"].detach().squeeze().cpu().numpy()


    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam /= cam.max() if cam.max() != 0 else 1

    fwd.remove()
    bwd.remove()
    return cam

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return np.uint8(heatmap * alpha + img)

# -------------------- HISTORY --------------------
def save_upload_history(filename):
    with open(HISTORY_FILE, "a") as f:
        f.write(f"{filename}\t{datetime.datetime.now()}\n")

def load_upload_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return f.readlines()
    return []

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="🌿 Krishi AI", layout="centered")

selected_lang = st.selectbox(
    "🌐 Select Language / भाषा चुनें",
    list(LANG_DATA.keys())
)
T = LANG_DATA[selected_lang]

st.markdown(f"""
<h1 style='text-align:center; color:#2d4d1f;'>🌿 {T["title"]}</h1>
<p style='text-align:center; font-size:18px; color:#4a773c;'>
AI-powered plant disease detection with camera & remedies
</p>
""", unsafe_allow_html=True)

# -------------------- IMAGE INPUT --------------------
st.markdown("## 📸 Capture or Upload Leaf Image")

input_mode = st.radio(
    "Choose input method:",
    ("Upload Image", "Use Camera")
)

image = None
uploaded_file = None

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader(
        T["upload"],
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_image = st.camera_input("📷 Capture Leaf Image")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")
        uploaded_file = camera_image

# -------------------- PREDICTION --------------------
if image is not None:
    st.image(image, caption="🖼️ Input Image", use_column_width=True)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(os.path.join(UPLOAD_DIR, filename))
    save_upload_history(filename)

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs[0], dim=0)
        confidence = probs.max().item()
        class_id = probs.argmax().item()
        detected_disease = class_names[class_id]

    st.markdown("### 🧬 Result")
    st.success(f"**{T['detected']}:** `{detected_disease}`")
    st.info(f"**{T['confidence']}:** `{confidence:.2f}`")

    # Grad-CAM
    heatmap = get_gradcam_heatmap(img_tensor, model)
    cam_img = overlay_heatmap(np.array(image), heatmap)
    st.image(cam_img, caption="🌡️ Grad-CAM Disease Localization", use_column_width=True)

    # Remedy
    st.markdown(f"### 💊 {T['remedy']}")
    remedy = get_remedy(detected_disease)
    st.text_area("🌱", remedy, height=180)

# -------------------- HISTORY --------------------
st.markdown("---")
with st.expander("📂 Upload History"):
    history = load_upload_history()
    if history:
        for h in reversed(history):
            name, time = h.strip().split("\t")
            st.markdown(f"- `{name}` → {time}")
    else:
        st.info("No uploads yet")

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<center>
Built by <b>Devanshu Dasgupta</b> 🌱<br>
ML Engineer | Explainable AI | AgriTech
</center>
""", unsafe_allow_html=True)
