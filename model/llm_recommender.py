import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from model import utils
from visualization.gradcam_demo import generate_gradcam
import base64
import io

# Set page config and style
st.set_page_config(page_title="🌿 AI Plant Doctor", layout="wide")
st.markdown("""
    <style>
        body {
            background-image: url('https://static.vecteezy.com/system/resources/previews/007/449/070/non_2x/agriculture-plant-seedling-growing-step-concept-with-mountain-and-sunrise-background-free-photo.jpg');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }
        .block-container {
 background-color: rgba(255, 255, 255, 0.85); /* Slight white overlay for readability */
            padding: 2rem;
            border-radius: 15px;
        }
        h1 {
            text-align: center;
            color: #2e4e1c;
        }
        footer {
            background-color: #dcedc8;
            text-align: center;
            padding: 1rem;
            font-size: 0.9rem;
            color: #2e4e1c;
            border-top: 2px solid #c5e1a5;
        }
    </style>
""", unsafe_allow_html=True)
# Title and header
st.markdown("""
    <h1 style='text-align: center; color: #2d4d1f;'>🌿 AI Plant Doctor</h1>
""", unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size:18px; color: #4a773c;'>
        Identify plant diseases using AI and Grad-CAM visualization.
    </p>
""", unsafe_allow_html=True)




# File uploader
uploaded_image = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

# Load model and classes
model = utils.load_model()
class_names = utils.load_class_names()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Upload history list
if "upload_history" not in st.session_state:
    st.session_state.upload_history = []

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save upload to history
    st.session_state.upload_history.append(uploaded_image.name)

    input_tensor = transform(image).unsqueeze(0)
    input_tensor = input_tensor.to(utils.DEVICE)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = prediction.argmax().item()
        predicted_class = class_names[predicted_class_idx]
        confidence_score = prediction[predicted_class_idx].item()
        st.markdown(f"### 🧠 Prediction: {predicted_class}")
        st.markdown(f"*Confidence:* {confidence_score:.2%}")
        if confidence_score < 0.3:
            st.error("🚨 Very low confidence! The result is likely unreliable.")
        elif confidence_score < 0.5:
            st.warning("⚠ Low confidence. You may want to double-check this result.")

        # Grad-CAM
        gradcam_img = generate_gradcam(
            model=model,
            input_tensor=input_tensor,
            target_layer=model.features[-1],
            predicted_class=predicted_class_idx,
            original_image=image,
            class_names=class_names
        )
        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_column_width=True)

        # Downloadable Grad-CAM result
        buffered = io.BytesIO()
        gradcam_img.save(buffered, format="PNG")
        img_data = buffered.getvalue()
        b64 = base64.b64encode(img_data).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="gradcam_result.png">📥 Download Grad-CAM Image</a>'
        st.markdown(href, unsafe_allow_html=True)

# Upload history
if st.session_state.upload_history:
    st.markdown("---")
    st.markdown("### 📝 Upload History")
    for i, fname in enumerate(reversed(st.session_state.upload_history), 1):
        st.markdown(f"{i}. {fname}")

# Footer
st.markdown("""
    <pre><footer>
    Built by Devanshu Dasgupta 🌱❤<br>
    <span style="font-size: 0.85rem;">ML Engineer</span>
</footer></pre>

""", unsafe_allow_html=True)