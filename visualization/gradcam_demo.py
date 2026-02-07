import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, input_tensor, target_layer, predicted_class, original_image, class_names):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    # Forward and backward pass
    model.eval()
    output = model(input_tensor)
    model.zero_grad()

    class_loss = output[0, predicted_class]
    class_loss.backward()

    # Clean up hooks
    handle_fw.remove()
    handle_bw.remove()

    # Process activations and gradients
    grads_val = gradients[0][0].detach().cpu().numpy()
    activations_val = activations[0][0].detach().cpu().numpy()

    weights = np.mean(grads_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations_val[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_image.size[0], original_image.size[1]))
    cam -= np.min(cam)
    cam /= np.max(cam)
    cam = np.uint8(255 * cam)

    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    image_np = np.array(original_image)
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = image_np[:, :, :3]

    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    final_image = Image.fromarray(superimposed_img)

    return final_image
