# inference.py
# Loads the trained checkpoint once at startup, and exposes run_inference()
# which takes raw uploaded image bytes and returns a result dict.

import base64
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import QuantumGraphSAGE
from preprocessing import image_to_graph, PATCH_GRID

CLASS_NAMES = ["NonDemented", "MildDemented", "ModerateDemented", "VeryMildDemented"]
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "quantum_gat_sage_final.pth")

# PennyLane's default.qubit simulator cannot run on GPU, so we keep everything
# on CPU to avoid device-mismatch errors (same fix applied in the notebook).
device = torch.device("cpu")

print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

_config = checkpoint.get("config", {})
model = QuantumGraphSAGE(
    in_channels=_config.get("node_features", 8),
    hidden_channels=_config.get("hidden", 128),
    out_channels=4,
    n_qubits=_config.get("n_qubits", 8),
    n_layers=_config.get("n_layers", 4),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Use the class order that was actually saved in the checkpoint, if present.
CLASS_NAMES = checkpoint.get("class_names", CLASS_NAMES)

print("Model loaded successfully.")
print(f"  Reported test accuracy in checkpoint: {checkpoint.get('test_accuracy', 'n/a')}")


def _image_to_base64_png(img_array: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", img_array)
    if not ok:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def _build_heatmap_overlay(node_embeddings: torch.Tensor, base_img_uint8: np.ndarray) -> np.ndarray:
    """
    Builds a simple patch-level 'attention' style heatmap: the L2 norm of each
    patch's node embedding is used as an intensity score (higher = the network
    found more to respond to in that patch), reshaped to the 8x8 grid, resized
    up, colorized, and blended over the skull-stripped image.
    """
    emb = node_embeddings.detach().cpu().numpy()
    intensity = np.linalg.norm(emb, axis=1).reshape(PATCH_GRID, PATCH_GRID)

    # normalize to 0-255
    intensity = intensity - intensity.min()
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
    intensity_uint8 = (intensity * 255).astype(np.uint8)

    h, w = base_img_uint8.shape[:2]
    heat_resized = cv2.resize(intensity_uint8, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap(heat_resized, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(base_img_uint8, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.55, heat_color, 0.45, 0)
    return overlay


def run_inference(image_bytes: bytes) -> dict:
    graph, stripped_img = image_to_graph(image_bytes)
    graph = graph.to(device)

    with torch.no_grad():
        out = model(graph)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()
        node_embeddings = model.get_node_embeddings(graph)

    pred_idx = int(probs.argmax())
    heatmap_img = _build_heatmap_overlay(node_embeddings, stripped_img)

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "skull_stripped_preview": _image_to_base64_png(stripped_img),
        "attention_heatmap": _image_to_base64_png(heatmap_img),
        "model_info": {
            "reported_test_accuracy": checkpoint.get("test_accuracy"),
            "reported_test_auc": checkpoint.get("test_auc"),
            "reported_test_f1": checkpoint.get("test_f1"),
        },
    }
