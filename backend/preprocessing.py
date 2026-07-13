# preprocessing.py
# Mirrors Cell 4 of the notebook: skull stripping + 8x8 patch graph construction.

import cv2
import numpy as np
import torch
from torch_geometric.data import Data

IMAGE_SIZE = (128, 128)
PATCH_GRID = 8  # 8x8 = 64 patches/nodes


def skull_strip(img_uint8):
    """Otsu threshold + morphological ops + largest connected component."""
    blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        brain_mask = (labels == largest).astype(np.uint8) * 255
    else:
        brain_mask = closed
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    brain_mask = cv2.dilate(brain_mask, kernel2, iterations=2)
    return cv2.bitwise_and(img_uint8, img_uint8, mask=brain_mask)


def extract_patch_features(patch):
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    mx = float(np.max(patch))
    mn = float(np.min(patch))
    med = float(np.median(patch))
    var = float(np.var(patch))
    grad = float(np.sum(np.abs(np.diff(patch.astype(np.float32), axis=0))) / max(patch.size, 1))
    mad = float(np.mean(np.abs(patch - mean)))
    return [mean, std, mx, mn, med, var, grad, mad]


def _build_edges():
    """8-connected adjacency between the 64 patches (same for every image)."""
    edges = []
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            idx = r * PATCH_GRID + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < PATCH_GRID and 0 <= nc < PATCH_GRID:
                        edges.append([idx, nr * PATCH_GRID + nc])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


EDGE_INDEX = _build_edges()


def image_to_graph(image_bytes: bytes):
    """
    Takes raw uploaded image bytes -> returns (PyG Data graph, skull-stripped uint8 image)
    """
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file (jpg/png).")

    img = cv2.resize(img, IMAGE_SIZE)
    stripped = skull_strip(img)
    normalized = stripped.astype(np.float32) / 255.0

    patch_size = IMAGE_SIZE[0] // PATCH_GRID
    node_features = []
    for r in range(PATCH_GRID):
        for c in range(PATCH_GRID):
            patch = normalized[r * patch_size:(r + 1) * patch_size,
                                c * patch_size:(c + 1) * patch_size]
            node_features.append(extract_patch_features(patch))

    x = torch.tensor(node_features, dtype=torch.float32)
    data = Data(x=x, edge_index=EDGE_INDEX)
    data.batch = torch.zeros(x.shape[0], dtype=torch.long)  # single graph in this "batch"
    return data, stripped
