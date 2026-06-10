"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _load_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")

def _load_mask(mask: str | Path | Image.Image | np.ndarray | torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask_array = mask.detach().to("cpu")
        if mask_array.ndim > 2:
            mask_array = mask_array.squeeze()
        mask_array = mask_array.numpy()
    elif isinstance(mask, np.ndarray):
        mask_array = mask
    elif isinstance(mask, (str, Path)):
        mask_array = np.array(Image.open(mask))
    else:
        mask_array = np.array(mask)

    mask_array = mask_array.squeeze()
    mask_array = mask_array > 0

    if mask_array.shape != size:
        mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255)
        mask_array = np.array(mask_image.resize((size[1], size[0]), resample=Image.NEAREST)) > 0

    return mask_array


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32) * 255.0
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_prediction_referring(
    target_image: str | Path | Image.Image,
    expression: str,
    predicted_mask: str | Path | Image.Image | np.ndarray | torch.Tensor,
    output_path: str | Path | None = None,
    *,
    alpha: float = 0.45,
    visualize: bool = False,
) -> None:
    """Save or show the target image with the predicted mask for a Referring Expression."""
    target_pil = _load_image(target_image)
    target_np = np.array(target_pil)

    predicted_mask_np = _load_mask(predicted_mask, target_np.shape[:2])
    target_overlay = _overlay_mask(target_np, predicted_mask_np, color=(0.15, 0.8, 0.35), alpha=alpha)

    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.imshow(target_overlay)
    axis.set_title(f'"{expression}"')
    axis.axis("off")

    if visualize:
        plt.show()
        plt.close(fig)
        return

    if output_path is None:
        raise ValueError("output_path must be provided when visualize is False")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)