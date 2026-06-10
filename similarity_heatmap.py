"""Render the patch-vs-text similarity heatmap for one image and expression."""

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from models.dinotxt import build_dinotxt
from utils.data import build_transform, load_image


def main() -> None:
    parser = argparse.ArgumentParser("dino.txt similarity heatmap")
    parser.add_argument("--image", default="assets/cat_image.jpg")
    parser.add_argument("--expression", default="a cat")
    parser.add_argument("--output", default="similarity_heatmap.png")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    encoder = build_dinotxt(device=args.device)
    image, _ = load_image(args.image, build_transform(args.image_size), args.device)
    sim = torch.einsum(
        "c,chw->hw",
        encoder.encode_expression(args.expression),
        encoder.encode_patches(image),
    )

    pil = Image.open(args.image).convert("RGB")
    sim_up = F.interpolate(
        sim[None, None], size=(pil.height, pil.width),
        mode="bilinear", align_corners=False,
    )[0, 0].cpu()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(pil)
    axes[0].set_title("input")
    axes[1].imshow(pil)
    axes[1].imshow(sim_up, alpha=0.6, cmap="jet")
    axes[1].set_title(f'similarity to "{args.expression}"')
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
