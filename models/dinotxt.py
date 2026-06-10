"""dino.txt encoder: Text Prototypes and Aligned Features for TFRIS.

Hub entry and API verified against the official dinov3 repo
(dinov3/hub/dinotxt.py, dinov3/eval/text/dinotxt_model.py):
`dinov3_vitl16_dinotxt_tet1280d20h24l(weights=..., backbone_weights=...,
bpe_path_or_url=...)` accepts local paths and returns (DINOTxt, tokenizer).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

_DINOTXT_WEIGHTS = "pretrain/dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth"
_BACKBONE_WEIGHTS = "pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
_BPE_VOCAB = "pretrain/bpe_simple_vocab_16e6.txt.gz"

# Patch-vs-text similarity maps share a dominant non-semantic component
# (maps for unrelated texts correlate at 0.8-0.9), so the raw similarity of a
# single expression is unusable for localization. Subtracting the mean
# embedding of a fixed bank of neutral texts removes that shared bias - the
# text-space analogue of INSID3's positional debiasing. The bank is a fixed
# implementation constant, not a tuned hyperparameter.
_NEUTRAL_TEXTS = (
    "a photo", "an image", "a thing", "an object", "the background",
    "a wall", "the sky", "a person", "an animal", "a building",
    "a plant", "food",
)


class DinoTxtEncoder(nn.Module):
    """Wraps the dino.txt model + tokenizer behind two methods."""

    def __init__(self, model: nn.Module, tokenizer, device: str):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._neutral_mean = torch.stack(
            [self._embed_text(t) for t in _NEUTRAL_TEXTS]
        ).mean(dim=0)

    @torch.no_grad()
    def _embed_text(self, text: str) -> torch.Tensor:
        """Raw patch-aligned text embedding: second half of the 2048-d output.

        Per the official dinotxt notebooks, the first half aligns with the CLS
        token and is dropped; the second half is what patch tokens match.
        """
        tokens = self.tokenizer.tokenize([text]).to(self.device)
        feats = self.model.encode_text(tokens)[0]
        return torch.nn.functional.normalize(feats[feats.shape[0] // 2:], p=2, dim=0)

    @torch.no_grad()
    def encode_expression(self, expression: str) -> torch.Tensor:
        """Embed a referring expression as an L2-normalized Text Prototype (C,)."""
        debiased = self._embed_text(expression) - self._neutral_mean
        return torch.nn.functional.normalize(debiased, p=2, dim=0)

    @torch.no_grad()
    def encode_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Project a (1, 3, H, W) image to L2-normalized Aligned Features (C, h, w)."""
        _, patch_tokens, _ = self.model.encode_image_with_patch_tokens(
            image.to(self.device)
        )
        h = image.shape[-2] // 16
        w = image.shape[-1] // 16
        feats = patch_tokens[0].reshape(h, w, -1).permute(2, 0, 1)
        return torch.nn.functional.normalize(feats, p=2, dim=0)


def build_dinotxt(device: str = "cuda") -> DinoTxtEncoder:
    """Build the dino.txt encoder from local weights in ``pretrain/``."""
    missing = [p for p in (_DINOTXT_WEIGHTS, _BACKBONE_WEIGHTS, _BPE_VOCAB)
               if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "dino.txt weights not found: " + ", ".join(missing)
            + ". Place the files under pretrain/ (see README)."
        )
    model, tokenizer = torch.hub.load(
        "facebookresearch/dinov3",
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        weights=_DINOTXT_WEIGHTS,
        backbone_weights=_BACKBONE_WEIGHTS,
        bpe_path_or_url=_BPE_VOCAB,
    )
    model.eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return DinoTxtEncoder(model, tokenizer, device)
