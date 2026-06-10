"""Behavioral tests for the dino.txt encoder (Aligned Features + Text Prototype)."""

import pytest
import torch

from models.dinotxt import build_dinotxt

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture(scope="session")
def encoder():
    return build_dinotxt(device="cuda")


def test_missing_weights_raise_descriptive_error(tmp_path, monkeypatch):
    """Construction must fail at startup, naming the missing weight files."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"pretrain"):
        build_dinotxt(device="cpu")


@requires_cuda
def test_encode_expression_returns_unit_norm_text_prototype(encoder):
    proto = encoder.encode_expression("a cat")
    assert proto.shape == (1024,)
    assert torch.isclose(proto.norm(), torch.tensor(1.0, device=proto.device), atol=1e-4)


@requires_cuda
def test_encode_patches_returns_unit_norm_aligned_feature_grid(encoder):
    image = torch.rand(1, 3, 1024, 1024, device="cuda")
    feats = encoder.encode_patches(image)
    assert feats.shape == (1024, 64, 64)
    norms = feats.flatten(1).norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)


@requires_cuda
def test_text_similarity_localizes_referred_object(encoder):
    """Argmax of the patch-vs-text similarity must land inside the GT object."""
    from utils.data import build_transform, downsample_mask, load_image, load_mask

    image, _ = load_image("assets/cat_image.jpg", build_transform(1024), "cuda")
    feats = encoder.encode_patches(image)
    proto = encoder.encode_expression("a cat")
    sim = torch.einsum("c,chw->hw", proto, feats)

    gt = load_mask("assets/cat_mask.png", 1024, "cuda")
    gt_small = downsample_mask(gt.unsqueeze(0), sim.shape[0], sim.shape[1])

    flat_idx = int(sim.flatten().argmax().item())
    y, x = divmod(flat_idx, sim.shape[1])
    assert gt_small[y, x], f"argmax patch ({y}, {x}) falls outside the cat mask"
