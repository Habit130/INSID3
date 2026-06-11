"""Behavioral tests for the TFRIS model (text-to-mask path)."""

import pytest
import torch

from models import build_tfris

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture(scope="session")
def model():
    return build_tfris(device="cuda")


@pytest.fixture(autouse=True)
def clean_state(request):
    """A failed segment() leaves state set; start every test from a clean slate."""
    if "model" in request.fixturenames:
        request.getfixturevalue("model").reset_state()


@requires_cuda
def test_text_to_mask_returns_boolean_mask_at_source_resolution(model):
    """README minimal usage: one image + one Referring Expression -> (H, W) bool mask."""
    from PIL import Image

    source = Image.open("assets/cat_image.jpg")
    model.set_target("assets/cat_image.jpg")
    model.set_text("a cat")
    pred = model.segment()

    assert pred.dtype == torch.bool
    assert pred.shape == (source.height, source.width)


@requires_cuda
def test_prediction_overlaps_referred_object(model):
    """The predicted mask for "a cat" must substantially overlap the GT cat mask."""
    import numpy as np
    from PIL import Image

    model.set_target("assets/cat_image.jpg")
    model.set_text("a cat")
    pred = model.segment().cpu().numpy()

    gt = np.array(Image.open("assets/cat_mask.png")) > 0
    iou = (pred & gt).sum() / (pred | gt).sum()
    assert iou > 0.25, f"IoU with GT cat mask too low: {iou:.3f}"


@requires_cuda
def test_state_resets_after_segment(model):
    """segment() consumes the episode; a second call must demand fresh inputs."""
    model.set_target("assets/cat_image.jpg")
    model.set_text("a cat")
    model.segment()

    with pytest.raises(RuntimeError):
        model.segment()


@requires_cuda
def test_crf_refinement_produces_mask(model):
    """mask_refiner="crf" must still run end-to-end (skipped if CRF is not installed)."""
    pytest.importorskip("CRF")
    from PIL import Image

    from models.tfris import TFRIS

    crf_model = TFRIS(dinotxt=model.dinotxt, mask_refiner="crf", device="cuda")
    source = Image.open("assets/cat_image.jpg")
    crf_model.set_target("assets/cat_image.jpg")
    crf_model.set_text("a cat")
    pred = crf_model.segment()

    assert pred.dtype == torch.bool
    assert pred.shape == (source.height, source.width)


def test_visualization_saves_target_expression_and_mask(tmp_path):
    """The referring visualization helper writes an image file; no GPU needed."""
    import numpy as np

    from utils.visualization import visualize_prediction_referring

    mask = np.zeros((480, 640), dtype=bool)
    mask[100:300, 200:400] = True
    out = tmp_path / "viz" / "pred.png"
    visualize_prediction_referring(
        "assets/cat_image.jpg", "a cat", mask, out
    )

    assert out.is_file() and out.stat().st_size > 0


def test_build_tfris_from_args_uses_parser_defaults(monkeypatch):
    """The args builder wires parser defaults into the model; dino.txt loading is
    stubbed out (torch.hub boundary) so this runs without weights or GPU."""
    import models
    from opts import get_args_parser

    monkeypatch.setattr(models, "build_dinotxt", lambda device: object())
    args = get_args_parser().parse_args([])

    model = models.build_tfris_from_args(args)

    assert args.cand_quantile == 0.9
    assert model.cand_quantile == 0.9
    assert model.tau == 0.6
    assert model.merge_threshold == 0.32
    assert model.resize_to_orig_size is True


@requires_cuda
def test_segment_without_target_raises(model):
    model.set_text("a cat")
    with pytest.raises(RuntimeError, match="set_target"):
        model.segment()


@requires_cuda
def test_segment_without_text_raises(model):
    model.set_target("assets/cat_image.jpg")
    with pytest.raises(RuntimeError, match="set_text"):
        model.segment()
