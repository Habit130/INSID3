"""Referring inference script: prediction contract and evaluator closure.

The model is stubbed at the torch.hub boundary (build_tfris_from_args), so these
tests run without GPU or weights; everything else (parser, dataset registry,
inference loop, PNG layout, evaluator subprocess) is the real code path.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import opts

# (width, height) per expression — distinct sizes catch dimension mix-ups.
SIZES = [(32, 24), (48, 40), (56, 32)]


@pytest.fixture
def fake_package(tmp_path):
    """Minimal eval-package layout: refcoco/val with one record per SIZES entry,
    GT masks all-foreground so an all-True prediction scores IoU 1.0."""
    pkg = tmp_path / "pkg"
    img_dir = pkg / "images" / "train2014"
    img_dir.mkdir(parents=True)
    split_dir = pkg / "refcoco" / "val"
    mask_dir = split_dir / "masks"
    mask_dir.mkdir(parents=True)

    rows = []
    for i, (w, h) in enumerate(SIZES):
        image_name = f"COCO_train2014_{i:012d}.jpg"
        Image.new("RGB", (w, h), (10 * i, 100, 50)).save(img_dir / image_name)
        mask_name = f"ann_{i:012d}.png"
        Image.fromarray(np.full((h, w), 255, dtype=np.uint8), "L").save(mask_dir / mask_name)
        rows.append({
            "dataset": "refcoco", "split": "val",
            "ann_id": i, "image_id": i,
            "image_path": f"images/train2014/{image_name}",
            "mask_path": f"refcoco/val/masks/{mask_name}",
            "width": w, "height": h,
            "sent_id": 100 + i, "expression": f"object {i}",
        })
    (split_dir / "expressions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return pkg


class StubModel:
    """Mimics the TFRIS stateful API; segments everything (all-True mask)."""

    def __init__(self):
        self._size = None

    def set_target(self, image):
        self._size = (image.height, image.width)

    def set_text(self, expression):
        pass

    def segment(self):
        h, w = self._size
        self._size = None
        return torch.ones(h, w, dtype=torch.bool)

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])


def make_args(pkg, out_dir, extra=()):
    out_dir.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(parents=[opts.get_args_parser()])
    return parser.parse_args([
        "--dataset", "refcoco", "--split", "val",
        "--data-root", str(pkg), "--output-dir", str(out_dir),
        "--device", "cpu", *extra,
    ])


def run_main(args, monkeypatch):
    import inference_referring

    monkeypatch.setattr(inference_referring, "build_tfris_from_args", lambda a: StubModel())
    inference_referring.main(args)


def test_capped_run_writes_contract_satisfying_pngs(fake_package, tmp_path, monkeypatch):
    """Capped run: one binary PNG per processed expression, named by sent_id,
    at source-image dimensions, under predictions/<dataset>/<split>/."""
    args = make_args(fake_package, tmp_path / "run", ["--limit", "2"])
    run_main(args, monkeypatch)

    pred_dir = tmp_path / "run" / "predictions" / "refcoco" / "val"
    assert sorted(p.name for p in pred_dir.glob("*.png")) == ["100.png", "101.png"]
    for i, (w, h) in enumerate(SIZES[:2]):
        arr = np.asarray(Image.open(pred_dir / f"{100 + i}.png").convert("L"))
        assert arr.shape == (h, w)
        assert set(np.unique(arr)) <= {0, 255}


def test_capped_run_validates_contract_and_skips_evaluator(fake_package, tmp_path, monkeypatch):
    """With --limit the official evaluator must not run (it requires complete
    splits); the log records contract validation instead."""
    args = make_args(fake_package, tmp_path / "run", ["--limit", "2"])
    run_main(args, monkeypatch)

    log = (tmp_path / "run" / "log.txt").read_text(encoding="utf-8")
    assert "contract" in log.lower()
    assert "expression_weighted" not in log  # evaluator report absent


def test_inline_iou_estimate_in_progress_bar_and_log(fake_package, tmp_path, monkeypatch, capfd):
    """The tqdm bar carries a running IoU estimate (all-True predictions vs
    all-foreground GT -> 1.000); the observed mean also lands in the log,
    marked observational."""
    args = make_args(fake_package, tmp_path / "run", ["--limit", "3"])
    run_main(args, monkeypatch)

    assert "IoU: 1.000" in capfd.readouterr().err  # tqdm renders on stderr
    log = (tmp_path / "run" / "log.txt").read_text(encoding="utf-8")
    assert "IoU" in log and "1.000" in log


@pytest.mark.parametrize("argv", [
    ["--fold", "0"],
    ["--shots", "1"],
    ["--svd-comps", "500"],
])
def test_removed_arguments_are_rejected(argv):
    parser = argparse.ArgumentParser(parents=[opts.get_args_parser()])
    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_supported_datasets_are_the_refcoco_family():
    assert opts.SUPPORTED_DATASETS == ["refcoco", "refcoco+", "refcocog"]

    parser = argparse.ArgumentParser(parents=[opts.get_args_parser()])
    assert parser.parse_args([]).dataset == "refcoco"
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset", "coco"])


REAL_PKG = Path(__file__).resolve().parents[2] / "refcoco_eval_package"


@pytest.mark.skipif(not REAL_PKG.is_dir(), reason="refcoco eval package not found")
def test_uncapped_run_invokes_official_evaluator_and_logs_report(fake_package, tmp_path, monkeypatch):
    """Without --limit the official evaluate_predictions.py runs as a subprocess;
    its report lands in the run log. All-True predictions against all-foreground
    GT masks must score mIoU 1.0."""
    tools = fake_package / "tools"
    tools.mkdir()
    for name in ("evaluate_predictions.py", "package_common.py"):
        shutil.copy(REAL_PKG / "tools" / name, tools / name)

    args = make_args(fake_package, tmp_path / "run")
    run_main(args, monkeypatch)

    log = (tmp_path / "run" / "log.txt").read_text(encoding="utf-8")
    assert "expression_weighted" in log
    assert "overall_IoU" in log
    report_text = log[log.index("Official evaluator report:"):]
    report = json.loads(report_text[report_text.index("{"):])
    result = report["results"][0]
    assert (result["dataset"], result["split"]) == ("refcoco", "val")
    assert result["expression_weighted"]["mIoU"] == 1.0
    assert result["expression_weighted"]["Precision@0.5"] == 1.0
