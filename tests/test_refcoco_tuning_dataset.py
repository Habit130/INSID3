"""RefCOCO-family dataset module: reconciliation against the tuning package.

Mirrors test_refcoco_dataset.py, but points at the seed-42 tuning package and
its train splits (train / train_U / train_G). Skipped when the package is absent.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from datasets import build_dataset

PKG_ROOT = Path(__file__).resolve().parents[2] / "refcoco_tuning_package_seed42"

pytestmark = pytest.mark.skipif(
    not PKG_ROOT.is_dir(), reason="refcoco tuning package not found"
)


def make_args(dataset: str, split: str) -> SimpleNamespace:
    return SimpleNamespace(dataset=dataset, split=split, data_root=str(PKG_ROOT))


def _all_splits():
    report = json.loads((PKG_ROOT / "validation_report.json").read_text())
    for key, counts in report["split_counts"].items():
        dataset, split = key.split("/")
        yield pytest.param(dataset, split, counts, id=key)


@pytest.mark.parametrize("dataset_name,split,counts", _all_splits())
def test_reconciles_with_validation_report(dataset_name, split, counts):
    dataset = build_dataset(dataset_name, make_args(dataset_name, split))

    assert len(dataset) == counts["expressions"]

    sent_ids = dataset.sent_ids()
    assert len(sent_ids) == counts["expressions"]
    assert len(set(sent_ids)) == counts["expressions"]

    records = dataset.expressions
    assert len({r["ann_id"] for r in records}) == counts["instances"]
    assert len({r["image_id"] for r in records}) == counts["images"]


def test_train_split_item_exposes_image_expression_mask_and_sent_id():
    import PIL.Image
    import torch

    dataset = build_dataset("refcoco", make_args("refcoco", "train"))
    item = dataset[0]

    assert isinstance(item["tgt_img"], PIL.Image.Image)
    assert isinstance(item["expression"], str) and item["expression"]
    assert isinstance(item["sent_id"], int)

    mask = item["tgt_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (item["tgt_img"].height, item["tgt_img"].width)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0
