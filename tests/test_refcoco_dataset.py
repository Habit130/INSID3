"""RefCOCO-family dataset module: reconciliation against the eval package."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from datasets import build_dataset

PKG_ROOT = Path(__file__).resolve().parents[2] / "refcoco_eval_package"

pytestmark = pytest.mark.skipif(
    not PKG_ROOT.is_dir(), reason="refcoco eval package not found"
)


def make_args(dataset: str, split: str) -> SimpleNamespace:
    return SimpleNamespace(dataset=dataset, split=split, data_root=str(PKG_ROOT))


def test_refcoco_val_has_one_item_per_expression():
    dataset = build_dataset("refcoco", make_args("refcoco", "val"))
    assert len(dataset) == 10834


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


def test_unknown_split_fails_with_clear_error():
    with pytest.raises(ValueError, match="split"):
        build_dataset("refcoco", make_args("refcoco", "testC"))
    with pytest.raises(ValueError, match="split"):
        build_dataset("refcocog", make_args("refcocog", "testA"))


def test_unknown_dataset_fails_with_clear_error():
    with pytest.raises(ValueError, match="dataset"):
        build_dataset("refcoco_wrong", make_args("refcoco_wrong", "val"))


def test_opts_accept_refcoco_datasets_and_split():
    from opts import get_args_parser

    parser = get_args_parser()
    args = parser.parse_args(
        ["--dataset", "refcoco+", "--split", "testA", "--data-root", str(PKG_ROOT)]
    )
    assert args.dataset == "refcoco+"
    assert args.split == "testA"

    assert parser.parse_args([]).split == "val"


def test_item_exposes_image_expression_mask_and_sent_id():
    import PIL.Image
    import torch

    dataset = build_dataset("refcoco", make_args("refcoco", "val"))
    item = dataset[0]

    assert isinstance(item["tgt_img"], PIL.Image.Image)
    assert isinstance(item["expression"], str) and item["expression"]
    assert isinstance(item["sent_id"], int)

    mask = item["tgt_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (item["tgt_img"].height, item["tgt_img"].width)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0


def test_works_with_repo_dataloader_convention():
    import PIL.Image
    from torch.utils.data import DataLoader

    dataset = build_dataset("refcocog", make_args("refcocog", "val"))
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    episode = next(iter(loader))

    assert isinstance(episode["tgt_img"], PIL.Image.Image)
    assert isinstance(episode["expression"], str)
