"""RefCOCO-family referring image segmentation datasets (self-contained eval package)."""
from __future__ import annotations

import json
import os

from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np

_SPLITS = {
    'refcoco': ('val', 'testA', 'testB'),
    'refcoco+': ('val', 'testA', 'testB'),
    'refcocog': ('val', 'test_U', 'test_G'),
}


class DatasetRefCOCO(Dataset):
    """One item per Referring Expression, read from the eval package root."""

    def __init__(self, datapath: str, dataset: str, split: str):
        if dataset not in _SPLITS:
            raise ValueError(f'Unknown RefCOCO dataset: {dataset}. '
                             f'Supported: {list(_SPLITS.keys())}')
        if split not in _SPLITS[dataset]:
            raise ValueError(f'Unknown split for {dataset}: {split}. '
                             f'Supported: {list(_SPLITS[dataset])}')
        self.benchmark = dataset
        self.split = split
        self.base_path = datapath

        expressions_file = os.path.join(datapath, dataset, split, 'expressions.jsonl')
        with open(expressions_file, encoding='utf-8') as f:
            self.expressions = [json.loads(line) for line in f]

    def __len__(self) -> int:
        return len(self.expressions)

    def sent_ids(self) -> list[int]:
        return [record['sent_id'] for record in self.expressions]

    def __getitem__(self, idx: int) -> dict:
        record = self.expressions[idx]

        tgt_img = Image.open(os.path.join(self.base_path, record['image_path'])).convert('RGB')
        tgt_mask = self.read_mask(os.path.join(self.base_path, record['mask_path']))

        return {'tgt_img': tgt_img,
                'tgt_mask': tgt_mask.float(),
                'expression': record['expression'],
                'sent_id': record['sent_id']}

    def read_mask(self, mask_path: str) -> torch.Tensor:
        # Package convention: background 0, any non-zero pixel is foreground.
        mask = torch.tensor(np.array(Image.open(mask_path).convert('L')))
        return (mask > 0).long()


def build(args) -> DatasetRefCOCO:
    return DatasetRefCOCO(datapath=args.data_root, dataset=args.dataset, split=args.split)
