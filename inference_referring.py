"""Referring image segmentation inference script with TFRIS."""

import argparse
import datetime
import os
import random
import subprocess
import sys
import time
from os.path import join

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_tfris_from_args


def main(args: argparse.Namespace) -> None:
    print(args)

    # ──────── Reproducibility and logging setup ────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_file = join(args.output_dir, 'log.txt')
    with open(log_file, 'w') as fp:
        fp.write(" ".join(sys.argv) + '\n')
        fp.write(str(vars(args)) + '\n\n')

    # ──────── Model setup ────────
    model = build_tfris_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    start_time = time.time()
    ds = build_dataset(args.dataset, args=args)
    predictions_root = run_inference(args, model, ds, log_file)
    print(f'Total inference time: {time.time() - start_time:.1f}s')

    if args.limit is None:
        run_official_evaluator(args, predictions_root, log_file)
    else:
        validate_predictions(args, ds, predictions_root, log_file)


def run_inference(args: argparse.Namespace, model: torch.nn.Module,
                  ds: torch.utils.data.Dataset, log_file: str) -> str:
    """Segment every expression and write one binary PNG per `sent_id`.

    Returns the predictions root directory (re-evaluable manually at any time).
    """
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        collate_fn=lambda x: x[0])
    predictions_root = join(args.output_dir, 'predictions')
    pred_dir = join(predictions_root, args.dataset, args.split)
    os.makedirs(pred_dir, exist_ok=True)

    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    iou_sum = 0.0

    # ──────── Inference loop ────────
    pbar = tqdm(loader, total=total, ncols=80)
    for idx, batch in enumerate(pbar):
        model.set_target(batch['tgt_img'])
        model.set_text(batch['expression'])
        pred = model.segment().cpu()  # (H, W) bool at source resolution

        Image.fromarray(pred.numpy().astype(np.uint8) * 255, mode='L').save(
            join(pred_dir, f"{batch['sent_id']}.png"))

        # Running IoU estimate — observational only; official numbers come
        # from the evaluator.
        gt = batch['tgt_mask'].bool()
        union = (pred | gt).sum().item()
        iou_sum += 1.0 if union == 0 else (pred & gt).sum().item() / union
        pbar.set_description(f'IoU: {iou_sum / (idx + 1):.3f}')

        if idx + 1 == total:
            break
    pbar.close()

    out_str = (f'Observed mean IoU over {total} expressions: {iou_sum / total:.3f} '
               f'(observational; official metrics come from the evaluator)')
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')

    return predictions_root


def validate_predictions(args: argparse.Namespace, ds, predictions_root: str,
                         log_file: str) -> None:
    """Check the written PNGs against the eval-package prediction contract:
    one file per processed `sent_id`, source-image dimensions, binary content."""
    records = ds.expressions[:args.limit] if args.limit is not None else ds.expressions
    pred_dir = join(predictions_root, args.dataset, args.split)

    expected = {f"{int(r['sent_id'])}.png" for r in records}
    actual = set(os.listdir(pred_dir))
    if expected != actual:
        raise ValueError(f'Prediction set mismatch: missing={sorted(expected - actual)}, '
                         f'extra={sorted(actual - expected)}')
    for r in records:
        arr = np.asarray(Image.open(join(pred_dir, f"{int(r['sent_id'])}.png")).convert('L'))
        if arr.shape != (int(r['height']), int(r['width'])):
            raise ValueError(f"{r['sent_id']}.png shape {arr.shape} != "
                             f"({r['height']}, {r['width']})")
        if not set(np.unique(arr)) <= {0, 255}:
            raise ValueError(f"{r['sent_id']}.png is not binary")

    out_str = (f'Capped run (--limit {args.limit}): official evaluator skipped, '
               f'prediction contract validated on {len(records)} PNGs')
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')


def run_official_evaluator(args: argparse.Namespace, predictions_root: str,
                           log_file: str) -> None:
    """Invoke the eval package's evaluate_predictions.py and append its report
    (expression/instance-weighted mIoU, overall IoU, Precision@0.5/0.7/0.9)."""
    cmd = [sys.executable, join(args.data_root, 'tools', 'evaluate_predictions.py'),
           args.data_root, predictions_root,
           '--dataset', args.dataset, '--split', args.split]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    with open(log_file, 'a') as fp:
        fp.write('Official evaluator report:\n')
        fp.write(result.stdout)
        if result.returncode != 0:
            fp.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'Official evaluator failed (exit {result.returncode}): '
                           f'{result.stderr.strip()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'TFRIS inference on referring image segmentation',
        parents=[opts.get_args_parser()],
    )
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
