"""Merge-threshold calibration for the WS2 soft product score (issue #15).

Calibrates the threshold of the redesigned cluster-aggregation rule

    score_k = intra_sim_k * cross_sim_norm_k > threshold

on the tuning package only (`../refcoco_tuning_package_seed42`). Each episode's
per-cluster score is computed once (one backbone + clustering pass); the
threshold is then swept cheaply, so the whole curve comes from a single pass.

Two threshold forms are evaluated:
  - global constant: score_k > t
  - per-episode adaptive: score_k > alpha * max_k(score_k)

mIoU is the expression-weighted full-resolution IoU, measured through the same
upsample path the model uses in `_finalize_mask` (no CRF), so the selected
threshold transfers to `inference_referring.py`.

RED LINE: tuning package only — never the eval package. GT masks are used for
scoring the curve, not fed to the model.

Usage:
  conda run -n insid3 python tools/calibrate_merge_threshold.py \
      --data-root ../refcoco_tuning_package_seed42 --out output/ws2_calib
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from models import build_tfris
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import load_image
from utils.refinement import upsample_mask

_PACKAGE_SPLITS = [
    ("refcoco", "train"),
    ("refcoco+", "train"),
    ("refcocog", "train_U"),
    ("refcocog", "train_G"),
]


def read_episodes(data_root: str, dataset: str, split: str) -> list[dict]:
    path = os.path.join(data_root, dataset, split, "expressions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_mask(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"))
    return torch.from_numpy(arr) > 0


@torch.no_grad()
def episode_score(model, ep: dict, data_root: str, device: str):
    """One backbone + clustering pass; return everything needed to sweep thresholds.

    Returns (cluster_labels(h,w), seed_id, score(K,), tgt_image, orig_size, gt_full)
    or None if the episode has no candidates (mask is empty regardless of threshold).
    """
    gt_full = load_mask(os.path.join(data_root, ep["mask_path"])).to(device)
    img, orig_size = load_image(
        os.path.join(data_root, ep["image_path"]), model._transform, device)

    text_proto = model.dinotxt.encode_expression(ep["expression"])
    feat_native, feat_aligned = model.dinotxt.encode_image_features(img)
    _, h, w = feat_native.shape

    sim = torch.einsum("c,chw->hw", text_proto, feat_aligned)
    candidate_mask = sim > torch.quantile(sim, model.cand_quantile)
    if candidate_mask.sum() == 0:
        return None, img, orig_size, gt_full

    feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
    labels = agglomerative_clustering(feat_native_flat, model.tau).reshape(h, w)
    K = int(labels.max().item()) + 1
    feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
    cluster_protos = compute_cluster_prototypes(feat_aligned_flat, labels.view(-1), K)

    matched_mask = candidate_mask & (labels >= 0)
    if matched_mask.sum() == 0:
        return None, img, orig_size, gt_full
    matched_ids = labels[matched_mask].unique()
    cross_sim_matched = cluster_protos[matched_ids] @ text_proto
    seed_id = int(matched_ids[int(torch.argmax(cross_sim_matched).item())].item())

    native_protos = compute_cluster_prototypes(feat_native_flat, labels.view(-1), K)
    intra_sim = torch.einsum("c,kc->k", native_protos[seed_id], native_protos)
    cross_sim = torch.empty(K, device=device, dtype=sim.dtype)
    for k in range(K):
        idx = labels == k
        cross_sim[k] = sim[idx].mean() if idx.any() else 0.0
    cross_sim_norm = (cross_sim - cross_sim.min()) / (cross_sim.max() - cross_sim.min() + 1e-8)
    score = intra_sim * cross_sim_norm
    return (labels, seed_id, score), img, orig_size, gt_full


def mask_iou(keep: torch.Tensor, labels: torch.Tensor, seed_id: int,
             img: torch.Tensor, orig_size, gt_full: torch.Tensor) -> float:
    """Build the feature-res mask from a per-cluster keep flag, finalize, score IoU."""
    h, w = labels.shape
    final = torch.zeros(h, w, dtype=torch.bool, device=labels.device)
    valid = labels >= 0
    final[valid] = keep[labels[valid]]
    final |= labels == seed_id
    up = upsample_mask(final, img.shape[-2], img.shape[-1])
    up = upsample_mask(up, orig_size[0], orig_size[1])
    inter = (up & gt_full).sum().item()
    union = (up | gt_full).sum().item()
    return 1.0 if union == 0 else inter / union


def empty_iou(gt_full: torch.Tensor) -> float:
    return 1.0 if gt_full.sum().item() == 0 else 0.0


def md_table(headers: list[str], rows: list[list]) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="../refcoco_tuning_package_seed42")
    p.add_argument("--sample", type=int, default=None,
                   help="expressions per split (default: full split, matching #12)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="output/ws2_calib")
    args = p.parse_args()

    model = build_tfris(device=args.device)

    const_grid = [round(t, 3) for t in np.arange(0.0, 0.605, 0.02).tolist()]
    alpha_grid = [round(a, 3) for a in np.arange(0.0, 0.95, 0.05).tolist()]

    # per-split accumulators: split -> {"const": {t: [ious]}, "adapt": {a: [ious]}}
    splits = [f"{d}/{s}" for d, s in _PACKAGE_SPLITS]
    const_iou = {sp: {t: [] for t in const_grid} for sp in splits}
    adapt_iou = {sp: {a: [] for a in alpha_grid} for sp in splits}

    rng = random.Random(args.seed)
    for dataset, split in _PACKAGE_SPLITS:
        sp = f"{dataset}/{split}"
        recs = read_episodes(args.data_root, dataset, split)
        if args.sample and args.sample < len(recs):
            recs = rng.sample(recs, args.sample)
        for j, ep in enumerate(recs):
            with torch.no_grad():
                packed, img, orig_size, gt_full = episode_score(
                    model, ep, args.data_root, args.device)
            if packed is None:
                e = empty_iou(gt_full)
                for t in const_grid:
                    const_iou[sp][t].append(e)
                for a in alpha_grid:
                    adapt_iou[sp][a].append(e)
            else:
                labels, seed_id, score = packed
                smax = float(score.max().item())
                for t in const_grid:
                    keep = score > t
                    const_iou[sp][t].append(
                        mask_iou(keep, labels, seed_id, img, orig_size, gt_full))
                for a in alpha_grid:
                    keep = score > a * smax
                    adapt_iou[sp][a].append(
                        mask_iou(keep, labels, seed_id, img, orig_size, gt_full))
            if (j + 1) % 25 == 0:
                print(f"[{sp}] {j+1}/{len(recs)}")

    def miou(d):  # d: {key: [ious]} -> {key: mean}
        return {k: float(np.mean(v)) if v else 0.0 for k, v in d.items()}

    def overall(per_split):  # combine all splits' raw ious
        merged = {}
        for sp in splits:
            for k, v in per_split[sp].items():
                merged.setdefault(k, []).extend(v)
        return miou(merged)

    const_per_split = {sp: miou(const_iou[sp]) for sp in splits}
    adapt_per_split = {sp: miou(adapt_iou[sp]) for sp in splits}
    const_all = overall(const_iou)
    adapt_all = overall(adapt_iou)

    best_t = max(const_grid, key=lambda t: const_all[t])
    best_a = max(alpha_grid, key=lambda a: adapt_all[a])

    # ── Report ──
    lines = ["## WS2 — merge-threshold calibration (issue #15)\n"]
    n_per = {sp: len(next(iter(const_iou[sp].values()))) for sp in splits}
    lines.append(f"**Provenance.** Tuning package "
                 f"`{os.path.basename(args.data_root.rstrip('/'))}`, "
                 f"splits {', '.join(splits)}; "
                 f"{'full split' if not args.sample else f'sample={args.sample} (seed {args.seed})'}; "
                 f"episodes/split " + ", ".join(f"{sp}={n_per[sp]}" for sp in splits)
                 + f"; device {args.device}; rule `score_k = intra_sim_k * cross_sim_norm_k`. "
                 f"mIoU = expression-weighted full-res IoU.\n")

    lines.append("### Global constant threshold — mIoU vs threshold\n")
    rows = [[t] + [round(const_per_split[sp][t], 4) for sp in splits]
            + [round(const_all[t], 4)] for t in const_grid]
    lines.append(md_table(["threshold"] + splits + ["all"], rows) + "\n")
    lines.append(f"**Best global constant: threshold = {best_t}** "
                 f"(overall mIoU {const_all[best_t]:.4f}).\n")

    lines.append("### Per-episode adaptive threshold (alpha * max score) — mIoU vs alpha\n")
    rows = [[a] + [round(adapt_per_split[sp][a], 4) for sp in splits]
            + [round(adapt_all[a], 4)] for a in alpha_grid]
    lines.append(md_table(["alpha"] + splits + ["all"], rows) + "\n")
    lines.append(f"**Best adaptive: alpha = {best_a}** "
                 f"(overall mIoU {adapt_all[best_a]:.4f}).\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    with open(f"{args.out}_curves.json", "w", encoding="utf-8") as f:
        json.dump({"const_per_split": const_per_split, "const_all": const_all,
                   "adapt_per_split": adapt_per_split, "adapt_all": adapt_all,
                   "best_const": best_t, "best_adapt": best_a,
                   "episodes_per_split": n_per}, f, indent=2)
    print("\n" + report)
    print(f"\nWrote {args.out}_report.md, {args.out}_curves.json")


if __name__ == "__main__":
    main()
