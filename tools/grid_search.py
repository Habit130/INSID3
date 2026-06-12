"""Hyperparameter grid over tau x cand_quantile x merge_threshold (WS3, issue #16).

Sweeps the three inference hyperparameters on the tuning package only
(`../refcoco_tuning_package_seed42`) and reports, for every `tau`, both the
oracle cluster-subset IoU ceiling and the actual mIoU — `tau` sets the
clustering granularity, which moves the ceiling, so coarser/finer clustering is
judged by both numbers.

Cost structure: the backbone pass (the expensive stage) runs once per episode
and is reused across the whole grid. For each `tau` the native features are
re-clustered once (and the oracle ceiling + per-cluster text term computed once);
for each `cand_quantile` the candidate mask + seed are recomputed; the
`merge_threshold` axis is then a cheap per-cluster score threshold. So one
backbone pass per episode covers |tau| clusterings and |tau|*|q|*|t| mask scorings.

Conventions match the existing tools so numbers are comparable:
  - actual mIoU = expression-weighted full-resolution IoU through the model's
    own upsample path (`_finalize_mask`, no CRF), exactly as
    `tools/calibrate_merge_threshold.py`.
  - oracle ceiling = unconstrained greedy cluster-subset IoU at feature
    resolution against the downsampled GT, exactly as
    `tools/diagnose_expansion.py` (the ~0.67 ceiling reported in WS1).
  The two live in different resolutions by design — same as WS1's stage table.

RED LINE: tuning package only — never the eval package. GT masks score the grid;
they are never fed to the model.

Usage:
  conda run -n insid3 python tools/grid_search.py \
      --data-root ../refcoco_tuning_package_seed42 --sample 100 --seed 42 \
      --out output/ws3_grid
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
import torch.nn.functional as F
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

# WS2 default cell (ADR 0003); kept in the grid so the comparison is exact.
DEFAULT_TAU = 0.6
DEFAULT_Q = 0.9
DEFAULT_T = 0.32


def read_episodes(data_root: str, dataset: str, split: str) -> list[dict]:
    path = os.path.join(data_root, dataset, split, "expressions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_mask(path: str) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("L"))
    return torch.from_numpy(arr) > 0


def downsample_gt(gt: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(gt[None, None].float(), size=(h, w), mode="bilinear")[0, 0] > 0.5


def greedy_oracle_iou(labels: torch.Tensor, gt: torch.Tensor, K: int) -> float:
    """Unconstrained greedy cluster-subset IoU ceiling (matches diagnose_expansion)."""
    stats = []
    for k in range(K):
        m = labels == k
        area = m.sum().item()
        inter = (m & gt).sum().item()
        stats.append((k, inter, area))
    stats.sort(key=lambda t: t[1] / max(t[2], 1), reverse=True)
    sel = torch.zeros_like(gt)
    best = 0.0
    for k, inter, _area in stats:
        if inter == 0:
            break
        cand = sel | (labels == k)
        iou = (cand & gt).sum().item() / max((cand | gt).sum().item(), 1)
        if iou > best:
            best, sel = iou, cand
    return best


def finalize_iou(feat_mask: torch.Tensor, img: torch.Tensor, orig_size,
                 gt_full: torch.Tensor) -> float:
    """Upsample a feature-res mask through the model's path and score full-res IoU."""
    up = upsample_mask(feat_mask, img.shape[-2], img.shape[-1])
    up = upsample_mask(up, orig_size[0], orig_size[1])
    inter = (up & gt_full).sum().item()
    union = (up | gt_full).sum().item()
    return 1.0 if union == 0 else inter / union


def md_table(headers: list[str], rows: list[list]) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


@torch.no_grad()
def episode_features(model, ep: dict, data_root: str, device: str):
    """One backbone pass; everything the grid needs, independent of hyperparams."""
    gt_full = load_mask(os.path.join(data_root, ep["mask_path"])).to(device)
    img, orig_size = load_image(
        os.path.join(data_root, ep["image_path"]), model._transform, device)
    text_proto = model.dinotxt.encode_expression(ep["expression"])
    feat_native, feat_aligned = model.dinotxt.encode_image_features(img)
    _, h, w = feat_native.shape
    sim = torch.einsum("c,chw->hw", text_proto, feat_aligned)
    feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
    feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
    gt_down = downsample_gt(gt_full, h, w)
    return {
        "img": img, "orig_size": orig_size, "gt_full": gt_full, "gt_down": gt_down,
        "text_proto": text_proto, "sim": sim, "h": h, "w": w,
        "native_flat": feat_native_flat, "aligned_flat": feat_aligned_flat,
    }


@torch.no_grad()
def cluster_for_tau(f: dict, tau: float, device: str):
    """Cluster native features at tau; return per-tau quantities (oracle + text term)."""
    h, w = f["h"], f["w"]
    labels = agglomerative_clustering(f["native_flat"], tau).reshape(h, w)
    K = int(labels.max().item()) + 1
    native_protos = compute_cluster_prototypes(f["native_flat"], labels.view(-1), K)
    aligned_protos = compute_cluster_prototypes(f["aligned_flat"], labels.view(-1), K)

    sim = f["sim"]
    cross_sim = torch.empty(K, device=device, dtype=sim.dtype)
    for k in range(K):
        idx = labels == k
        cross_sim[k] = sim[idx].mean() if idx.any() else 0.0
    cross_sim_norm = (cross_sim - cross_sim.min()) / (cross_sim.max() - cross_sim.min() + 1e-8)

    oracle = greedy_oracle_iou(labels, f["gt_down"], K)
    return {"labels": labels, "K": K, "native_protos": native_protos,
            "aligned_protos": aligned_protos, "cross_sim_norm": cross_sim_norm,
            "oracle": oracle}


@torch.no_grad()
def scores_for_quantile(f: dict, c: dict, q: float):
    """Candidate mask + seed at quantile q; return (per-cluster score, seed mask) or None.

    Returns None when there is no candidate-covered cluster, i.e. the model's
    prediction is the candidate mask itself (replicated by the caller).
    """
    sim, labels = f["sim"], c["labels"]
    cand = sim > torch.quantile(sim, q)
    if cand.sum() == 0:
        return None, cand
    matched = cand & (labels >= 0)
    if matched.sum() == 0:
        return None, cand
    matched_ids = labels[matched].unique()
    cross_sim_matched = c["aligned_protos"][matched_ids] @ f["text_proto"]
    seed_id = int(matched_ids[int(torch.argmax(cross_sim_matched).item())].item())
    intra_sim = torch.einsum("c,kc->k", c["native_protos"][seed_id], c["native_protos"])
    score = intra_sim * c["cross_sim_norm"]
    return (score, seed_id), cand


def mask_from_keep(keep: torch.Tensor, labels: torch.Tensor, seed_id: int,
                   h: int, w: int) -> torch.Tensor:
    final = torch.zeros(h, w, dtype=torch.bool, device=labels.device)
    valid = labels >= 0
    final[valid] = keep[labels[valid]]
    final |= labels == seed_id
    return final


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="../refcoco_tuning_package_seed42")
    p.add_argument("--sample", type=int, default=100,
                   help="expressions per split (seeded); None = full split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="output/ws3_grid")
    p.add_argument("--tau-grid", type=float, nargs="+",
                   default=[0.5, 0.6, 0.7, 0.8])
    p.add_argument("--quantile-grid", type=float, nargs="+",
                   default=[0.85, 0.9, 0.95])
    p.add_argument("--thresh-grid", type=float, nargs="+",
                   default=[0.24, 0.28, 0.32, 0.36, 0.40])
    args = p.parse_args()

    tau_grid = [round(t, 4) for t in args.tau_grid]
    q_grid = [round(q, 4) for q in args.quantile_grid]
    t_grid = [round(t, 4) for t in args.thresh_grid]
    for name, grid, val in (("tau", tau_grid, DEFAULT_TAU),
                            ("quantile", q_grid, DEFAULT_Q),
                            ("thresh", t_grid, DEFAULT_T)):
        if val not in grid:
            print(f"[warn] default {name}={val} not in grid; default-cell "
                  f"comparison will be unavailable.")

    model = build_tfris(device=args.device)

    splits = [f"{d}/{s}" for d, s in _PACKAGE_SPLITS]
    # accumulators: per split
    oracle_iou = {sp: {tau: [] for tau in tau_grid} for sp in splits}
    grid_iou = {sp: {(tau, q, t): []
                     for tau in tau_grid for q in q_grid for t in t_grid}
                for sp in splits}

    rng = random.Random(args.seed)
    for dataset, split in _PACKAGE_SPLITS:
        sp = f"{dataset}/{split}"
        recs = read_episodes(args.data_root, dataset, split)
        if args.sample and args.sample < len(recs):
            recs = rng.sample(recs, args.sample)
        for j, ep in enumerate(recs):
            with torch.no_grad():
                f = episode_features(model, ep, args.data_root, args.device)
            for tau in tau_grid:
                c = cluster_for_tau(f, tau, args.device)
                oracle_iou[sp][tau].append(c["oracle"])
                for q in q_grid:
                    packed, cand = scores_for_quantile(f, c, q)
                    if packed is None:
                        # Model's prediction is the candidate mask itself.
                        iou = finalize_iou(cand, f["img"], f["orig_size"], f["gt_full"])
                        for t in t_grid:
                            grid_iou[sp][(tau, q, t)].append(iou)
                    else:
                        score, seed_id = packed
                        for t in t_grid:
                            keep = score > t
                            m = mask_from_keep(keep, c["labels"], seed_id, f["h"], f["w"])
                            grid_iou[sp][(tau, q, t)].append(
                                finalize_iou(m, f["img"], f["orig_size"], f["gt_full"]))
            if (j + 1) % 25 == 0:
                print(f"[{sp}] {j+1}/{len(recs)}")

    # ── Aggregate ──
    def mean(v):
        return float(np.mean(v)) if v else 0.0

    def overall(per_split_lists):
        merged = []
        for sp in splits:
            merged.extend(per_split_lists[sp])
        return mean(merged)

    oracle_per_split = {sp: {tau: mean(oracle_iou[sp][tau]) for tau in tau_grid}
                        for sp in splits}
    oracle_all = {tau: overall({sp: oracle_iou[sp][tau] for sp in splits})
                  for tau in tau_grid}
    grid_per_split = {sp: {cell: mean(grid_iou[sp][cell]) for cell in grid_iou[sp]}
                      for sp in splits}
    grid_all = {cell: overall({sp: grid_iou[sp][cell] for sp in splits})
                for cell in grid_iou[splits[0]]}

    best_cell = max(grid_all, key=lambda cell: grid_all[cell])
    default_cell = (DEFAULT_TAU, DEFAULT_Q, DEFAULT_T)
    n_per = {sp: len(oracle_iou[sp][tau_grid[0]]) for sp in splits}

    # ── Report ──
    lines = ["## WS3 — hyperparameter grid (issue #16)\n"]
    lines.append(
        f"**Provenance.** Tuning package "
        f"`{os.path.basename(args.data_root.rstrip('/'))}`, splits "
        f"{', '.join(splits)}; "
        f"{'full split' if not args.sample else f'sample={args.sample} (seed {args.seed})'}; "
        f"episodes/split " + ", ".join(f"{sp}={n_per[sp]}" for sp in splits)
        + f"; device {args.device}. Grid: tau={tau_grid} x "
        f"cand_quantile={q_grid} x merge_threshold={t_grid}. "
        f"actual mIoU = expression-weighted full-res IoU (model upsample path, no "
        f"CRF); oracle = unconstrained greedy cluster-subset IoU at feature "
        f"resolution (per-tau ceiling, matches WS1). RED LINE: tuning package only.\n")

    # Oracle ceiling vs tau
    lines.append("### Oracle cluster-subset IoU ceiling vs tau\n")
    lines.append("Higher tau = finer clustering (lower merge distance) = more "
                 "clusters. The ceiling is the best achievable by any subset of "
                 "clusters, so it bounds what tuning the merge rule can reach.\n")
    rows = [[tau] + [round(oracle_per_split[sp][tau], 4) for sp in splits]
            + [round(oracle_all[tau], 4)] for tau in tau_grid]
    lines.append(md_table(["tau"] + splits + ["all"], rows) + "\n")

    # Full grid: per tau, a q x t sub-table of overall mIoU
    lines.append("### Actual mIoU grid (overall, all splits combined)\n")
    for tau in tau_grid:
        lines.append(f"**tau = {tau}** (oracle ceiling {oracle_all[tau]:.4f})\n")
        rows = [[q] + [round(grid_all[(tau, q, t)], 4) for t in t_grid]
                for q in q_grid]
        lines.append(md_table(["q \\ t"] + [str(t) for t in t_grid], rows) + "\n")

    # Best config + attribution vs WS2 default
    lines.append("### Best configuration vs WS2 default\n")
    bt, bq, bth = best_cell
    rows = [
        ["WS2 default", DEFAULT_TAU, DEFAULT_Q, DEFAULT_T,
         round(grid_all[default_cell], 4) if default_cell in grid_all else "—",
         round(oracle_all[DEFAULT_TAU], 4) if DEFAULT_TAU in oracle_all else "—"],
        ["grid best", bt, bq, bth, round(grid_all[best_cell], 4),
         round(oracle_all[bt], 4)],
    ]
    lines.append(md_table(
        ["config", "tau", "cand_quantile", "merge_thresh", "overall mIoU",
         "oracle ceiling"], rows) + "\n")
    if default_cell in grid_all:
        delta = grid_all[best_cell] - grid_all[default_cell]
        lines.append(
            f"**Tuning gain (attributable to WS3): {delta:+.4f} overall mIoU** "
            f"({grid_all[default_cell]:.4f} default -> {grid_all[best_cell]:.4f} best). "
            f"This is the grid-search delta only; the WS2 mechanism-fix gain over "
            f"the v1 baseline is reported separately in ADR 0003.\n")

    # Per-split best config
    lines.append("### Best configuration per split\n")
    rows = []
    for sp in splits:
        cell = max(grid_per_split[sp], key=lambda cc: grid_per_split[sp][cc])
        dval = (round(grid_per_split[sp][default_cell], 4)
                if default_cell in grid_per_split[sp] else "—")
        rows.append([sp, cell[0], cell[1], cell[2],
                     round(grid_per_split[sp][cell], 4), dval])
    lines.append(md_table(
        ["split", "tau", "cand_quantile", "merge_thresh", "best mIoU",
         "default-cell mIoU"], rows) + "\n")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_report.md", "w", encoding="utf-8") as fh:
        fh.write(report)
    serializable = {
        "oracle_per_split": oracle_per_split, "oracle_all": oracle_all,
        "grid_per_split": {sp: {f"{c[0]},{c[1]},{c[2]}": v
                                for c, v in grid_per_split[sp].items()}
                           for sp in splits},
        "grid_all": {f"{c[0]},{c[1]},{c[2]}": v for c, v in grid_all.items()},
        "best_cell": list(best_cell), "default_cell": list(default_cell),
        "tau_grid": tau_grid, "quantile_grid": q_grid, "thresh_grid": t_grid,
        "episodes_per_split": n_per,
    }
    with open(f"{args.out}_grid.json", "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    print("\n" + report)
    print(f"\nWrote {args.out}_report.md, {args.out}_grid.json")


if __name__ == "__main__":
    main()
