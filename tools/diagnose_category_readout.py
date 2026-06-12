"""Expression-side category-readout diagnostic for TFRIS (WS1, issue #26).

Phase-3 module 1 ("what" decomposition) premise check, built diagnostic-first
like the phase-2 expansion diagnostic: tuning package only, sampled with the
project-wide seed 42, markdown report + per-episode JSONL. It does NOT change any
inference code — it only reuses the frozen model's text/feature/seed paths.

Three measurements (issue #26 acceptance criteria):

  1. Expression-side category readout accuracy. Each Referring Expression is
     encoded by the frozen dino.txt text encoder (the existing Text Prototype
     path, Text Bias removed) and scored against a COCO-80 category-name bank;
     the argmax category is compared to the episode's GT `category_name`.
  2. Bucket decomposition. Episodes are split into correct / wrong readout
     buckets; per bucket we report the share, the readout-confidence
     distribution (top-1 similarity and top-1/top-2 margin), the wrong-target
     rate, and the wrong-class rate among wrong-target (phase-2 seed-readout
     proxy). "no-signal" is deliberately NOT a hard bucket: instead of fixing a
     confidence threshold we report a threshold sweep so the HITL gate (#27)
     picks the criterion from the data.
  3. Class-restricted seeding oracle. The seed is re-selected restricted to
     clusters whose zero-shot readout category matches (a) the GT category
     (ORACLE = the module's hard ceiling) and (b) the expression readout
     (realistic variant). All other stages are unchanged (the model's own
     `_seed_and_aggregate` is reused with a restricted candidate mask — only
     seed selection is constrained). The mIoU lift over the v2 seed is reported
     for each.

RED LINE: GT `category_name` is a diagnosis-only input. It is read straight from
the package's `expressions.jsonl` and used ONLY for scoring readout accuracy,
bucketing, and the (clearly labelled) GT-category oracle ceiling. It never enters
the v2 prediction path nor the realistic (expression-readout) oracle. The
standard RIS input stays image + expression only.

Self-contained data caveat: the tuning package ships masks only for its sampled
referred instances, not full per-image COCO instance maps, so both the cluster
category and the seed wrong-class proxy are zero-shot readouts of a region
(aligned features vs COCO category prototypes), not a per-pixel category map. The
GT-category oracle therefore gates clusters by their *readout* matching the true
target category — it is the ceiling of THIS readout-gated module, not of a
perfect per-pixel class oracle. See the report header.

Usage (run through the insid3 env):
  conda run -n insid3 python tools/diagnose_category_readout.py \
      --data-root ../refcoco_tuning_package_seed42 --sample 100 --seed 42 \
      --out output/ws1cat_diag
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from models import build_tfris
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import load_image
from utils.refinement import upsample_mask

# Tuning-package train splits (isomorphic to the eval package).
_PACKAGE_SPLITS = [
    ("refcoco", "train"),
    ("refcoco+", "train"),
    ("refcocog", "train_U"),
    ("refcocog", "train_G"),
]

# Standard COCO-80 thing categories — the zero-shot readout vocabulary. The
# package's GT categories are a verified subset of this list.
COCO_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
_CAT_INDEX = {c: i for i, c in enumerate(COCO_CATEGORIES)}

# Candidate no-signal thresholds, swept (not hard-coded) so the HITL gate can
# pick a class-plausibility / bypass criterion from the data.
_MARGIN_GRID = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
_TOP1_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def downsample_gt(gt: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(gt[None, None].float(), size=(h, w), mode="bilinear")[0, 0] > 0.5


def read_episodes(data_root: str, dataset: str, split: str) -> list[dict]:
    path = os.path.join(data_root, dataset, split, "expressions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_mask(path: str) -> torch.Tensor:
    import numpy as np
    from PIL import Image
    arr = np.array(Image.open(path).convert("L"))
    return torch.from_numpy(arr) > 0


def finalize_iou(mask_feat: torch.Tensor, img: torch.Tensor,
                 orig_size, gt_full: torch.Tensor) -> float:
    """Upsample a feature-res mask to source resolution and IoU it against GT.

    Mirrors `TFRIS._finalize_mask` for the bilinear (no-CRF) path, so the v2
    branch here is pixel-equivalent to the deployed inference output.
    """
    up = upsample_mask(mask_feat, img.shape[-2], img.shape[-1])
    up = upsample_mask(up, orig_size[0], orig_size[1]) > 0.5
    return (up & gt_full).sum().item() / max((up | gt_full).sum().item(), 1)


def diagnose_episode(model, ep: dict, data_root: str, ep_id: str,
                     dataset: str, split: str,
                     cat_protos: torch.Tensor, device: str) -> dict:
    """Run one episode; return its diagnostic record (no GT in any prediction)."""
    gt_full = load_mask(os.path.join(data_root, ep["mask_path"])).to(device)
    img, orig_size = load_image(
        os.path.join(data_root, ep["image_path"]), model._transform, device)
    gt_cat = ep["category_name"]
    gt_in_bank = gt_cat in _CAT_INDEX

    # ── Measurement 1: expression-side category readout (scoring only) ──
    text_proto = model.dinotxt.encode_expression(ep["expression"])
    readout = cat_protos @ text_proto                      # (80,) cosine scores
    order = torch.argsort(readout, descending=True)
    top1_i = int(order[0].item())
    top2_i = int(order[1].item())
    expr_pred_cat = COCO_CATEGORIES[top1_i]
    top1 = readout[top1_i].item()
    margin = top1 - readout[top2_i].item()
    readout_correct = gt_in_bank and (expr_pred_cat == gt_cat)

    base = {
        "ep_id": ep_id, "dataset": dataset, "split": split,
        "sent_id": ep["sent_id"], "expression": ep["expression"],
        "gt_category": gt_cat, "gt_in_bank": gt_in_bank,
        "expr_pred_category": expr_pred_cat,
        "readout_top1": round(top1, 5), "readout_top2": round(readout[top2_i].item(), 5),
        "readout_margin": round(margin, 5), "readout_correct": readout_correct,
        "gt_area_frac": round(gt_full.float().mean().item(), 4),
    }

    # ── Feature extraction + candidate localization (v2, unchanged) ──
    feat_native, feat_aligned = model.dinotxt.encode_image_features(img)
    _, h, w = feat_native.shape
    gt = downsample_gt(gt_full, h, w)
    sim = torch.einsum("c,chw->hw", text_proto, feat_aligned)
    candidate_mask = sim > torch.quantile(sim, model.cand_quantile)

    if candidate_mask.sum() == 0:
        v2_iou = finalize_iou(candidate_mask, img, orig_size, gt_full)
        base.update({"has_seed": False, "K": 0, "v2_iou": round(v2_iou, 3),
                     "oracle_gt_iou": round(v2_iou, 3), "oracle_gt_applicable": False,
                     "oracle_expr_iou": round(v2_iou, 3), "oracle_expr_applicable": False,
                     "wrong_target": False})
        return base

    feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
    labels = agglomerative_clustering(feat_native_flat, model.tau).reshape(h, w)
    K = int(labels.max().item()) + 1
    feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
    protos_aligned = compute_cluster_prototypes(feat_aligned_flat, labels.view(-1), K)

    # ── v2 seed + mask (the deployed path; no GT, no category gate) ──
    matched_ids = labels[candidate_mask & (labels >= 0)].unique()
    cross_sim_matched = protos_aligned[matched_ids] @ text_proto
    seed_id = int(matched_ids[int(torch.argmax(cross_sim_matched).item())].item())
    seed_mask = labels == seed_id
    seed_prec = (seed_mask & gt).sum().item() / max(seed_mask.sum().item(), 1)
    seed_recall = (seed_mask & gt).sum().item() / max(gt.sum().item(), 1)

    v2_feat = model._seed_and_aggregate(
        candidate_mask, labels, protos_aligned, K,
        text_proto, feat_native_flat, sim, h, w)
    v2_iou = finalize_iou(v2_feat, img, orig_size, gt_full)

    # ── Cluster zero-shot category readout (proxy; diagnosis-only target match) ──
    cluster_cat = (protos_aligned @ cat_protos.t()).argmax(dim=1)   # (K,) COCO index

    def restricted_oracle(target_idx: int):
        """Re-seed within candidate clusters whose readout == target_idx.

        Reuses the model's own merge rule via a restricted candidate mask, so
        ONLY seed selection is constrained; aggregation is identical to v2.
        Returns (iou, applicable).
        """
        if target_idx < 0:
            return v2_iou, False
        allowed = cluster_cat == target_idx                         # (K,) bool
        restricted_cand = candidate_mask & allowed[labels]
        if restricted_cand.sum() == 0:
            return v2_iou, False                                    # gate abstains -> v2
        mask_feat = model._seed_and_aggregate(
            restricted_cand, labels, protos_aligned, K,
            text_proto, feat_native_flat, sim, h, w)
        return finalize_iou(mask_feat, img, orig_size, gt_full), True

    gt_idx = _CAT_INDEX.get(gt_cat, -1)
    oracle_gt_iou, oracle_gt_ok = restricted_oracle(gt_idx)         # ORACLE (GT)
    oracle_expr_iou, oracle_expr_ok = restricted_oracle(top1_i)     # realistic

    # ── Phase-2 wrong-target / wrong-class proxy (seed readout vs GT) ──
    wrong_target = seed_prec < 0.3
    seed_class = None
    if wrong_target:
        seed_pred_idx = int((protos_aligned[seed_id] @ cat_protos.t()).argmax().item())
        seed_pred_cat = COCO_CATEGORIES[seed_pred_idx]
        seed_class = "wrong-instance" if (gt_in_bank and seed_pred_cat == gt_cat) else "wrong-class"
        base["seed_pred_category"] = seed_pred_cat

    base.update({
        "has_seed": True, "K": K, "seed_id": seed_id,
        "seed_prec": round(seed_prec, 3), "seed_recall": round(seed_recall, 3),
        "v2_iou": round(v2_iou, 3),
        "oracle_gt_iou": round(oracle_gt_iou, 3), "oracle_gt_applicable": oracle_gt_ok,
        "oracle_expr_iou": round(oracle_expr_iou, 3), "oracle_expr_applicable": oracle_expr_ok,
        "correct_seed": not wrong_target, "wrong_target": wrong_target,
        "seed_class": seed_class,
    })
    return base


# ──────────────────────────── reporting ────────────────────────────

def md_table(headers: list[str], rows: list[list]) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round((q / 100) * (len(s) - 1)))))
    return s[i]


def build_report(episodes: list[dict], args) -> str:
    splits = [f"{d}/{s}" for d, s in _PACKAGE_SPLITS]
    scored = [e for e in episodes if e["gt_in_bank"]]
    n = len(episodes)
    L: list[str] = []

    L.append("## WS1 — Expression-side category-readout diagnostic (issue #26)\n")
    L.append(
        f"**Provenance.** Tuning package `{os.path.basename(args.data_root.rstrip('/'))}`, "
        f"splits {', '.join(splits)}; sample={args.sample} expressions/split "
        f"(seed {args.seed}); device {args.device}; frozen hyperparams "
        f"tau={model_hp(args,'tau')} cand_quantile={model_hp(args,'cand_quantile')} "
        f"merge_threshold={model_hp(args,'merge_threshold')} image_size=1024. "
        f"Total episodes={n} (GT category in COCO-80 bank={len(scored)}). "
        f"Category bank: COCO-80.\n")
    L.append(
        "**Red line.** GT `category_name` is used ONLY to score readout accuracy, "
        "bucket episodes, and define the GT-category oracle ceiling (labelled "
        "ORACLE below). It never enters the v2 prediction path nor the realistic "
        "expression-readout oracle. No `models/`, `datasets/`, or "
        "`inference_referring.py` behaviour was changed.\n")
    L.append(
        "**Proxy caveat.** The package ships no per-pixel COCO instance map, so "
        "cluster categories (and the seed wrong-class proxy) are zero-shot "
        "readouts of a region (aligned features vs COCO-80 prototypes). The "
        "GT-category oracle gates clusters by their *readout* matching the true "
        "target category — it is the hard ceiling of THIS readout-gated module, "
        "not of a perfect per-pixel class oracle.\n")

    # ── Measurement 1: readout accuracy ──
    L.append("### 1. Expression-side category readout accuracy\n")
    L.append("Readout = argmax of the expression's cosine similarity to the "
             "COCO-80 category-name bank; agreement = argmax category equals GT "
             "`category_name`.\n")
    rows = []
    for d, s in _PACKAGE_SPLITS:
        sel = [e for e in scored if e["dataset"] == d and e["split"] == s]
        acc = mean([e["readout_correct"] for e in sel])
        rows.append([f"{d}/{s}", len(sel), f"{acc:.1%}",
                     round(mean([e["readout_top1"] for e in sel]), 3),
                     round(mean([e["readout_margin"] for e in sel]), 3)])
    acc_all = mean([e["readout_correct"] for e in scored])
    rows.append(["**all**", len(scored), f"{acc_all:.1%}",
                 round(mean([e["readout_top1"] for e in scored]), 3),
                 round(mean([e["readout_margin"] for e in scored]), 3)])
    L.append(md_table(["split", "n", "readout agreement",
                       "mean top-1 sim", "mean margin"], rows) + "\n")

    # ── Measurement 2: bucket decomposition ──
    L.append("### 2. Bucket decomposition (correct vs wrong readout)\n")
    L.append("Buckets are by readout correctness only — 'no-signal' is reported "
             "as a confidence sweep below, not a fixed bucket. Wrong-target rate "
             "uses the phase-2 definition (seed precision < 0.3); wrong-class "
             "rate is the share of a bucket's wrong-target episodes whose seed "
             "reads out a category other than GT (phase-2 seed proxy).\n")
    rows = []
    for label, pred in (("correct", True), ("wrong", False)):
        sel = [e for e in scored if e["readout_correct"] == pred]
        wt = [e for e in sel if e.get("wrong_target")]
        wc = sum(1 for e in wt if e.get("seed_class") == "wrong-class")
        rows.append([
            label, len(sel), f"{len(sel)/max(len(scored),1):.0%}",
            round(mean([e["readout_top1"] for e in sel]), 3),
            round(mean([e["readout_margin"] for e in sel]), 3),
            f"{mean([e.get('wrong_target', False) for e in sel]):.0%}",
            f"{wc/max(len(wt),1):.0%}" if wt else "—",
        ])
    L.append(md_table(["readout bucket", "n", "share", "mean top-1 sim",
                       "mean margin", "wrong-target rate",
                       "wrong-class rate (of wrong-target)"], rows) + "\n")

    # confidence distribution per bucket
    L.append("**Readout-confidence distribution per bucket** "
             "(top-1 similarity / top-1−top-2 margin percentiles):\n")
    rows = []
    for label, pred in (("correct", True), ("wrong", False)):
        sel = [e for e in scored if e["readout_correct"] == pred]
        t1 = [e["readout_top1"] for e in sel]
        mg = [e["readout_margin"] for e in sel]
        rows.append([label,
                     *(round(pct(t1, q), 3) for q in (10, 25, 50, 75, 90)),
                     *(round(pct(mg, q), 3) for q in (10, 25, 50, 75, 90))])
    L.append(md_table(["bucket", "t1 p10", "t1 p25", "t1 p50", "t1 p75", "t1 p90",
                       "mg p10", "mg p25", "mg p50", "mg p75", "mg p90"], rows) + "\n")

    # no-signal threshold sweep
    L.append("**No-signal threshold sweep.** For each candidate cutoff, episodes "
             "*below* the cutoff are the would-be no-signal bucket. If low "
             "confidence flags unreliable readout, accuracy-below should be far "
             "under accuracy-above — that gap is the evidence for (or against) a "
             "bypass. (Whole sample, GT-in-bank.)\n")
    L.append("_By top-1/top-2 margin:_\n")
    L.append(_sweep_table(scored, "readout_margin", _MARGIN_GRID) + "\n")
    L.append("_By top-1 similarity:_\n")
    L.append(_sweep_table(scored, "readout_top1", _TOP1_GRID) + "\n")

    # ── Measurement 3: class-restricted seeding oracle ──
    L.append("### 3. Class-restricted seeding oracle\n")
    L.append("Seed re-selected within candidate clusters whose zero-shot readout "
             "matches the target category; all other stages unchanged. "
             "**ORACLE (GT)** uses the true category = the module's hard ceiling; "
             "**realistic (expr)** uses the expression's own readout = no GT. "
             "Δ is the per-episode mIoU change vs the v2 seed, averaged over all "
             "episodes. 'applicable' = a candidate cluster matched the target "
             "category (else the gate abstains and v2 stands).\n")
    rows = []
    for d, s in _PACKAGE_SPLITS:
        rows.append(_oracle_row(f"{d}/{s}",
                    [e for e in scored if e["dataset"] == d and e["split"] == s]))
    rows.append(_oracle_row("**all**", scored))
    L.append(md_table(
        ["split", "n", "v2 mIoU", "ORACLE(GT) mIoU", "ΔGT",
         "GT applic.", "realistic(expr) mIoU", "Δexpr", "expr applic."], rows) + "\n")

    # guard breakdown: correct-seed vs wrong-target
    L.append("**Guard breakdown (the module's risk surface).** A class gate can "
             "only help wrong-target episodes; on correct-seed episodes it can "
             "only hold or regress. The realistic gate is the deployable one, so "
             "its correct-seed Δ is the acceptance guard (#24: no correct-seed "
             "regression).\n")
    rows = []
    for label, sub in (
        ("wrong-target", [e for e in scored if e.get("wrong_target")]),
        ("correct-seed", [e for e in scored if e.get("correct_seed")]),
    ):
        if not sub:
            continue
        rows.append([
            label, len(sub), round(mean([e["v2_iou"] for e in sub]), 3),
            round(mean([e["oracle_gt_iou"] for e in sub]), 3),
            f"{mean([e['oracle_gt_iou'] - e['v2_iou'] for e in sub]):+.3f}",
            round(mean([e["oracle_expr_iou"] for e in sub]), 3),
            f"{mean([e['oracle_expr_iou'] - e['v2_iou'] for e in sub]):+.3f}",
        ])
    L.append(md_table(["subset", "n", "v2 mIoU", "ORACLE(GT) mIoU", "ΔGT",
                       "realistic mIoU", "Δexpr"], rows) + "\n")

    # ── HITL decision points ──
    L.append(_hitl_section(scored))
    return "\n".join(L)


def _sweep_table(scored: list[dict], key: str, grid: list[float]) -> str:
    rows = []
    for t in grid:
        below = [e for e in scored if e[key] < t]
        above = [e for e in scored if e[key] >= t]
        rows.append([
            t, len(below), f"{len(below)/max(len(scored),1):.0%}",
            f"{mean([e['readout_correct'] for e in below]):.0%}" if below else "—",
            f"{mean([e['readout_correct'] for e in above]):.0%}" if above else "—",
        ])
    return md_table(["cutoff", "n below (no-signal)", "share below",
                     "readout acc. below", "readout acc. above"], rows)


def _oracle_row(label: str, sel: list[dict]) -> list:
    if not sel:
        return [label, 0, "—", "—", "—", "—", "—", "—", "—"]
    v2 = mean([e["v2_iou"] for e in sel])
    ogt = mean([e["oracle_gt_iou"] for e in sel])
    oex = mean([e["oracle_expr_iou"] for e in sel])
    gt_ap = mean([e.get("oracle_gt_applicable", False) for e in sel])
    ex_ap = mean([e.get("oracle_expr_applicable", False) for e in sel])
    return [label, len(sel), round(v2, 3), round(ogt, 3), f"{ogt - v2:+.3f}",
            f"{gt_ap:.0%}", round(oex, 3), f"{oex - v2:+.3f}", f"{ex_ap:.0%}"]


def model_hp(args, name: str):
    v = getattr(args, name)
    return v if v is not None else "default"


def _hitl_section(scored: list[dict]) -> str:
    """Open decisions for the HITL gate (#27), each backed by a report number."""
    n = len(scored)
    acc = mean([e["readout_correct"] for e in scored])
    v2 = mean([e["v2_iou"] for e in scored])
    d_gt = mean([e["oracle_gt_iou"] - e["v2_iou"] for e in scored])
    d_ex = mean([e["oracle_expr_iou"] - e["v2_iou"] for e in scored])
    correct_seed = [e for e in scored if e.get("correct_seed")]
    d_ex_cs = mean([e["oracle_expr_iou"] - e["v2_iou"] for e in correct_seed]) if correct_seed else 0.0
    wt = [e for e in scored if e.get("wrong_target")]
    d_ex_wt = mean([e["oracle_expr_iou"] - e["v2_iou"] for e in wt]) if wt else 0.0
    ex_ap = mean([e.get("oracle_expr_applicable", False) for e in scored])

    # low-confidence (would-be no-signal) at a representative margin cutoff
    cut = 0.03
    low = [e for e in scored if e["readout_margin"] < cut]
    acc_low = mean([e["readout_correct"] for e in low]) if low else float("nan")
    acc_high = mean([e["readout_correct"] for e in scored if e["readout_margin"] >= cut]) or 0.0

    L = ["### Open decisions for the HITL gate (#27)\n"]
    L.append(
        f"1. **Build the module at all?** Realistic expression-readout gating "
        f"lifts tuning mIoU {d_ex:+.3f} (v2 {v2:.3f} → {v2 + d_ex:.3f}); the "
        f"GT-category oracle ceiling is {d_gt:+.3f}. PRD acceptance floor is "
        f"+0.03. **Decision:** proceed only if the realistic Δ (or a thresholded "
        f"variant from the sweep) clears the floor with margin; if Δexpr is below "
        f"floor while ΔGT is large, the bottleneck is readout accuracy "
        f"({acc:.0%}), not the gate — reconsider the category source before "
        f"building.\n")
    L.append(
        f"2. **Class-plausibility criterion (parameter-free vs thresholded).** "
        f"Plain argmax-match gating gives realistic Δexpr {d_ex:+.3f} at "
        f"{ex_ap:.0%} applicability. The margin sweep above shows readout "
        f"accuracy below vs above each cutoff (e.g. margin<{cut}: {acc_low:.0%} "
        f"vs {acc_high:.0%}). **Decision:** prefer the parameter-free argmax-hit "
        f"set unless the sweep shows a cutoff that removes mostly-wrong readouts "
        f"without discarding correct ones — pick the criterion from the sweep gap, "
        f"not a guessed threshold.\n")
    L.append(
        f"3. **No-signal bypass — needed or not?** Of {n} episodes, "
        f"{len(low)} ({len(low)/max(n,1):.0%}) fall below margin {cut} with "
        f"readout accuracy {acc_low:.0%} (vs {acc_high:.0%} above). A bypass "
        f"(skip the gate when confidence is low, fall back to v2) is justified "
        f"only if this bucket is both large and low-accuracy. **Decision:** add a "
        f"bypass only if the low-confidence share × its accuracy gap is material; "
        f"otherwise ship gate-always (the realistic oracle already abstains when "
        f"no cluster matches, {1 - ex_ap:.0%} of the time).\n")
    L.append(
        f"4. **Correct-seed safety (the only regression risk).** Realistic gating "
        f"changes correct-seed mIoU by {d_ex_cs:+.3f} (n={len(correct_seed)}) and "
        f"wrong-target by {d_ex_wt:+.3f} (n={len(wt)}). PRD requires no "
        f"correct-seed regression. **Decision:** if Δexpr on correct-seed is "
        f"negative, the gate must abstain on already-correct seeds (e.g. keep the "
        f"v2 seed when it already lies in a class-plausible cluster) before WS2 "
        f"implementation, not after.\n")
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../refcoco_tuning_package_seed42")
    parser.add_argument("--sample", type=int, default=100,
                        help="expressions sampled per split (seeded)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/ws1cat_diag",
                        help="prefix for <out>_episodes.jsonl / _report.md")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--cand-quantile", type=float, default=None)
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild <out>_report.md from existing jsonl, no GPU pass")
    args = parser.parse_args()

    # Reports/logs carry non-cp936 glyphs (−, Δ, →); keep console echo from
    # crashing on the default Windows GBK stdout. Files are already UTF-8.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.report_only:
        with open(f"{args.out}_episodes.jsonl", encoding="utf-8") as f:
            episodes = [json.loads(line) for line in f]
        report = build_report(episodes, args)
        with open(f"{args.out}_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return

    build_kwargs = {"device": args.device}
    if args.tau is not None:
        build_kwargs["tau"] = args.tau
    if args.cand_quantile is not None:
        build_kwargs["cand_quantile"] = args.cand_quantile
    if args.merge_threshold is not None:
        build_kwargs["merge_threshold"] = args.merge_threshold
    model = build_tfris(**build_kwargs)

    # COCO category text bank (diagnosis-only, built once via the frozen text path).
    cat_protos = torch.stack(
        [model.dinotxt.encode_expression(c) for c in COCO_CATEGORIES])

    rng = random.Random(args.seed)
    episodes: list[dict] = []
    for dataset, split in _PACKAGE_SPLITS:
        recs = read_episodes(args.data_root, dataset, split)
        if args.sample and args.sample < len(recs):
            recs = rng.sample(recs, args.sample)
        for j, ep in enumerate(recs):
            ep_id = f"{dataset}/{split}/{ep['sent_id']}"
            with torch.no_grad():
                base = diagnose_episode(model, ep, args.data_root, ep_id,
                                        dataset, split, cat_protos, args.device)
            episodes.append(base)
            print(f"[{dataset}/{split} {j+1}/{len(recs)}] "
                  f"readout={base['expr_pred_category']:14s} "
                  f"gt={base['gt_category']:14s} ok={base['readout_correct']!s:5s} "
                  f"v2={base.get('v2_iou', 0):.3f} "
                  f"oGT={base.get('oracle_gt_iou', 0):.3f} "
                  f"oEx={base.get('oracle_expr_iou', 0):.3f} "
                  f"expr={ep['expression'][:32]!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_episodes.jsonl", "w", encoding="utf-8") as f:
        for r in episodes:
            f.write(json.dumps(r) + "\n")
    report = build_report(episodes, args)
    with open(f"{args.out}_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nWrote {args.out}_episodes.jsonl, {args.out}_report.md")


if __name__ == "__main__":
    main()
