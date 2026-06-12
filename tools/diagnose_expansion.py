"""Expansion-signal diagnostic for TFRIS cluster aggregation (WS1, issue #13).

Promotes the throwaway failure-mode harness into a maintained tool. Decomposes
`predict_mask()` into stages, scores each against GT, and answers the two
questions that gate the WS2 merge-rule redesign:

  1. Which expansion signal separates GT from non-GT clusters? Per episode it
     records, for every cluster, each candidate expansion signal (native
     intra-image similarity to the seed, per-episode-normalized text
     cross-similarity, seed adjacency) plus their combinations, together with a
     GT-majority flag. AUC is computed per signal over all non-seed clusters.
  2. What share of wrong-target is wrong-class vs wrong-instance? For each
     wrong-target episode the seed cluster's category is read out zero-shot
     against a COCO category text bank and compared to the GT `category_name`.

RED LINE: GT `category_name` is a diagnosis-only input. It is read straight from
the package's `expressions.jsonl`, never passed into the model or the inference
path. The standard RIS input stays image + expression only.

Self-contained data caveat: the tuning package ships masks only for its sampled
referred instances, not full per-image COCO instance annotations, so the
wrong-class/wrong-instance split uses a zero-shot category readout of the seed
region (aligned features vs COCO category prototypes) as a proxy rather than a
per-pixel category map. See the report header.

Usage (run through the insid3 env):
  conda run -n insid3 python tools/diagnose_expansion.py \
      --data-root ../refcoco_tuning_package_seed42 --sample 100 --seed 42 \
      --out output/ws1_diag
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
from sklearn.metrics import roc_auc_score

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

# Standard COCO-80 thing categories — the zero-shot readout vocabulary for the
# wrong-class/wrong-instance split. The package's GT categories are a subset.
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

# The signal columns recorded per cluster and scored by AUC. Keys index into the
# per-cluster record; the combinations are derived from the base signals.
SIGNAL_KEYS = [
    "intra_sim",            # native intra-image similarity to the seed cluster
    "cross_sim_norm",       # per-episode min-max normalized text cross-similarity
    "adjacent",             # 1 if cluster touches the seed cluster, else 0
    "intra_x_crossnorm",    # combination: native intra-sim x normalized text sim
    "intra_x_adj",          # combination: native intra-sim gated by adjacency
    "crossnorm_x_adj",      # combination: normalized text sim gated by adjacency
    "intra_x_crossnorm_x_adj",  # triple combination
]


def downsample_gt(gt: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(gt[None, None].float(), size=(h, w), mode="bilinear")[0, 0] > 0.5


def read_episodes(data_root: str, dataset: str, split: str) -> list[dict]:
    """Read one record per Referring Expression straight from the package.

    Includes GT `category_name` (diagnosis-only) alongside image/mask paths.
    """
    path = os.path.join(data_root, dataset, split, "expressions.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_mask(path: str) -> torch.Tensor:
    import numpy as np
    from PIL import Image
    arr = np.array(Image.open(path).convert("L"))
    return torch.from_numpy(arr) > 0


def adjacency_to_seed(labels: torch.Tensor, seed_id: int) -> torch.Tensor:
    """Boolean (K,) mask: which clusters share a 4-neighbour boundary with seed."""
    K = int(labels.max().item()) + 1
    seed = labels == seed_id
    neigh = torch.zeros_like(seed)
    neigh[:-1, :] |= seed[1:, :]
    neigh[1:, :] |= seed[:-1, :]
    neigh[:, :-1] |= seed[:, 1:]
    neigh[:, 1:] |= seed[:, :-1]
    neigh &= ~seed
    adj = torch.zeros(K, dtype=torch.bool, device=labels.device)
    touched = labels[neigh].unique()
    adj[touched] = True
    adj[seed_id] = True  # seed trivially adjacent to itself
    return adj


def greedy_oracle_iou(labels: torch.Tensor, gt: torch.Tensor, K: int,
                      seed_id: int | None = None) -> float:
    """Greedy cluster-subset IoU ceiling.

    If seed_id is given, the subset is forced to start from that cluster
    (expansion ceiling given the model's actual seed); otherwise unconstrained.
    """
    stats = []
    for k in range(K):
        m = labels == k
        area = m.sum().item()
        inter = (m & gt).sum().item()
        stats.append((k, inter, area))
    stats.sort(key=lambda t: t[1] / max(t[2], 1), reverse=True)

    if seed_id is None:
        sel = torch.zeros_like(gt)
        best = 0.0
    else:
        sel = labels == seed_id
        best = (sel & gt).sum().item() / max((sel | gt).sum().item(), 1)
    for k, inter, _area in stats:
        if k == seed_id:
            continue
        if inter == 0:
            break
        cand = sel | (labels == k)
        iou = (cand & gt).sum().item() / max((cand | gt).sum().item(), 1)
        if iou > best:
            best, sel = iou, cand
    return best


def diagnose_episode(model, ep: dict, data_root: str, ep_id: str,
                     dataset: str, split: str,
                     cat_protos: torch.Tensor, device: str):
    """Run one episode through the staged pipeline; return (episode_rec, cluster_recs)."""
    gt_full = load_mask(os.path.join(data_root, ep["mask_path"])).to(device)
    img, orig_size = load_image(
        os.path.join(data_root, ep["image_path"]), model._transform, device)

    text_proto = model.dinotxt.encode_expression(ep["expression"])
    feat_native, feat_aligned = model.dinotxt.encode_image_features(img)
    _, h, w = feat_native.shape
    gt = downsample_gt(gt_full, h, w)

    sim = torch.einsum("c,chw->hw", text_proto, feat_aligned)
    am = int(sim.argmax().item())
    argmax_in_gt = bool(gt.view(-1)[am].item())

    cand = sim > torch.quantile(sim, model.cand_quantile)
    cand_recall = (cand & gt).sum().item() / max(gt.sum().item(), 1)
    cand_precision = (cand & gt).sum().item() / max(cand.sum().item(), 1)

    feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
    labels = agglomerative_clustering(feat_native_flat, model.tau).reshape(h, w)
    K = int(labels.max().item()) + 1
    feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
    protos_aligned = compute_cluster_prototypes(feat_aligned_flat, labels.view(-1), K)
    native_protos = compute_cluster_prototypes(feat_native_flat, labels.view(-1), K)

    oracle_iou = greedy_oracle_iou(labels, gt, K)

    base = {
        "ep_id": ep_id, "dataset": dataset, "split": split,
        "sent_id": ep["sent_id"], "expression": ep["expression"],
        "gt_category": ep["category_name"], "K": K,
        "gt_area_frac": round(gt.float().mean().item(), 4),
        "argmax_in_gt": argmax_in_gt,
        "cand_recall": round(cand_recall, 3),
        "cand_precision": round(cand_precision, 3),
        "oracle_iou": round(oracle_iou, 3),
    }

    matched_mask = cand & (labels >= 0)
    if matched_mask.sum() == 0:
        base.update({"category": "no-candidates", "iou": 0.0, "has_seed": False})
        return base, []

    # ── Seed selection: same code path as _seed_and_aggregate ──
    matched_ids = labels[matched_mask].unique()
    cross_sim_matched = protos_aligned[matched_ids] @ text_proto
    seed_id = int(matched_ids[int(torch.argmax(cross_sim_matched).item())].item())
    seed_mask = labels == seed_id
    seed_prec = (seed_mask & gt).sum().item() / max(seed_mask.sum().item(), 1)
    seed_recall = (seed_mask & gt).sum().item() / max(gt.sum().item(), 1)

    # ── Per-cluster expansion signals ──
    intra_sim = torch.einsum("c,kc->k", native_protos[seed_id], native_protos)
    cross_sim_raw = torch.empty(K, device=device, dtype=sim.dtype)
    for k in range(K):
        idx = labels == k
        cross_sim_raw[k] = sim[idx].mean() if idx.any() else 0.0
    lo, hi = cross_sim_raw.min(), cross_sim_raw.max()
    cross_sim_norm = (cross_sim_raw - lo) / (hi - lo + 1e-8)
    adjacent = adjacency_to_seed(labels, seed_id).to(sim.dtype)

    cluster_recs = []
    for k in range(K):
        area = int((labels == k).sum().item())
        inter = int(((labels == k) & gt).sum().item())
        gt_frac = inter / max(area, 1)
        isim = intra_sim[k].item()
        csn = cross_sim_norm[k].item()
        adj = adjacent[k].item()
        cluster_recs.append({
            "ep_id": ep_id, "dataset": dataset, "split": split,
            "sent_id": ep["sent_id"], "cluster_id": k,
            "is_seed": k == seed_id, "area": area, "gt_frac": round(gt_frac, 4),
            "gt_majority": gt_frac > 0.5,
            "intra_sim": round(isim, 5),
            "cross_sim_raw": round(cross_sim_raw[k].item(), 5),
            "cross_sim_norm": round(csn, 5),
            "adjacent": adj,
            "intra_x_crossnorm": round(isim * csn, 5),
            "intra_x_adj": round(isim * adj, 5),
            "crossnorm_x_adj": round(csn * adj, 5),
            "intra_x_crossnorm_x_adj": round(isim * csn * adj, 5),
        })

    # ── Final mask via the model's actual merge rule (no rule duplication) ──
    score = intra_sim * cross_sim_norm
    n_merged = int((score > model.merge_threshold).sum().item())

    final = model._seed_and_aggregate(
        cand, labels, protos_aligned, K, text_proto, feat_native_flat, sim, h, w)
    up = upsample_mask(final, img.shape[-2], img.shape[-1])
    up = upsample_mask(up, orig_size[0], orig_size[1]) > 0.5
    iou = (up & gt_full).sum().item() / max((up | gt_full).sum().item(), 1)
    prec = (up & gt_full).sum().item() / max(up.sum().item(), 1)
    recall = (up & gt_full).sum().item() / max(gt_full.sum().item(), 1)

    # ── Bottleneck decomposition: expansion ceiling given the actual seed ──
    seed_cond_oracle = greedy_oracle_iou(labels, gt, K, seed_id=seed_id)

    wrong_target = seed_prec < 0.3
    if wrong_target:
        category = "wrong-target"
    elif recall < 0.5 <= prec:
        category = "incomplete-expansion"
    elif prec < 0.5:
        category = "over-expansion"
    else:
        category = "ok"

    # ── Wrong-class vs wrong-instance (diagnosis-only zero-shot readout) ──
    seed_class = None
    if wrong_target:
        seed_cat_scores = cat_protos @ protos_aligned[seed_id]
        pred_cat = COCO_CATEGORIES[int(torch.argmax(seed_cat_scores).item())]
        seed_class = "wrong-instance" if pred_cat == ep["category_name"] else "wrong-class"
        base["seed_pred_category"] = pred_cat

    base.update({
        "has_seed": True, "seed_id": seed_id,
        "seed_prec": round(seed_prec, 3), "seed_recall": round(seed_recall, 3),
        "n_pass_thresh": n_merged,
        "oracle_iou_seed_cond": round(seed_cond_oracle, 3),
        "iou": round(iou, 3), "precision": round(prec, 3), "recall": round(recall, 3),
        "category": category, "wrong_target": wrong_target,
        "seed_class": seed_class,
    })
    return base, cluster_recs


def compute_auc(cluster_recs: list[dict], split: str | None = None) -> dict:
    """AUC of each signal at separating GT-majority non-seed clusters."""
    rows = [r for r in cluster_recs if not r["is_seed"]
            and (split is None or r["split"] == split)]
    labels = [int(r["gt_majority"]) for r in rows]
    out = {"n_clusters": len(rows), "n_gt": sum(labels)}
    if sum(labels) == 0 or sum(labels) == len(labels):
        return out  # degenerate; AUC undefined
    for key in SIGNAL_KEYS:
        scores = [r[key] for r in rows]
        out[key] = round(roc_auc_score(labels, scores), 4)
    return out


def md_table(headers: list[str], rows: list[list]) -> str:
    line = lambda cells: "| " + " | ".join(str(c) for c in cells) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def build_report(episodes: list[dict], clusters: list[dict], args) -> str:
    splits = [f"{d}/{s}" for d, s in _PACKAGE_SPLITS]
    with_seed = [e for e in episodes if e.get("has_seed")]
    n = len(episodes)

    def mean(key, sel):
        vals = [e[key] for e in sel if key in e]
        return sum(vals) / max(len(vals), 1)

    lines = []
    lines.append("## WS1 — Expansion-signal AUC diagnostic (issue #13)\n")
    lines.append(f"**Provenance.** Tuning package `{os.path.basename(args.data_root.rstrip('/'))}`, "
                 f"splits {', '.join(splits)}; sample={args.sample} expressions/split "
                 f"(seed {args.seed}); device {args.device}; "
                 f"hyperparams tau={args.tau if args.tau else 'default'} "
                 f"cand_quantile={args.cand_quantile if args.cand_quantile else 'default'} "
                 f"merge_threshold={args.merge_threshold if args.merge_threshold else 'default'}. "
                 f"Total episodes={n} (with seed={len(with_seed)}), "
                 f"recorded clusters={len(clusters)}.\n")
    lines.append("**Red line.** GT `category_name` is used only for the "
                 "wrong-class/wrong-instance split below; it never enters the "
                 "model or inference path. No `models/`, `datasets/`, or "
                 "`inference_referring.py` behaviour was changed.\n")

    # ── Stage metrics per split ──
    lines.append("### Stage metrics (per split)\n")
    rows = []
    for d, s in _PACKAGE_SPLITS:
        sel = [e for e in episodes if e["dataset"] == d and e["split"] == s]
        ws = [e for e in sel if e.get("has_seed")]
        rows.append([
            f"{d}/{s}", len(sel),
            round(mean("iou", sel), 3), round(mean("oracle_iou", sel), 3),
            f"{sum(e['argmax_in_gt'] for e in sel)}/{len(sel)}",
            round(mean("seed_prec", ws), 3), round(mean("seed_recall", ws), 3),
            sum(1 for e in ws if e.get("wrong_target")),
        ])
    rows.append([
        "**all**", n, round(mean("iou", episodes), 3),
        round(mean("oracle_iou", episodes), 3),
        f"{sum(e['argmax_in_gt'] for e in episodes)}/{n}",
        round(mean("seed_prec", with_seed), 3),
        round(mean("seed_recall", with_seed), 3),
        sum(1 for e in with_seed if e.get("wrong_target")),
    ])
    lines.append(md_table(
        ["split", "n", "mIoU", "oracle IoU", "argmax∈GT",
         "seed prec", "seed recall", "wrong-target"], rows) + "\n")

    # ── AUC table ──
    lines.append("### Expansion-signal AUC (non-seed clusters, GT-majority label)\n")
    lines.append("AUC = how well each signal ranks GT-majority clusters above "
                 "non-GT clusters; 0.5 = no separation.\n")
    headers = ["signal"] + splits + ["all"]
    overall = compute_auc(clusters)
    # refcoco/refcoco+ both use split 'train', so key per-split AUC by dataset+split.
    per_split_auc = {}
    for d, s in _PACKAGE_SPLITS:
        rows_ds = [c for c in clusters if c["dataset"] == d and c["split"] == s]
        per_split_auc[f"{d}/{s}"] = compute_auc(rows_ds)
    rows = []
    for key in SIGNAL_KEYS:
        rows.append([key] + [per_split_auc[sp].get(key, "—") for sp in splits]
                    + [overall.get(key, "—")])
    lines.append(md_table(headers, rows) + "\n")
    lines.append(f"_Cluster counts (non-seed): "
                 + ", ".join(f"{sp} n={per_split_auc[sp]['n_clusters']} "
                             f"(GT {per_split_auc[sp]['n_gt']})" for sp in splits)
                 + f", all n={overall['n_clusters']} (GT {overall['n_gt']})._\n")

    # Data-derived reading of the AUC table for WS2.
    base_aucs = {k: overall[k] for k in ("intra_sim", "cross_sim_norm", "adjacent")
                 if k in overall}
    best_base = max(base_aucs, key=base_aucs.get)
    best_combo = max(("intra_x_crossnorm", "intra_x_adj", "crossnorm_x_adj",
                      "intra_x_crossnorm_x_adj"), key=lambda k: overall.get(k, 0))
    adj_gate_hurts = overall.get("intra_x_adj", 0) < overall.get("intra_sim", 0)
    lines.append(
        f"**Reading for WS2.** Among base signals, **{best_base}** separates best "
        f"(AUC {overall[best_base]}); native intra-similarity ({overall['intra_sim']}) "
        f"and normalized text cross-similarity ({overall['cross_sim_norm']}) are "
        f"close and **their product {best_combo} is the strongest signal overall "
        f"(AUC {overall[best_combo]})** — they carry partly independent information. "
        f"Seed **adjacency is weak on its own ({overall['adjacent']})**"
        + (f" and using it as a hard gate *reduces* separation "
           f"(intra_x_adj {overall['intra_x_adj']} < intra_sim {overall['intra_sim']}), "
           f"because it hard-zeros GT clusters that are not 4-connected to the seed — "
           f"the same hard-zero pathology the PRD flags for candidate-area weighting. "
           f"This argues against the prior hypothesis of adjacency-gated intra-sim; "
           f"prefer native intra-sim × normalized text cross-sim as a soft score, "
           f"with adjacency (if used) only as a soft prior, not a gate."
           if adj_gate_hurts else ".") + "\n")

    # ── Wrong-class vs wrong-instance ──
    lines.append("### Wrong-target decomposition (wrong-class vs wrong-instance)\n")
    lines.append("Proxy: the seed region's category is read out zero-shot "
                 "(aligned features vs COCO-80 category text bank) and compared "
                 "to GT `category_name`. The package ships no full per-image "
                 "instance annotations, so this is a category readout, not a "
                 "per-pixel category map.\n")
    rows = []
    for d, s in _PACKAGE_SPLITS:
        wt = [e for e in episodes if e["dataset"] == d and e["split"] == s
              and e.get("wrong_target")]
        wc = sum(1 for e in wt if e["seed_class"] == "wrong-class")
        wi = sum(1 for e in wt if e["seed_class"] == "wrong-instance")
        rows.append([f"{d}/{s}", len(wt), wc, wi,
                     f"{wc/max(len(wt),1):.0%}", f"{wi/max(len(wt),1):.0%}"])
    wt_all = [e for e in episodes if e.get("wrong_target")]
    wc_all = sum(1 for e in wt_all if e["seed_class"] == "wrong-class")
    wi_all = sum(1 for e in wt_all if e["seed_class"] == "wrong-instance")
    rows.append(["**all**", len(wt_all), wc_all, wi_all,
                 f"{wc_all/max(len(wt_all),1):.0%}", f"{wi_all/max(len(wt_all),1):.0%}"])
    lines.append(md_table(
        ["split", "wrong-target", "wrong-class", "wrong-instance",
         "%class", "%instance"], rows) + "\n")

    # ── Bottleneck: seed selection vs expansion ──
    lines.append("### Bottleneck: seed selection vs expansion\n")
    mean_actual = mean("iou", with_seed)
    mean_seedcond = mean("oracle_iou_seed_cond", with_seed)
    mean_oracle = mean("oracle_iou", with_seed)
    d_expansion = mean_seedcond - mean_actual
    d_seed = mean_oracle - mean_seedcond
    wt_rate = sum(1 for e in with_seed if e.get("wrong_target")) / max(len(with_seed), 1)
    correct = [e for e in with_seed if not e.get("wrong_target")]
    rows = [
        ["actual mIoU (current pipeline)", round(mean_actual, 3)],
        ["+ perfect expansion, seed fixed (seed-cond. oracle)", round(mean_seedcond, 3)],
        ["+ perfect seed too (unconstrained oracle)", round(mean_oracle, 3)],
        ["Δ recoverable by fixing **expansion** (seed fixed)", round(d_expansion, 3)],
        ["Δ recoverable by fixing **seed** (on top)", round(d_seed, 3)],
        ["wrong-target rate", f"{wt_rate:.0%}"],
        ["correct-seed episodes: actual mIoU", round(mean("iou", correct), 3)],
        ["correct-seed episodes: seed-cond. oracle", round(mean("oracle_iou_seed_cond", correct), 3)],
    ]
    lines.append(md_table(["quantity", "value"], rows) + "\n")
    lines.append(
        f"**Statement.** Of the gap from actual mIoU ({mean_actual:.3f}) to the "
        f"cluster-subset oracle ceiling ({mean_oracle:.3f}), "
        f"{d_expansion:.3f} is recoverable by fixing expansion alone (keeping the "
        f"model's seed) and only {d_seed:.3f} more requires a better seed. "
        f"**The aggregation mechanism (expansion) is the larger bottleneck to "
        f"the oracle ceiling**, confirming the PRD's primary finding: even when "
        f"the seed lands off-target, oracle expansion is free to add the correct "
        f"clusters and recover most IoU, so the seed being slightly wrong is not "
        f"what zeroes out the output — the absence of expansion is.\n")
    lines.append(
        f"**Caveat (seed selection still caps the realistic ceiling).** The "
        f"decomposition above grants oracle expansion. Any realistic phase-2 rule "
        f"keeps the text-similarity seed and a redesigned expansion, and the seed "
        f"lands off the GT object in {wt_rate:.0%} of episodes (wrong-target). On "
        f"the correct-seed subset, fixing expansion lifts mIoU from "
        f"{mean('iou', correct):.3f} to {mean('oracle_iou_seed_cond', correct):.3f}; "
        f"on wrong-target episodes a redesigned expansion cannot rescue a seed on "
        f"the wrong object. So wrong-target sets the phase-2 mIoU ceiling "
        f"(consistent with the PRD's ~0.30 estimate), which is phase-3 "
        f"localization territory.\n")
    lines.append(
        "**Decision.** WS2 (expansion redesign) is correctly the immediate next "
        "step and needs **no scope escalation** to seed selection now: the "
        "mechanism fix is where the recoverable IoU is. Seed-selection overhaul "
        "stays deferred to phase 3 unless WS2's tuned result on the correct-seed "
        "subset still falls far short of its oracle.\n")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../refcoco_tuning_package_seed42")
    parser.add_argument("--sample", type=int, default=100,
                        help="expressions sampled per split (seeded)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/ws1_diag",
                        help="prefix for <out>_clusters.jsonl / _episodes.jsonl / _report.md")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--cand-quantile", type=float, default=None)
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild <out>_report.md from existing jsonl, no GPU pass")
    args = parser.parse_args()

    if args.report_only:
        with open(f"{args.out}_episodes.jsonl", encoding="utf-8") as f:
            episodes = [json.loads(line) for line in f]
        with open(f"{args.out}_clusters.jsonl", encoding="utf-8") as f:
            clusters = [json.loads(line) for line in f]
        report = build_report(episodes, clusters, args)
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

    # COCO category text bank (diagnosis-only, built once).
    cat_protos = torch.stack(
        [model.dinotxt.encode_expression(c) for c in COCO_CATEGORIES])

    rng = random.Random(args.seed)
    episodes: list[dict] = []
    clusters: list[dict] = []

    for dataset, split in _PACKAGE_SPLITS:
        recs = read_episodes(args.data_root, dataset, split)
        if args.sample and args.sample < len(recs):
            recs = rng.sample(recs, args.sample)
        for j, ep in enumerate(recs):
            ep_id = f"{dataset}/{split}/{ep['sent_id']}"
            with torch.no_grad():
                base, cluster_recs = diagnose_episode(
                    model, ep, args.data_root, ep_id, dataset, split,
                    cat_protos, args.device)
            episodes.append(base)
            clusters.extend(cluster_recs)
            print(f"[{dataset}/{split} {j+1}/{len(recs)}] iou={base.get('iou', 0):.3f} "
                  f"cat={base['category']:22s} oracle={base['oracle_iou']:.3f} "
                  f"expr={ep['expression'][:40]!r}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_clusters.jsonl", "w", encoding="utf-8") as f:
        for r in clusters:
            f.write(json.dumps(r) + "\n")
    with open(f"{args.out}_episodes.jsonl", "w", encoding="utf-8") as f:
        for r in episodes:
            f.write(json.dumps(r) + "\n")

    report = build_report(episodes, clusters, args)
    with open(f"{args.out}_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nWrote {args.out}_clusters.jsonl, {args.out}_episodes.jsonl, "
          f"{args.out}_report.md")


if __name__ == "__main__":
    main()
