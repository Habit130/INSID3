"""WS1: spatial-language diagnostic for TFRIS module 2 (issue #35).

Phase-3 module 2 ("where" decomposition) premise check, built diagnostic-first
like the module-1 diagnostics (`tools/diagnose_category_readout.py` and
`tools/diagnose_category_ceiling.py`, merged): tuning package only, sampled with
the project-wide seed 42, markdown report + per-episode JSONL. It does NOT change
any inference code — it only reuses the frozen model's text/feature/seed paths and
scores deterministic spatial mechanisms offline.

Four measurements (issue #35 acceptance criteria):

  1. Lexicon coverage x target sizing. Every sampled expression is parsed by the
     rule lexicon into four relation families — absolute frame (left/right/top/
     bottom/middle/corner), ordinals ("2nd from right"), relative-to-object
     ("left of the dog"), depth/proximity (closest/front/behind/next to). ALL
     families are parsed for statistics, even though the v1 mechanism acts only on
     absolute + ordinal. Per family x per dataset: frequency and v2 failure rate.
     Plus: the share of wrong-target episodes carrying a spatial term (the module's
     redefined target — wrong-class and wrong-instance alike) and the share with NO
     spatial term (residual headroom for a future signal module). The refcoco+
     near-no-op prior (its protocol bans location words) is checked numerically.
  2. Reachability slice. On parsed-spatial episodes, the oracle-seed (cluster with
     max IoU vs GT) mIoU delta, overall and per dataset, plus its conversion to an
     overall-mIoU contribution — the ceiling for ANY seed-selection intervention on
     the spatial subset.
  3. Mechanism variants, realistic, measured directly (parsing + geometry are
     deterministic, so these ARE the final mechanism's numbers):
       - V-soft (primary): seed = argmax over candidate-covered clusters of
         `cross_sim_norm * spatial_prior`, where the prior scores cluster centroids
         against the parsed relation (e.g. "left" -> 1 - x_norm). When no relation
         in the mechanism scope fires, the episode falls back to the deployed v2
         path bitwise (delta 0) — the no-op guard is structural.
       - V-hard (control): pure geometric re-rank among candidate-covered clusters
         when a relation fires (semantics discarded).
     Mechanism scope = absolute frame + ordinals only. Each reports overall delta,
     spatial-subset delta, correct-seed guard delta, and a per-relation-family
     breakdown.
  4. Appendix — intermediate-layer falsification. For spatial-word Text Prototypes
     (left/right/top/bottom), the correlation of their patch-similarity maps
     against the matching patch coordinate, at several backbone layers (6/12/18/24)
     and the aligned output layer. Pre-registered expectation: ~0 at intermediate
     layers (no trained alignment off the output layer), weak at best at the
     aligned layer. Does not block the main line.

Stopping line (fixed at planning, the HITL gate #36 only reads numbers):
spatial-subset oracle-seed ceiling converted to overall < +0.03 -> route dead;
V-soft realistic >= +0.03 overall with no correct-seed regression -> GO to WS2;
in between -> gray zone, human verdict.

RED LINE: GT masks are used ONLY for scoring (mIoU, seed precision) and the
clearly-labelled oracle-seed. They never enter the v2 path nor any mechanism
variant. Spatial parsing is rule-lexicon only — no embedding discrimination. No
`models/`, `datasets/`, or `inference_referring.py` behaviour changed.

Usage (run through the insid3 env):
  conda run -n insid3 python tools/diagnose_spatial_readout.py \
      --data-root ../refcoco_tuning_package_seed42 --sample 100 --seed 42 \
      --out output/ws1_spatial
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_tfris
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import load_image

# Reuse the module-1 building blocks verbatim (same splits, IoU mirror, helpers).
from tools.diagnose_category_readout import (
    _PACKAGE_SPLITS,
    downsample_gt,
    finalize_iou,
    load_mask,
    md_table,
    mean,
    model_hp,
    read_episodes,
)
from tools.diagnose_category_ceiling import _aggregate_from_seed, _seed_prec

# Wrong-target definition (phase-2 / module-1): seed precision below this floor.
_WRONG_TARGET_FLOOR = 0.3
# HITL stopping-line floor (#36): overall mIoU gain.
_OVERALL_FLOOR = 0.03

# ──────────────────────────── rule lexicon ────────────────────────────
# Absolute-frame direction terms -> (axis, peak) where peak names the coordinate
# extreme the prior favours. left peaks at low x, right at high x, top at low y
# (row 0 is the top of the grid), bottom at high y.
_ABS_DIR = {
    "left": ("x", "low"), "right": ("x", "high"),
    "top": ("y", "low"), "upper": ("y", "low"), "uppermost": ("y", "low"),
    "bottom": ("y", "high"), "lower": ("y", "high"), "lowermost": ("y", "high"),
}
_CENTER_WORDS = {"middle", "center", "centre", "central"}
_CORNER_WORDS = {"corner"}

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5,
    "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10,
}
# Relative-to-object cue words (a relation that references ANOTHER object). The
# "<dir> of" pattern is detected separately so "left of the dog" is relative, not
# absolute.
_RELATIVE_WORDS = {"beside", "above", "below", "under", "beneath", "behind", "atop"}
# Depth / proximity terms.
_DEPTH_WORDS = {
    "closest", "nearest", "close", "closer", "near", "nearer",
    "far", "farther", "farthest", "further", "furthest",
    "front", "frontmost", "back", "backmost", "behind", "foreground",
}
# Direction words that, when immediately followed by "of", form a relative-to-
# object relation rather than an absolute-frame one.
_DIR_OF_WORDS = {"left", "right", "top", "bottom", "front", "back", "side", "middle"}

_FAMILIES = ["absolute", "ordinal", "relative", "depth"]


def _tokenize(expression: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", expression.lower())


def parse_spatial(expression: str) -> dict:
    """Rule-lexicon parse of a Referring Expression. Pure function, deterministic.

    Returns a dict with per-family booleans plus a `mechanism` spec used by the
    geometric prior (absolute + ordinals only — the v1 mechanism scope). Returns
    `mechanism=None` when nothing in scope yields a usable prior.
    """
    toks = _tokenize(expression)
    tokset = set(toks)
    n = len(toks)

    # "<dir> of" -> relative-to-object; mark those positions so they do not also
    # count as absolute-frame terms.
    rel_of_pos = set()
    for i, t in enumerate(toks):
        if t in _DIR_OF_WORDS and i + 1 < n and toks[i + 1] == "of":
            rel_of_pos.add(i)

    # ── absolute frame (excluding "<dir> of" occurrences) ──
    abs_components = []          # list of (axis, peak) and ("center", None)
    has_corner = False
    for i, t in enumerate(toks):
        if i in rel_of_pos:
            continue
        if t in _ABS_DIR:
            abs_components.append(_ABS_DIR[t])
        elif t in _CENTER_WORDS:
            abs_components.append(("center", None))
        elif t in _CORNER_WORDS:
            has_corner = True
    fam_absolute = bool(abs_components) or has_corner

    # ── ordinals ──
    ord_n = None
    for t in toks:
        if t in _ORDINAL_WORDS:
            ord_n = _ORDINAL_WORDS[t]
            break
    if ord_n is None and "from" in tokset:                       # "2 from the right"
        for t in toks:
            if t.isdigit() and 1 <= int(t) <= 10:
                ord_n = int(t)
                break
    fam_ordinal = ord_n is not None

    # ── relative-to-object ──
    fam_relative = bool(rel_of_pos) or bool(tokset & _RELATIVE_WORDS)
    if "next" in tokset and "to" in tokset:
        fam_relative = True
    if "in" in tokset and "front" in tokset and "of" in tokset:
        fam_relative = True

    # ── depth / proximity ──
    fam_depth = bool(tokset & _DEPTH_WORDS)

    # ── mechanism spec (absolute + ordinal only) ──
    mechanism = None
    if fam_ordinal:
        # counting axis/direction: prefer "from <dir>", else any absolute dir token
        axis = start = None
        if "from" in toks:
            fi = toks.index("from")
            d = toks[fi + 1] if fi + 1 < n else None
            axis, start = _ordinal_axis(d)
        if axis is None:
            for t in toks:                                       # fallback: any dir word
                axis, start = _ordinal_axis(t)
                if axis is not None:
                    break
        if axis is not None:
            mechanism = {"kind": "ordinal", "n": ord_n, "axis": axis, "start": start}
    if mechanism is None and abs_components:
        mechanism = {"kind": "absolute", "components": abs_components}

    return {
        "fam_absolute": fam_absolute, "fam_ordinal": fam_ordinal,
        "fam_relative": fam_relative, "fam_depth": fam_depth,
        "has_corner": has_corner,
        "any_spatial": fam_absolute or fam_ordinal or fam_relative or fam_depth,
        "mech_scope_term": fam_absolute or fam_ordinal,
        "mechanism": mechanism,
    }


def _ordinal_axis(direction):
    """Map a counting-origin direction word to (axis, start) for ordinal ranking.

    start='high' means rank 0 is the largest coordinate along the axis (e.g. "from
    right" -> sort x descending), 'low' means smallest first.
    """
    if direction in ("right",):
        return "x", "high"
    if direction in ("left",):
        return "x", "low"
    if direction in ("top", "upper", "uppermost"):
        return "y", "low"
    if direction in ("bottom", "lower", "lowermost"):
        return "y", "high"
    return None, None


# ──────────────────────────── geometric prior ────────────────────────────

def _cluster_centroids(labels: torch.Tensor, K: int, h: int, w: int):
    """Per-cluster normalized (x, y) centroid in [0, 1]. Returns (cx, cy) tensors."""
    rows = torch.arange(h, device=labels.device).view(h, 1).expand(h, w).reshape(-1).float()
    cols = torch.arange(w, device=labels.device).view(1, w).expand(h, w).reshape(-1).float()
    flat = labels.reshape(-1)
    cx = torch.zeros(K, device=labels.device)
    cy = torch.zeros(K, device=labels.device)
    for k in range(K):
        idx = flat == k
        if idx.any():
            cx[k] = cols[idx].mean() / max(w - 1, 1)
            cy[k] = rows[idx].mean() / max(h - 1, 1)
    return cx, cy


def _absolute_prior(cx: torch.Tensor, cy: torch.Tensor, components) -> torch.Tensor:
    """Soft coordinate prior in [0, 1] over all K clusters for absolute-frame terms.

    Multiple terms (e.g. "bottom left") multiply — the same soft-product philosophy
    as the merge rule (ADR 0003).
    """
    prior = torch.ones_like(cx)
    for axis, peak in components:
        if axis == "x":
            prior = prior * (cx if peak == "high" else (1.0 - cx))
        elif axis == "y":
            prior = prior * (cy if peak == "high" else (1.0 - cy))
        else:                                                    # center
            centered = (1.0 - 2.0 * (cx - 0.5).abs()) * (1.0 - 2.0 * (cy - 0.5).abs())
            prior = prior * centered.clamp(min=0.0)
    return prior


def _ordinal_prior(cx, cy, matched_ids, spec) -> torch.Tensor:
    """Soft ordinal prior over ALL K clusters; peaks at the Nth matched cluster
    along the counting axis. Non-matched clusters get 0 (ordinals rank instances,
    i.e. the candidate-covered clusters)."""
    K = cx.shape[0]
    prior = torch.zeros(K, device=cx.device)
    coord = cx if spec["axis"] == "x" else cy
    ids = matched_ids.tolist()
    if not ids:
        return prior
    vals = [(coord[i].item(), i) for i in ids]
    vals.sort(key=lambda t: t[0], reverse=(spec["start"] == "high"))
    target = spec["n"] - 1
    for rank, (_, i) in enumerate(vals):
        prior[i] = float(torch.exp(torch.tensor(-abs(rank - target), dtype=torch.float32)))
    return prior


def _spatial_prior(cx, cy, matched_ids, mechanism) -> torch.Tensor:
    if mechanism["kind"] == "ordinal":
        return _ordinal_prior(cx, cy, matched_ids, mechanism)
    return _absolute_prior(cx, cy, mechanism["components"])


# ──────────────────────────── per-episode ────────────────────────────

def diagnose_episode(model, ep, data_root, ep_id, dataset, split, device) -> dict:
    """Run one episode; return its WS1 spatial diagnostic record (no GT in any
    decision path)."""
    parse = parse_spatial(ep["expression"])
    base = {
        "ep_id": ep_id, "dataset": dataset, "split": split,
        "sent_id": ep["sent_id"], "expression": ep["expression"],
        "fam_absolute": parse["fam_absolute"], "fam_ordinal": parse["fam_ordinal"],
        "fam_relative": parse["fam_relative"], "fam_depth": parse["fam_depth"],
        "any_spatial": parse["any_spatial"], "mech_scope_term": parse["mech_scope_term"],
        "mech_kind": parse["mechanism"]["kind"] if parse["mechanism"] else None,
    }

    gt_full = load_mask(os.path.join(data_root, ep["mask_path"])).to(device)
    img, orig_size = load_image(
        os.path.join(data_root, ep["image_path"]), model._transform, device)

    text_proto = model.dinotxt.encode_expression(ep["expression"])
    feat_native, feat_aligned = model.dinotxt.encode_image_features(img)
    _, h, w = feat_native.shape
    gt = downsample_gt(gt_full, h, w)

    sim_expr = torch.einsum("c,chw->hw", text_proto, feat_aligned)
    candidate_mask = sim_expr > torch.quantile(sim_expr, model.cand_quantile)

    if candidate_mask.sum() == 0:
        v2_iou = finalize_iou(candidate_mask, img, orig_size, gt_full)
        base.update({"has_seed": False, "v2_iou": round(v2_iou, 3),
                     "mech_fires": False})
        return base

    feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
    labels = agglomerative_clustering(feat_native_flat, model.tau).reshape(h, w)
    K = int(labels.max().item()) + 1
    feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
    protos_aligned = compute_cluster_prototypes(feat_aligned_flat, labels.view(-1), K)

    # ── v2 seed + mask (deployed path; no GT) ──
    matched_ids = labels[candidate_mask & (labels >= 0)].unique()
    seed_id = int(matched_ids[int(torch.argmax(protos_aligned[matched_ids] @ text_proto))].item())
    v2_seed_prec = _seed_prec(labels == seed_id, gt)
    v2_feat = model._seed_and_aggregate(
        candidate_mask, labels, protos_aligned, K, text_proto, feat_native_flat, sim_expr, h, w)
    v2_iou = finalize_iou(v2_feat, img, orig_size, gt_full)

    base.update({
        "has_seed": True, "K": K, "n_matched": int(matched_ids.numel()),
        "seed_prec": round(v2_seed_prec, 3),
        "correct_seed": v2_seed_prec >= _WRONG_TARGET_FLOOR,
        "wrong_target": v2_seed_prec < _WRONG_TARGET_FLOOR,
        "v2_iou": round(v2_iou, 3),
    })

    # ── oracle seed (max-IoU cluster vs GT) — reachability ceiling, GT-scored ──
    ious = torch.empty(K, device=device)
    for k in range(K):
        ck = labels == k
        ious[k] = (ck & gt).sum().item() / max((ck | gt).sum().item(), 1)
    oracle_seed = int(torch.argmax(ious).item())
    oracle_feat = _aggregate_from_seed(
        oracle_seed, labels, K, feat_native_flat, sim_expr, model.merge_threshold, h, w)
    base["oracle_iou"] = round(finalize_iou(oracle_feat, img, orig_size, gt_full), 3)
    base["oracle_seed_prec"] = round(_seed_prec(labels == oracle_seed, gt), 3)

    # ── mechanism variants (deterministic; no GT) ──
    mechanism = parse["mechanism"]
    mech_fires = mechanism is not None
    base["mech_fires"] = mech_fires
    if not mech_fires:
        # Structural no-op: bitwise v2. Record v2 values for both variants.
        base["vsoft_iou"] = round(v2_iou, 3)
        base["vhard_iou"] = round(v2_iou, 3)
        base["vsoft_seed_prec"] = round(v2_seed_prec, 3)
        base["vhard_seed_prec"] = round(v2_seed_prec, 3)
        return base

    cx, cy = _cluster_centroids(labels, K, h, w)
    prior = _spatial_prior(cx, cy, matched_ids, mechanism)        # (K,)

    # cross_sim_norm: per-episode min-max of per-cluster mean text-vs-patch sim
    cross_sim = torch.empty(K, device=device, dtype=sim_expr.dtype)
    for k in range(K):
        idx = labels == k
        cross_sim[k] = sim_expr[idx].mean() if idx.any() else 0.0
    cross_sim_norm = (cross_sim - cross_sim.min()) / (cross_sim.max() - cross_sim.min() + 1e-8)

    def _run(score_vec: torch.Tensor):
        """Pick seed = argmax over matched clusters of score_vec; aggregate as v2."""
        m = matched_ids
        sub = score_vec[m]
        if sub.numel() == 0:                                      # never on a fire episode
            return v2_iou, v2_seed_prec, False
        v_seed = int(m[int(torch.argmax(sub))].item())
        feat = _aggregate_from_seed(
            v_seed, labels, K, feat_native_flat, sim_expr, model.merge_threshold, h, w)
        return (finalize_iou(feat, img, orig_size, gt_full),
                _seed_prec(labels == v_seed, gt), True)

    # V-soft: cross_sim_norm * prior. V-hard: pure geometry (prior only).
    vsoft_iou, vsoft_sp, vsoft_ap = _run(cross_sim_norm * prior)
    vhard_iou, vhard_sp, vhard_ap = _run(prior)
    base.update({
        "vsoft_iou": round(vsoft_iou, 3), "vsoft_seed_prec": round(vsoft_sp, 3),
        "vsoft_applicable": vsoft_ap,
        "vhard_iou": round(vhard_iou, 3), "vhard_seed_prec": round(vhard_sp, 3),
        "vhard_applicable": vhard_ap,
    })
    return base


# ──────────────────────────── appendix ────────────────────────────

_APPENDIX_WORDS = {
    "left": ("x", -1), "right": ("x", 1),    # sign: +1 means sim should rise with coord
    "top": ("y", -1), "bottom": ("y", 1),
}
_APPENDIX_LAYERS = [5, 11, 17, 23]            # 0-indexed -> reported as 6/12/18/24


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return (a @ b).item() / denom if denom > 1e-12 else 0.0


def run_appendix(model, data_root, n_images, seed, device) -> list[dict]:
    """Per-layer correlation of spatial-word similarity maps vs patch coordinate.

    For each sampled image and word, sim = word Text Prototype . patch features at
    a layer; correlation is taken against the word's matching coordinate (signed so
    that a positive value means the map aligns with the word's spatial meaning).
    Intermediate backbone layers share the 1024-d width but live in an untrained
    space relative to the text embedding -> the falsification: expect ~0.
    """
    backbone = model.dinotxt.model.visual_model.backbone
    protos = {wd: model.dinotxt.encode_expression(wd) for wd in _APPENDIX_WORDS}

    # collect unique image paths across splits, sample deterministically
    paths = []
    seen = set()
    for dataset, split in _PACKAGE_SPLITS:
        for ep in read_episodes(data_root, dataset, split):
            p = ep["image_path"]
            if p not in seen:
                seen.add(p)
                paths.append(p)
    rng = random.Random(seed)
    if n_images < len(paths):
        paths = rng.sample(paths, n_images)

    # accumulate signed correlations per (word, layer-label)
    layer_labels = [f"L{ln + 1}" for ln in _APPENDIX_LAYERS] + ["aligned"]
    acc = {wd: {lab: [] for lab in layer_labels} for wd in _APPENDIX_WORDS}

    for p in paths:
        img, _ = load_image(os.path.join(data_root, p), model._transform, device)
        h = img.shape[-2] // 16
        w = img.shape[-1] // 16
        cols = (torch.arange(w, device=device).view(1, w).expand(h, w).reshape(-1).float()
                / max(w - 1, 1))
        rows = (torch.arange(h, device=device).view(h, 1).expand(h, w).reshape(-1).float()
                / max(h - 1, 1))
        coord = {"x": cols, "y": rows}

        inter = backbone.get_intermediate_layers(
            img.to(device), n=_APPENDIX_LAYERS, reshape=False, norm=True)
        _, aligned = model.dinotxt.encode_image_features(img)
        aligned_flat = torch.nn.functional.normalize(
            aligned.reshape(aligned.shape[0], -1).permute(1, 0), p=2, dim=1)

        for wd, (axis, sign) in _APPENDIX_WORDS.items():
            tp = protos[wd]
            tgt = coord[axis]
            for ln, feats in zip(_APPENDIX_LAYERS, inter):
                toks = torch.nn.functional.normalize(feats[0], p=2, dim=1)   # (P, 1024)
                sim = toks @ tp
                acc[wd][f"L{ln + 1}"].append(sign * _pearson(sim, tgt))
            sim_al = aligned_flat @ tp
            acc[wd]["aligned"].append(sign * _pearson(sim_al, tgt))

    rows_out = []
    for wd in _APPENDIX_WORDS:
        row = {"word": wd}
        for lab in layer_labels:
            vals = acc[wd][lab]
            row[lab] = round(mean(vals), 4) if vals else float("nan")
        rows_out.append(row)
    return rows_out


# ──────────────────────────── reporting ────────────────────────────

def _delta_overall(sub_deltas: list[float], n_total: int) -> float:
    """A subset's mean delta converted to a whole-sample mIoU contribution."""
    return sum(sub_deltas) / max(n_total, 1)


def build_report(episodes, appendix_rows, args) -> str:
    splits = [f"{d}/{s}" for d, s in _PACKAGE_SPLITS]
    seeded = [e for e in episodes if e.get("has_seed")]
    n_all = len(episodes)
    L: list[str] = []

    L.append("## WS1 — Spatial-language diagnostic (issue #35)\n")
    L.append(
        f"**Provenance.** Tuning package `{os.path.basename(args.data_root.rstrip('/'))}`, "
        f"splits {', '.join(splits)}; sample={args.sample} expressions/split "
        f"(seed {args.seed}); device {args.device}; frozen hyperparams "
        f"tau={model_hp(args,'tau')} cand_quantile={model_hp(args,'cand_quantile')} "
        f"merge_threshold={model_hp(args,'merge_threshold')} image_size=1024. "
        f"Episodes={n_all} (with a candidate seed={len(seeded)}). "
        f"Wrong-target = seed precision < {_WRONG_TARGET_FLOOR}.\n")
    L.append(
        "**Red line.** GT masks are used ONLY for scoring (mIoU, seed precision) and "
        "the labelled oracle-seed. They never enter the v2 path nor any mechanism "
        "variant. Spatial parsing is rule-lexicon only (no embedding discrimination). "
        "No inference code changed.\n")
    L.append(
        "**Lexicon.** absolute frame = {left, right, top/upper, bottom/lower, "
        "middle/center, corner}; ordinals = {first..tenth, 1st..10th, plus a bare "
        "digit with \"from\"}; relative-to-object = \"<dir> of\", {beside, above, "
        "below, under, beneath, behind, atop}, \"next to\", \"in front of\"; "
        "depth/proximity = {closest/nearest, close(r), near(er), far(ther/thest), "
        "front(most), back, behind, foreground}. Precedence: \"<dir> of\" is "
        "relative, not absolute. Mechanism scope (V-soft/V-hard) = absolute + "
        "ordinals only; relative & depth are parsed for statistics and are "
        "mechanism no-ops in v1. Families overlap (e.g. \"behind\" is both relative "
        "and depth) — coverage counts are per family, not a partition.\n")

    L.append(_measurement1(episodes, seeded))
    L.append(_measurement2(seeded, n_all))
    L.append(_measurement3(seeded, n_all))
    L.append(_appendix_section(appendix_rows, args))
    L.append(_stopping_section(seeded, n_all))
    return "\n".join(L)


def _measurement1(episodes, seeded) -> str:
    L = ["### 1. Lexicon coverage x target sizing\n"]
    L.append("Frequency = share of a split's episodes carrying >=1 term of the "
             "family. v2 failure rate = wrong-target share among that family's "
             "episodes (seeded only; no-seed episodes excluded from the rate). "
             "Families overlap.\n")
    rows = []
    for d, s in _PACKAGE_SPLITS + [("**all**", "")]:
        if d == "**all**":
            allep = episodes
            label = "**all**"
        else:
            allep = [e for e in episodes if e["dataset"] == d and e["split"] == s]
            label = f"{d}/{s}"
        row = [label, len(allep)]
        for fam in _FAMILIES:
            key = f"fam_{fam}"
            fam_eps = [e for e in allep if e.get(key)]
            freq = len(fam_eps) / max(len(allep), 1)
            fam_seeded = [e for e in fam_eps if e.get("has_seed")]
            fail = mean([e["wrong_target"] for e in fam_seeded]) if fam_seeded else float("nan")
            row.append(f"{freq:.0%} / {fail:.0%}" if fam_seeded else f"{freq:.0%} / —")
        rows.append(row)
    L.append(md_table(["split", "n"] + [f"{f} (freq/v2-fail)" for f in _FAMILIES], rows) + "\n")

    # wrong-target composition: spatial-term share vs residual no-spatial share
    L.append("**Wrong-target composition (the module's target sizing).** Among "
             "wrong-target episodes: share carrying ANY spatial term, share with a "
             "mechanism-scope term (absolute/ordinal — the v1 mechanism's reachable "
             "set), and share with NO spatial term (residual headroom for a future "
             "signal-enhancement module).\n")
    rows = []
    for d, s in _PACKAGE_SPLITS + [("**all**", "")]:
        if d == "**all**":
            wt = [e for e in seeded if e.get("wrong_target")]
            label = "**all**"
        else:
            wt = [e for e in seeded if e.get("wrong_target")
                  and e["dataset"] == d and e["split"] == s]
            label = f"{d}/{s}"
        if not wt:
            rows.append([label, 0, "—", "—", "—"]); continue
        any_sp = mean([e["any_spatial"] for e in wt])
        mech = mean([e["mech_scope_term"] for e in wt])
        none_sp = mean([not e["any_spatial"] for e in wt])
        rows.append([label, len(wt), f"{any_sp:.0%}", f"{mech:.0%}", f"{none_sp:.0%}"])
    L.append(md_table(["split", "n wrong-target", "any spatial term",
                       "mech-scope term (abs/ord)", "NO spatial term (residual)"],
                      rows) + "\n")

    # refcoco+ no-op check
    rc_plus = [e for e in episodes if e["dataset"] == "refcoco+"]
    abs_ord = mean([e["mech_scope_term"] for e in rc_plus]) if rc_plus else float("nan")
    any_sp = mean([e["any_spatial"] for e in rc_plus]) if rc_plus else float("nan")
    L.append(f"**refcoco+ no-op check.** mechanism-scope (absolute/ordinal) term "
             f"frequency = {abs_ord:.0%}; any-spatial-term frequency = {any_sp:.0%} "
             f"(n={len(rc_plus)}). Prior: near-zero, its annotation protocol bans "
             f"location words.\n")
    return "\n".join(L)


def _subset(seeded, pred):
    return [e for e in seeded if pred(e)]


def _measurement2(seeded, n_all) -> str:
    L = ["### 2. Reachability slice (oracle-seed ceiling on the spatial subset)\n"]
    L.append("Oracle-seed = the cluster with max IoU vs GT (GT-scored, labelled "
             "oracle — never in any decision path); aggregation is the v2 rule from "
             "that seed. Δ is per-episode oracle mIoU − v2 mIoU. **Δ→overall** "
             f"converts the subset Δ to a whole-sample contribution (×n_subset/"
             f"{n_all}). This is the ceiling for ANY seed-selection intervention on "
             "the subset; the stopping line reads the **all** mechanism-scope "
             "Δ→overall against +%.2f.\n" % _OVERALL_FLOOR)

    def block(name, pred):
        rows = []
        for d, s in _PACKAGE_SPLITS + [("**all**", "")]:
            if d == "**all**":
                sub = _subset(seeded, pred)
                label = "**all**"
            else:
                sub = _subset(seeded, lambda e, d=d, s=s: pred(e)
                              and e["dataset"] == d and e["split"] == s)
                label = f"{d}/{s}"
            if not sub:
                rows.append([label, 0, "—", "—", "—", "—"]); continue
            v2 = mean([e["v2_iou"] for e in sub])
            orc = mean([e["oracle_iou"] for e in sub])
            deltas = [e["oracle_iou"] - e["v2_iou"] for e in sub]
            rows.append([label, len(sub), round(v2, 3), round(orc, 3),
                         f"{mean(deltas):+.3f}", f"{_delta_overall(deltas, n_all):+.3f}"])
        return (f"\n_Spatial subset = {name}:_\n\n"
                + md_table(["split", "n", "v2 mIoU", "oracle mIoU", "Δ subset",
                            "Δ→overall"], rows) + "\n")

    L.append(block("mechanism-scope (absolute/ordinal)", lambda e: e.get("mech_fires")))
    L.append(block("any spatial term", lambda e: e.get("any_spatial")))
    return "\n".join(L)


def _measurement3(seeded, n_all) -> str:
    L = ["### 3. Mechanism variants — realistic deltas (V-soft primary, V-hard control)\n"]
    L.append("Both variants fall back to v2 bitwise when no mechanism-scope relation "
             "fires (delta 0 — the structural no-op guard). **overall Δ** = "
             f"contribution to the whole-sample mIoU (Σ fire-episode Δ / {n_all}). "
             "**spatial-subset Δ** = mean over mechanism-fire episodes. **correct-"
             "seed guard Δ** = mean Δ over episodes whose v2 seed is already correct "
             "(seed_prec ≥ %.1f; the no-regression guard — fire episodes can move a "
             "correct seed). Per-family rows restrict to that family's fire "
             "episodes.\n" % _WRONG_TARGET_FLOOR)

    fire = [e for e in seeded if e.get("mech_fires")]
    correct = [e for e in seeded if e.get("correct_seed")]

    rows = []
    for var in ("vsoft", "vhard"):
        ik = f"{var}_iou"
        d_overall = _delta_overall([e[ik] - e["v2_iou"] for e in fire], n_all)
        d_sub = mean([e[ik] - e["v2_iou"] for e in fire]) if fire else float("nan")
        d_guard = mean([e[ik] - e["v2_iou"] for e in correct]) if correct else float("nan")
        d_abs = [e[ik] - e["v2_iou"] for e in fire if e.get("mech_kind") == "absolute"]
        d_ord = [e[ik] - e["v2_iou"] for e in fire if e.get("mech_kind") == "ordinal"]
        rows.append([
            "V-soft" if var == "vsoft" else "V-hard", len(fire),
            f"{d_overall:+.3f}", f"{d_sub:+.3f}", f"{d_guard:+.3f}",
            f"{mean(d_abs):+.3f} (n={len(d_abs)})" if d_abs else "—",
            f"{mean(d_ord):+.3f} (n={len(d_ord)})" if d_ord else "—",
        ])
    L.append(md_table(["variant", "n fire", "overall Δ", "spatial-subset Δ",
                       "correct-seed guard Δ", "Δ absolute", "Δ ordinal"], rows) + "\n")

    # per-dataset overall-Δ
    L.append("**Per-dataset overall Δ** (whole-split contribution):\n")
    rows = []
    for d, s in _PACKAGE_SPLITS:
        sub = [e for e in seeded if e["dataset"] == d and e["split"] == s]
        nfire = sum(1 for e in sub if e.get("mech_fires"))
        row = [f"{d}/{s}", len(sub), nfire]
        for var in ("vsoft", "vhard"):
            ik = f"{var}_iou"
            f_eps = [e for e in sub if e.get("mech_fires")]
            row.append(f"{_delta_overall([e[ik]-e['v2_iou'] for e in f_eps], len(sub)):+.3f}"
                       if f_eps else "—")
        rows.append(row)
    L.append(md_table(["split", "n", "n fire", "V-soft overall Δ", "V-hard overall Δ"],
                      rows) + "\n")

    # seed-precision movement on fire episodes
    if fire:
        L.append("**Seed-precision movement on fire episodes** (share with a correct "
                 "seed, seed_prec ≥ %.1f):\n" % _WRONG_TARGET_FLOOR)
        v2_c = mean([e["correct_seed"] for e in fire])
        vs_c = mean([e["vsoft_seed_prec"] >= _WRONG_TARGET_FLOOR for e in fire])
        vh_c = mean([e["vhard_seed_prec"] >= _WRONG_TARGET_FLOOR for e in fire])
        L.append(md_table(["v2", "V-soft", "V-hard"],
                          [[f"{v2_c:.0%}", f"{vs_c:.0%}", f"{vh_c:.0%}"]]) + "\n")
    return "\n".join(L)


def _appendix_section(appendix_rows, args) -> str:
    L = ["### Appendix — intermediate-layer falsification\n"]
    L.append("Signed correlation of each spatial-word Text Prototype's patch-"
             "similarity map against the matching patch coordinate (positive = map "
             "aligns with the word's spatial meaning), averaged over "
             f"{args.appendix_images} sampled images (seed {args.seed}). Backbone "
             "layers 6/12/18/24 share the 1024-d width but live in an untrained "
             "space relative to the text embedding; the aligned layer is the head "
             "output (the deployed matching space). Pre-registered expectation: ~0 "
             "at intermediate layers, weak at the aligned layer.\n")
    if not appendix_rows:
        L.append("_(skipped: run without --no-appendix to populate)_\n")
        return "\n".join(L)
    cols = [k for k in appendix_rows[0] if k != "word"]
    rows = [[r["word"]] + [r[c] for c in cols] for r in appendix_rows]
    L.append(md_table(["word"] + cols, rows) + "\n")
    L.append("Read the numbers directly. Any non-zero correlation at an intermediate "
             "backbone layer reflects that layer's own positional-embedding structure "
             "(present with no text alignment), not text-driven spatial grounding — it "
             "is inconsistent across layers and words and so is not a usable, "
             "trainable-free matching signal. The aligned output layer is the deployed "
             "matching space and the relevant row for the route.\n")
    return "\n".join(L)


def _stopping_section(seeded, n_all) -> str:
    fire = [e for e in seeded if e.get("mech_fires")]
    correct = [e for e in seeded if e.get("correct_seed")]
    # oracle ceiling on mechanism-scope subset, converted to overall
    ceil_overall = _delta_overall([e["oracle_iou"] - e["v2_iou"] for e in fire], n_all)
    vsoft_overall = _delta_overall([e["vsoft_iou"] - e["v2_iou"] for e in fire], n_all)
    vsoft_guard = mean([e["vsoft_iou"] - e["v2_iou"] for e in correct]) if correct else 0.0
    vhard_overall = _delta_overall([e["vhard_iou"] - e["v2_iou"] for e in fire], n_all)

    L = ["### Mapping onto the stopping line (#36)\n"]
    L.append(f"Stopping line: spatial-subset oracle-seed ceiling → overall < "
             f"+{_OVERALL_FLOOR:.2f} → route dead; V-soft realistic ≥ +{_OVERALL_FLOOR:.2f} "
             f"overall AND no correct-seed regression → GO to WS2; in between → gray "
             f"zone, human verdict on #36.\n")
    L.append(md_table(
        ["metric", "value", "vs +%.2f" % _OVERALL_FLOOR],
        [["oracle-seed ceiling → overall (mech-scope subset)", f"{ceil_overall:+.3f}",
          _zone(ceil_overall)],
         ["V-soft realistic overall Δ", f"{vsoft_overall:+.3f}", _zone(vsoft_overall)],
         ["V-soft correct-seed guard Δ", f"{vsoft_guard:+.3f}",
          "no regression" if vsoft_guard >= -1e-9 else "REGRESSION"],
         ["V-hard realistic overall Δ", f"{vhard_overall:+.3f}", _zone(vhard_overall)]]) + "\n")
    L.append("Numbers only — the GO/NO-GO call is the human's on #36.\n")
    return "\n".join(L)


def _zone(v: float) -> str:
    return "≥ floor" if v >= _OVERALL_FLOOR else "< floor"


# ──────────────────────────── main ────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../refcoco_tuning_package_seed42")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/ws1_spatial")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--cand-quantile", type=float, default=None)
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--appendix-images", type=int, default=60)
    parser.add_argument("--no-appendix", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.report_only:
        with open(f"{args.out}_episodes.jsonl", encoding="utf-8") as f:
            episodes = [json.loads(line) for line in f]
        ap_path = f"{args.out}_appendix.json"
        appendix_rows = json.load(open(ap_path, encoding="utf-8")) if os.path.isfile(ap_path) else []
        report = build_report(episodes, appendix_rows, args)
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

    rng = random.Random(args.seed)
    episodes = []
    for dataset, split in _PACKAGE_SPLITS:
        recs = read_episodes(args.data_root, dataset, split)
        if args.sample and args.sample < len(recs):
            recs = rng.sample(recs, args.sample)
        for j, ep in enumerate(recs):
            ep_id = f"{dataset}/{split}/{ep['sent_id']}"
            with torch.no_grad():
                base = diagnose_episode(model, ep, args.data_root, ep_id,
                                        dataset, split, args.device)
            episodes.append(base)
            print(f"[{dataset}/{split} {j+1}/{len(recs)}] "
                  f"v2={base.get('v2_iou', 0):.3f} "
                  f"vsoft={base.get('vsoft_iou', float('nan')):.3f} "
                  f"vhard={base.get('vhard_iou', float('nan')):.3f} "
                  f"fire={base.get('mech_fires')} kind={base.get('mech_kind')} "
                  f"expr={ep['expression'][:30]!r}")

    appendix_rows = []
    if not args.no_appendix:
        with torch.no_grad():
            appendix_rows = run_appendix(
                model, args.data_root, args.appendix_images, args.seed, args.device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}_episodes.jsonl", "w", encoding="utf-8") as f:
        for r in episodes:
            f.write(json.dumps(r) + "\n")
    with open(f"{args.out}_appendix.json", "w", encoding="utf-8") as f:
        json.dump(appendix_rows, f)
    report = build_report(episodes, appendix_rows, args)
    with open(f"{args.out}_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\nWrote {args.out}_episodes.jsonl, {args.out}_appendix.json, {args.out}_report.md")


if __name__ == "__main__":
    main()
