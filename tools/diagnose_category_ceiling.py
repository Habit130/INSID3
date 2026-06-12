"""WS1b: category-gate ceiling decomposition + mechanism-variant oracles (issue #32).

WS1 (#26 / PR #31) showed the naive argmax class gate cannot clear the PRD floor
(GT-category oracle ceiling +0.012 < +0.03). The HITL gate (#27) inserted this
diagnostic BEFORE the final verdict, to decompose WHY the ceiling is so low and to
measure whether any continuous category mechanism can work. Same protocol, scale,
and red lines as WS1: tuning package only, sample 100/split, seed 42, frozen
hyperparameters, no inference-code change. GT `category_name` is used ONLY for
scoring and the clearly-labelled GT-category oracles.

Three measurements (issue #32 acceptance criteria):

  1. Cluster-side categorization accuracy (suspect 1: the right clusters are
     invisible to a discrete gate). For GT-majority clusters (>50% of the
     cluster's feature-grid patches fall inside the downsampled GT mask): the
     cluster-level zero-shot readout accuracy vs GT category, the episode-level
     rate of "at least one GT-majority cluster reads out as the GT class", and the
     false-positive rate (non-GT-majority clusters reading out as the target
     class). This says whether the discrete gate failed because the target
     clusters themselves do not read out as the target category.
  2. Within-class selection quality (suspect 2: even inside the right class, the
     full-expression argmax still picks the wrong cluster). Under the GT-class
     restriction, the seed hit rate (seed_prec >= 0.3, the phase-2 wrong-target
     definition) before (v2) vs after (GT-class-restricted seed), on the subset
     where the restriction is applicable.
  3. Mechanism-variant oracles — GT-category versions first. Three
     continuous-similarity variants that bypass discrete cluster categorization,
     scored as mIoU delta over the v2 seed (WS1 reference: v2 = 0.302 on this
     sample). In all three, aggregation cross_sim keeps the full-expression map
     (only candidate localization + seed selection — the "seeding signal" —
     change):
       3a full replacement: the category-prototype similarity map drives
          Candidate Localization and seed selection.
       3b product fusion: per-patch `sim_cat * sim_expr` is the seeding signal
          (parameter-free what-times-which); candidate localization by its
          quantile, seed = matched cluster with the highest mean fused score.
       3c continuous two-stage: category similarity circles candidate patches via
          the existing cand_quantile mechanism; full-expression similarity argmax
          within picks the seed.
     Each variant is run with the GT category (oracle ceiling) and, per the
     stopping line, the realistic expression-readout version is reported and used
     for the verdict only for variants whose GT-oracle clears +0.03. Every variant
     reports the correct-seed subset delta as the guard metric.

Stopping line (fixed at the HITL gate #27): any variant GT-oracle >= +0.05 -> GO
to WS2 with that mechanism; all variants GT-oracle < +0.03 -> category route dead,
phase 3 pivots to module 2; in between -> gray zone, human verdict on #27.

RED LINE: GT `category_name` is a diagnosis-only input — scoring + GT-category
oracles only. It never enters the v2 path nor the realistic (expression-readout)
variants. No `models/`, `datasets/`, or `inference_referring.py` behaviour changes.

Usage (run through the insid3 env):
  conda run -n insid3 python tools/diagnose_category_ceiling.py \
      --data-root ../refcoco_tuning_package_seed42 --sample 100 --seed 42 \
      --out output/ws1b_ceiling
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models import build_tfris
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import load_image

# Reuse the WS1 building blocks verbatim (same COCO-80 bank, splits, IoU mirror).
from tools.diagnose_category_readout import (
    COCO_CATEGORIES,
    _CAT_INDEX,
    _PACKAGE_SPLITS,
    downsample_gt,
    finalize_iou,
    load_mask,
    md_table,
    mean,
    model_hp,
    read_episodes,
)

# GT-oracle delta thresholds from the HITL stopping line (#27 / #32).
_GO_FLOOR = 0.05
_DEAD_FLOOR = 0.03

_VARIANTS = ["3a", "3b", "3c"]


def _aggregate_from_seed(
    seed_id: int,
    labels: torch.Tensor,
    K: int,
    feat_native_flat: torch.Tensor,
    agg_sim: torch.Tensor,
    merge_threshold: float,
    h: int,
    w: int,
) -> torch.Tensor:
    """Cluster aggregation for a pre-chosen seed — faithful copy of the deployed
    `TFRIS._seed_and_aggregate` aggregation block (tfris.py), with the seed
    injected rather than picked by a prototype dot product. Lets each variant pick
    its seed by its own signal while keeping aggregation pixel-identical to v2.
    `agg_sim` is the per-patch cross_sim map (always the full-expression map here).
    """
    native_protos = compute_cluster_prototypes(feat_native_flat, labels.view(-1), K)
    intra_sim = torch.einsum("c,kc->k", native_protos[seed_id], native_protos)

    cross_sim = torch.empty(K, device=agg_sim.device, dtype=agg_sim.dtype)
    for k in range(K):
        idx = labels == k
        cross_sim[k] = agg_sim[idx].mean() if idx.any() else 0.0
    cross_sim_norm = (cross_sim - cross_sim.min()) / (cross_sim.max() - cross_sim.min() + 1e-8)

    score = intra_sim * cross_sim_norm
    final = torch.zeros(h, w, dtype=torch.bool, device=labels.device)
    valid = labels >= 0
    final[valid] = score[labels[valid]] > merge_threshold
    final |= labels == seed_id
    return final


def _seed_prec(seed_mask: torch.Tensor, gt: torch.Tensor) -> float:
    return (seed_mask & gt).sum().item() / max(seed_mask.sum().item(), 1)


def diagnose_episode(model, ep, data_root, ep_id, dataset, split, cat_protos, device):
    """Run one episode; return its WS1b diagnostic record (no GT in any prediction)."""
    gt_full = load_mask(os.path.join(data_root, ep["mask_path"])).to(device)
    img, orig_size = load_image(
        os.path.join(data_root, ep["image_path"]), model._transform, device)
    gt_cat = ep["category_name"]
    gt_in_bank = gt_cat in _CAT_INDEX
    gt_idx = _CAT_INDEX.get(gt_cat, -1)

    text_proto = model.dinotxt.encode_expression(ep["expression"])
    feat_native, feat_aligned = model.dinotxt.encode_image_features(img)
    _, h, w = feat_native.shape
    gt = downsample_gt(gt_full, h, w)

    sim_expr = torch.einsum("c,chw->hw", text_proto, feat_aligned)
    candidate_mask = sim_expr > torch.quantile(sim_expr, model.cand_quantile)

    base = {
        "ep_id": ep_id, "dataset": dataset, "split": split,
        "sent_id": ep["sent_id"], "gt_category": gt_cat, "gt_in_bank": gt_in_bank,
    }

    if candidate_mask.sum() == 0:
        v2_iou = finalize_iou(candidate_mask, img, orig_size, gt_full)
        base.update({"has_seed": False, "v2_iou": round(v2_iou, 3)})
        return base

    feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
    labels = agglomerative_clustering(feat_native_flat, model.tau).reshape(h, w)
    K = int(labels.max().item()) + 1
    feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
    protos_aligned = compute_cluster_prototypes(feat_aligned_flat, labels.view(-1), K)
    cluster_cat = (protos_aligned @ cat_protos.t()).argmax(dim=1)   # (K,) COCO index

    # ── v2 seed + mask (deployed path; no GT, no category) ──
    matched_ids = labels[candidate_mask & (labels >= 0)].unique()
    seed_id = int(matched_ids[int(torch.argmax(protos_aligned[matched_ids] @ text_proto))].item())
    v2_seed_prec = _seed_prec(labels == seed_id, gt)
    v2_feat = model._seed_and_aggregate(
        candidate_mask, labels, protos_aligned, K, text_proto, feat_native_flat, sim_expr, h, w)
    v2_iou = finalize_iou(v2_feat, img, orig_size, gt_full)

    base.update({
        "has_seed": True, "K": K,
        "seed_prec": round(v2_seed_prec, 3),
        "correct_seed": v2_seed_prec >= 0.3,
        "wrong_target": v2_seed_prec < 0.3,
        "v2_iou": round(v2_iou, 3),
    })

    # ── Measurement 1: cluster-side categorization accuracy ──
    if gt_in_bank:
        cluster_sizes = torch.stack([(labels == k).sum() for k in range(K)])
        gt_overlap = torch.stack([((labels == k) & gt).sum() for k in range(K)]).float()
        frac = gt_overlap / cluster_sizes.clamp(min=1).float()
        gt_majority = frac > 0.5                                    # (K,) bool
        reads_gt = cluster_cat == gt_idx
        n_gtmaj = int(gt_majority.sum().item())
        n_nongt = K - n_gtmaj
        base.update({
            "m1_n_gt_majority": n_gtmaj,
            "m1_n_gt_majority_read_gt": int((gt_majority & reads_gt).sum().item()),
            "m1_episode_has_gt_read": bool((gt_majority & reads_gt).any().item()),
            "m1_n_nongt": n_nongt,
            "m1_n_nongt_read_gt": int(((~gt_majority) & reads_gt).sum().item()),
        })

    # ── Measurement 2: within-class selection (GT-class-restricted seed) ──
    if gt_in_bank:
        allowed = cluster_cat == gt_idx
        restricted_cand = candidate_mask & allowed[labels]
        if restricted_cand.sum() > 0:
            r_matched = labels[restricted_cand & (labels >= 0)].unique()
            r_seed = int(r_matched[int(torch.argmax(protos_aligned[r_matched] @ text_proto))].item())
            base["m2_gt_restrict_applicable"] = True
            base["m2_seed_prec_gtrestrict"] = round(_seed_prec(labels == r_seed, gt), 3)
        else:
            base["m2_gt_restrict_applicable"] = False

    # ── Measurement 3: mechanism-variant oracles ──
    def run_variant(mode: str, cat_proto: torch.Tensor):
        """Returns (iou, applicable, seed_prec). Aggregation always uses sim_expr."""
        sim_cat = torch.einsum("c,chw->hw", cat_proto, feat_aligned)
        if mode == "3b":
            seed_signal = sim_cat * sim_expr
            cand = seed_signal > torch.quantile(seed_signal, model.cand_quantile)
        else:                                                       # 3a / 3c: category circles candidates
            cand = sim_cat > torch.quantile(sim_cat, model.cand_quantile)
        m_ids = labels[cand & (labels >= 0)].unique()
        if m_ids.numel() == 0:
            return v2_iou, False, v2_seed_prec
        if mode == "3a":                                           # seed by category prototype
            v_seed = int(m_ids[int(torch.argmax(protos_aligned[m_ids] @ cat_proto))].item())
        elif mode == "3c":                                         # seed by full-expression prototype
            v_seed = int(m_ids[int(torch.argmax(protos_aligned[m_ids] @ text_proto))].item())
        else:                                                      # 3b: seed by mean fused score
            fused_mean = torch.stack(
                [seed_signal[labels == int(k)].mean() for k in m_ids])
            v_seed = int(m_ids[int(torch.argmax(fused_mean))].item())
        mask = _aggregate_from_seed(
            v_seed, labels, K, feat_native_flat, sim_expr, model.merge_threshold, h, w)
        return finalize_iou(mask, img, orig_size, gt_full), True, _seed_prec(labels == v_seed, gt)

    for mode in _VARIANTS:
        if gt_in_bank:
            iou, ok, sp = run_variant(mode, cat_protos[gt_idx])
            base[f"v{mode}_gt_iou"] = round(iou, 3)
            base[f"v{mode}_gt_applic"] = ok
            base[f"v{mode}_gt_seedprec"] = round(sp, 3)
        # realistic expression-readout category (no GT)
        top1_i = int(torch.argmax(cat_protos @ text_proto).item())
        iou, ok, sp = run_variant(mode, cat_protos[top1_i])
        base[f"v{mode}_expr_iou"] = round(iou, 3)
        base[f"v{mode}_expr_applic"] = ok
        base[f"v{mode}_expr_seedprec"] = round(sp, 3)

    return base


# ──────────────────────────── reporting ────────────────────────────

def _rate(sel, key):
    vals = [e[key] for e in sel if key in e]
    return mean(vals) if vals else float("nan")


def build_report(episodes, args):
    splits = [f"{d}/{s}" for d, s in _PACKAGE_SPLITS]
    scored = [e for e in episodes if e.get("gt_in_bank")]
    seeded = [e for e in scored if e.get("has_seed")]
    n = len(episodes)
    L: list[str] = []

    L.append("## WS1b — category-gate ceiling decomposition + mechanism-variant oracles (issue #32)\n")
    L.append(
        f"**Provenance.** Tuning package `{os.path.basename(args.data_root.rstrip('/'))}`, "
        f"splits {', '.join(splits)}; sample={args.sample}/split (seed {args.seed}); "
        f"device {args.device}; frozen hyperparams tau={model_hp(args,'tau')} "
        f"cand_quantile={model_hp(args,'cand_quantile')} "
        f"merge_threshold={model_hp(args,'merge_threshold')} image_size=1024. "
        f"Episodes={n} (GT category in COCO-80 bank={len(scored)}; with a candidate "
        f"seed={len(seeded)}). Category bank: COCO-80.\n")
    L.append(
        "**Red line.** GT `category_name` is used ONLY to score cluster/seed readout "
        "and to define the GT-category oracle variants. It never enters the v2 path "
        "nor the realistic (expression-readout) variants. No inference code changed.\n")
    L.append(
        "**Proxy caveat (inherited from WS1).** The package ships no per-pixel COCO "
        "instance map, so cluster categories are zero-shot readouts of a region "
        "(aligned features vs COCO-80 prototypes), and GT-majority is defined by "
        "feature-grid overlap with the downsampled GT mask. The GT-category oracles "
        "are the ceiling of THIS readout-driven mechanism, not of a perfect "
        "per-pixel class oracle.\n")

    # ── Measurement 1 ──
    L.append("### 1. Cluster-side categorization accuracy (suspect 1)\n")
    L.append("A *GT-majority* cluster has >50% of its feature-grid patches inside the "
             "downsampled GT mask. Cluster-level accuracy = share of GT-majority "
             "clusters whose zero-shot readout argmax equals the GT category. "
             "Episode-level = share of episodes (with >=1 GT-majority cluster) where "
             "at least one GT-majority cluster reads out as GT. False-positive rate = "
             "share of non-GT-majority clusters that read out as the target category.\n")
    rows = []
    for d, s in _PACKAGE_SPLITS + [("**all**", "")]:
        if d == "**all**":
            sel = [e for e in scored if "m1_n_gt_majority" in e]
            label = "**all**"
        else:
            sel = [e for e in scored
                   if e["dataset"] == d and e["split"] == s and "m1_n_gt_majority" in e]
            label = f"{d}/{s}"
        gtmaj = sum(e["m1_n_gt_majority"] for e in sel)
        gtmaj_hit = sum(e["m1_n_gt_majority_read_gt"] for e in sel)
        nongt = sum(e["m1_n_nongt"] for e in sel)
        nongt_hit = sum(e["m1_n_nongt_read_gt"] for e in sel)
        ep_has = [e for e in sel if e["m1_n_gt_majority"] > 0]
        ep_rate = mean([e["m1_episode_has_gt_read"] for e in ep_has]) if ep_has else float("nan")
        rows.append([
            label, len(sel), gtmaj,
            f"{gtmaj_hit/max(gtmaj,1):.0%}",
            f"{ep_rate:.0%}" if ep_has else "—",
            f"{nongt_hit/max(nongt,1):.0%}",
        ])
    L.append(md_table(
        ["split", "episodes", "GT-maj clusters", "cluster-acc",
         "episode-level (>=1 hit)", "FP rate (non-GT->target)"], rows) + "\n")

    # ── Measurement 2 ──
    L.append("### 2. Within-class selection quality (suspect 2)\n")
    L.append("On episodes where the GT-class restriction is applicable (>=1 candidate "
             "cluster reads out as GT), the seed hit rate (seed_prec >= 0.3) before "
             "(v2 full-expression argmax) vs after (argmax restricted to GT-class "
             "candidate clusters). If 'after' is not far above 'before', the "
             "full-expression argmax picks the wrong cluster even inside the right "
             "class — the class restriction alone does not fix seeding.\n")
    rows = []
    for d, s in _PACKAGE_SPLITS + [("**all**", "")]:
        if d == "**all**":
            sel = [e for e in seeded if e.get("m2_gt_restrict_applicable")]
            label = "**all**"
        else:
            sel = [e for e in seeded if e.get("m2_gt_restrict_applicable")
                   and e["dataset"] == d and e["split"] == s]
            label = f"{d}/{s}"
        if not sel:
            rows.append([label, 0, "—", "—", "—"])
            continue
        before = mean([e["seed_prec"] >= 0.3 for e in sel])
        after = mean([e["m2_seed_prec_gtrestrict"] >= 0.3 for e in sel])
        rows.append([label, len(sel), f"{before:.0%}", f"{after:.0%}", f"{after-before:+.0%}"])
    L.append(md_table(
        ["split", "n applicable", "seed-hit before (v2)",
         "seed-hit after (GT-class)", "Δ"], rows) + "\n")

    # ── Measurement 3 ──
    L.append("### 3. Mechanism-variant oracles\n")
    L.append("mIoU delta over the v2 seed (all episodes with a candidate seed; "
             "variants that fall back to v2 contribute Δ=0). In all variants the "
             "aggregation cross_sim keeps the full-expression map; only candidate "
             "localization + seed selection change. **ORACLE(GT)** uses the GT "
             "category; **realistic(expr)** uses the expression's own COCO readout "
             "(no GT). Correct-seed Δ is the guard (no regression on episodes whose "
             "v2 seed is already correct).\n")

    def variant_block(src: str):
        out = ["", f"**{src.upper()} source**", ""]
        rows = []
        v2_all = mean([e["v2_iou"] for e in seeded])
        for mode in _VARIANTS:
            ik, ak = f"v{mode}_{src}_iou", f"v{mode}_{src}_applic"
            sub = [e for e in seeded if ik in e]
            if not sub:
                rows.append([mode, 0, "—", "—", "—", "—", "—"]); continue
            v2 = mean([e["v2_iou"] for e in sub])
            vv = mean([e[ik] for e in sub])
            applic = mean([e[ak] for e in sub])
            cseed = [e for e in sub if e.get("correct_seed")]
            d_cs = mean([e[ik] - e["v2_iou"] for e in cseed]) if cseed else float("nan")
            wt = [e for e in sub if e.get("wrong_target")]
            d_wt = mean([e[ik] - e["v2_iou"] for e in wt]) if wt else float("nan")
            rows.append([mode, len(sub), round(vv, 3), f"{vv-v2:+.3f}",
                         f"{applic:.0%}", f"{d_cs:+.3f}", f"{d_wt:+.3f}"])
        out.append(md_table(
            ["variant", "n", f"{src} mIoU", "Δ vs v2", "applic.",
             "Δ correct-seed (guard)", "Δ wrong-target"], rows))
        # per-split ΔGT/Δexpr breakdown
        prows = []
        for d, s in _PACKAGE_SPLITS:
            sub = [e for e in seeded if e["dataset"] == d and e["split"] == s]
            row = [f"{d}/{s}", len(sub)]
            for mode in _VARIANTS:
                ik = f"v{mode}_{src}_iou"
                ss = [e for e in sub if ik in e]
                row.append(f"{mean([e[ik]-e['v2_iou'] for e in ss]):+.3f}" if ss else "—")
            prows.append(row)
        out.append("")
        out.append("_Per-split Δ:_")
        out.append(md_table(["split", "n"] + [f"Δ {m}" for m in _VARIANTS], prows))
        return "\n".join(out) + "\n"

    L.append(variant_block("gt"))
    # realistic only matters for the verdict on variants clearing +0.03; report it
    # for all but flag the gate.
    L.append(variant_block("expr"))

    L.append(_verdict_section(seeded))
    return "\n".join(L)


def _verdict_section(seeded):
    """Map GT-oracle deltas onto the HITL stopping line (#27). No GO/NO-GO call."""
    L = ["### Mapping onto the stopping line (#27)\n"]
    L.append(f"Stopping line: any variant GT-oracle Δ >= +{_GO_FLOOR:.2f} → GO to WS2 "
             f"with it; all variants < +{_DEAD_FLOOR:.2f} → category route dead, pivot "
             f"to module 2; in between → gray zone, human verdict.\n")
    rows = []
    best = None
    for mode in _VARIANTS:
        ik = f"v{mode}_gt_iou"
        sub = [e for e in seeded if ik in e]
        if not sub:
            continue
        d_gt = mean([e[ik] - e["v2_iou"] for e in sub])
        ek = f"v{mode}_expr_iou"
        esub = [e for e in seeded if ek in e]
        d_ex = mean([e[ek] - e["v2_iou"] for e in esub]) if esub else float("nan")
        cseed = [e for e in sub if e.get("correct_seed")]
        d_cs = mean([e[ik] - e["v2_iou"] for e in cseed]) if cseed else float("nan")
        zone = ("GO (>=+0.05)" if d_gt >= _GO_FLOOR
                else "dead (<+0.03)" if d_gt < _DEAD_FLOOR else "gray (+0.03..+0.05)")
        rows.append([mode, f"{d_gt:+.3f}", zone, f"{d_cs:+.3f}",
                     f"{d_ex:+.3f}" if d_gt >= _DEAD_FLOOR else "n/a (GT < floor)"])
        if best is None or d_gt > best[1]:
            best = (mode, d_gt)
    L.append(md_table(
        ["variant", "ΔGT-oracle", "zone", "Δ correct-seed (guard)",
         "Δexpr (realistic; only if ΔGT>=+0.03)"], rows) + "\n")
    if best is not None:
        m, d = best
        verdict = ("GO candidate" if d >= _GO_FLOOR
                   else "category route DEAD" if d < _DEAD_FLOOR else "GRAY ZONE")
        L.append(f"Best GT-oracle variant: **{m}** at {d:+.3f} → **{verdict}**. "
                 f"Final GO/NO-GO is the human's on #27.\n")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="../refcoco_tuning_package_seed42")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/ws1b_ceiling")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--cand-quantile", type=float, default=None)
    parser.add_argument("--merge-threshold", type=float, default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

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

    cat_protos = torch.stack([model.dinotxt.encode_expression(c) for c in COCO_CATEGORIES])

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
                                        dataset, split, cat_protos, args.device)
            episodes.append(base)
            print(f"[{dataset}/{split} {j+1}/{len(recs)}] "
                  f"v2={base.get('v2_iou', 0):.3f} "
                  f"3a={base.get('v3a_gt_iou', float('nan')):.3f} "
                  f"3b={base.get('v3b_gt_iou', float('nan')):.3f} "
                  f"3c={base.get('v3c_gt_iou', float('nan')):.3f} "
                  f"gt={base['gt_category']}")

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
