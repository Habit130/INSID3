# TFRIS v2 — Phase 2 Report (frozen 2026-06-11)

Phase 2 goal: fix the cluster-aggregation expansion bug that pinned v1 to
seed-only masks, with all iteration feedback drawn from the dedicated tuning
package (`refcoco_tuning_package_seed42`) and never the eval package. **Status:
complete.** This document freezes what was built, the certification numbers on
the eval package, the remaining failure decomposition, and the phase-3
candidate list. It is the phase-2 companion to `docs/v1-report.md`; the
corresponding code state is tagged on the phase-2 merge commit on `main`.

## What was delivered (issues #11–#16, all closed)

Diagnostic-first redesign of the aggregation stage, per the PRD (#10). The
frozen backbone, dino.txt heads, and the entire text-encoding path (Text
Prototype, Text Bias removal per ADR 0002) were untouched; no training anywhere.

- **#11 — Tuning-package dataset access (WS1).** The RefCOCO module reads the
  tuning package's train splits (`train`, `train_U`, `train_G`), reconciled
  against its own `validation_report.json`, so every phase-2 calibration run
  used legal data instead of the eval package.
- **#12 — v1 baseline on the tuning package (WS1).** Full-split official-evaluator
  metrics for `v1.0-zero-tuning` on all four tuning splits (mIoU 0.122 / 0.139 /
  0.158 / 0.162), the same-data baseline every later change is judged against.
- **#13 — Expansion-signal AUC diagnostic (WS1).** Promoted the throwaway
  failure harness into a maintained tool (`tools/diagnose_expansion.py`),
  measured how well each candidate expansion signal separates GT from non-GT
  clusters (AUC), and decomposed wrong-target episodes into wrong-class vs
  wrong-instance using GT `category_name` (diagnosis only). Result: the product
  `intra_sim × cross_sim_norm` is the strongest signal (AUC 0.7584); seed
  adjacency as a hard gate *reduces* separation (0.616 < 0.729).
- **#14 — HITL gate.** Chose the merge-rule form from #13's measured AUCs (soft
  product, no gates) and decided seed-selection stays out of phase-2 scope.
- **#15 — Merge-rule redesign + threshold calibration (WS2).** Replaced v1's
  hard-gated rule with the soft product score
  `score_k = intra_sim_k × cross_sim_norm_k > merge_threshold`, recalibrated to
  `merge_thresh = 0.32` on the tuning package (PR #21, ADR 0003). Incomplete-
  expansion failure share dropped 29.0% → 6.5%; tuning-split mIoU roughly
  doubled on every split.
- **#16 — Hyperparameter grid (WS3).** Grid over `tau` × `cand_quantile` ×
  `merge_thresh` on the tuning package, with the oracle cluster-subset ceiling
  reported per `tau`. **Null result for tuning** (see "One configuration" below).

## One configuration, not two

The PRD (#10, US16) asked for mechanism-fix numbers (WS2 at default
hyperparameters) and tuned numbers (WS2+WS3) reported separately. **WS3 (#16)
returned an honest null result: the orchestrator verdict kept the WS2 defaults
unchanged** (`tau=0.6, cand_quantile=0.9, merge_thresh=0.32`). The grid-best
config beat the default by only +0.0013 overall mIoU, and its per-split optima
were mutually inconsistent (tau 0.5/0.6/0.8 all appearing) — the classic
signature of overfitting a 100-episode grid. `cand_quantile` had near-zero
effect under the new rule (candidates now only pick the seed, as designed).

Consequently there is **a single certified configuration**, and the "WS2-at-
defaults" and "WS2+WS3-tuned" result sets are numerically identical. The
comparison below is therefore one column, not two. The mechanism-fix gain over
v1 is the entire phase-2 gain; the tuning gain is zero by decision. See #16 for
the full grid and verdict.

## Certification configuration

Final phase-2 configuration = current `main` defaults:

`tau=0.6, cand_quantile=0.9, merge_thresh=0.32, image_size=1024`, no CRF.

This is the single sanctioned eval-package touch for phase 2: one `--sample 200`,
seed-0 run per split across all 9 refcoco-family splits (the standing sampled
protocol). Script-computed estimates (the official evaluator requires complete
splits); `sampled_metrics.json` in each run directory under `output/v2cert-*`.
All 9 runs completed without failure (exit 0, 200/200 episodes each).

## v2 vs v1.0-zero-tuning (sampled protocol: random 200 expressions per split, seed 0)

mIoU (Δ vs v1 in parentheses):

| split | v1 mIoU | v2 mIoU | v1 P@0.5 | v2 P@0.5 | v1 P@0.7 | v2 P@0.7 | v1 ovIoU | v2 ovIoU |
|---|---|---|---|---|---|---|---|---|
| refcoco val   | 0.106 | **0.236** (+0.130) | 0.060 | 0.125 | 0.035 | 0.055 | 0.106 | 0.235 |
| refcoco testA | 0.104 | **0.255** (+0.151) | 0.050 | 0.175 | 0.020 | 0.075 | 0.095 | 0.226 |
| refcoco testB | 0.125 | **0.247** (+0.122) | 0.090 | 0.115 | 0.035 | 0.040 | 0.125 | 0.227 |
| refcoco+ val   | 0.158 | **0.276** (+0.118) | 0.095 | 0.230 | 0.040 | 0.050 | 0.169 | 0.257 |
| refcoco+ testA | 0.137 | **0.288** (+0.151) | 0.070 | 0.230 | 0.035 | 0.115 | 0.123 | 0.269 |
| refcoco+ testB | 0.138 | **0.258** (+0.120) | 0.095 | 0.150 | 0.035 | 0.050 | 0.137 | 0.254 |
| refcocog val    | 0.169 | **0.261** (+0.092) | 0.145 | 0.150 | 0.080 | 0.065 | 0.121 | 0.228 |
| refcocog test_U | 0.155 | **0.298** (+0.143) | 0.100 | 0.220 | 0.025 | 0.070 | 0.117 | 0.261 |
| refcocog test_G | 0.144 | **0.266** (+0.122) | 0.110 | 0.190 | 0.060 | 0.085 | 0.138 | 0.251 |
| **mean**        | 0.137 | **0.265** (+0.128) | 0.091 | 0.176 | 0.041 | 0.067 | 0.126 | 0.245 |

mIoU rises on every split (≈1.6–2.5×, mean 0.137 → 0.265, ~1.9×); P@0.5 and
overall IoU improve in step. P@0.7 improves on 8 of 9 splits; only refcocog/val
dips (0.080 → 0.065), within the noise floor at n=200 where each episode moves
P@0.7 by 0.005. The mean eval-package mIoU (0.265) clears the phase-2 success bar
(tuning-package mIoU ≥ 0.26) on untouched data. No regressions.

## Failure-mode decomposition (current state)

Phase 2 fixed the recoverable aggregation bug; the two structural failures the
PRD predicted now dominate, and one trade-off was deliberately accepted.

1. **The wrong-target wall (~51%, unchanged).** The text-similarity argmax
   lands inside the GT object only ~49% of the time; seed selection was not
   touched in phase 2 (#14), so this rate is identical to v1. On the
   correct-seed subset, fixed expansion lifts mIoU 0.315 → 0.756; on
   wrong-target episodes no expansion rule can rescue a seed on the wrong
   object. This caps the realistic phase-2 mIoU at ~0.30, consistent with the
   ~0.265 eval-package mean. Decomposition (#13, proxy zero-shot category
   readout): of 204 wrong-target episodes, **64% are wrong-class, 36%
   wrong-instance** — the model points at the wrong *kind* of object more often
   than the wrong instance of the right kind.
2. **Over-expansion rose (the accepted trade-off).** Removing v1's hard gates
   and lowering the effective scale moved the failure mix from incomplete-
   expansion (29.0% → 6.5%) toward over-expansion (7.0% → 20.8%, #15
   diagnostic). Net mIoU nearly doubled, so the trade was worth it, but
   over-expansion is now the larger of the two expansion errors and is the
   first thing a tighter rule would target.
3. **The clustering is still not the ceiling.** The oracle cluster-subset IoU
   at `tau=0.6` is ~0.67 on the tuning package against an actual ~0.29 — there
   is still ~0.38 of headroom that a better seed + expansion could reach with
   the *existing* clusters.

## Phase-3 candidates (deferred, human decision)

In expected-impact order, informed by the decomposition above:

1. **Category-level localization first.** Wrong-class is the single largest
   loss (64% of the 51% wrong-target wall). A "what" decomposition — resolving
   the head noun / object category before spatial or instance reasoning — is
   the highest-leverage move and unblocks everything downstream. This is the
   what-vs-which text decomposition the PRD flagged.
2. **Spatial-language handling.** Spatial expressions ("left", "closest",
   "2nd") do markedly worse than non-spatial ones (v1: mIoU 0.09 vs 0.20). The
   aligned space carries little spatial structure; explicit spatial reasoning
   (or a spatial prior over candidates) is the wrong-instance lever once
   wrong-class is addressed.
3. **The tau-refinement ceiling dividend.** WS3 (#16) showed the oracle ceiling
   rises monotonically with `tau` (0.567 at 0.5 → 0.877 at 0.8) while actual
   mIoU stays pinned at ~0.29 — the wrong-target wall binds everywhere, so
   finer clustering buys nothing *today*. Once localization is fixed, revisiting
   `tau=0.7–0.8` (ceiling 0.78–0.88) becomes a cheap, already-measured win.

## Standing policies (unchanged)

- The eval package is for evaluation only — no training, no tuning against it.
  All phase-2 calibration used `refcoco_tuning_package_seed42`; this report's
  9-split certification is the only phase-2 eval-package run.
- Benchmarks run as random 200-expression subsets per split (`--sample 200`,
  seed 0), covering all datasets; full-split runs are not used.
