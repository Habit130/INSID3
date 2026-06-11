# TFRIS v1 — Phase 1 Report (frozen 2026-06-10)

Phase 1 goal: adapt the INSID3 in-context pipeline into text-guided,
training-free referring image segmentation, and validate it end-to-end with
zero tuning. **Status: complete.** This document freezes what was built, how
it was validated, and what the numbers say. The corresponding code state is
tagged `v1.0-zero-tuning` on `main`.

## What was delivered (issues #1–#7, all closed)

- **dino.txt encoder** (`models/dinotxt.py`): offline loading of the ViT-L/16
  backbone + dino.txt vision head and text encoder; one backbone pass yields
  Native Features and Aligned Features (ADR 0001); Referring Expressions are
  embedded as Text Prototypes with the Text Bias removed via a fixed
  neutral-text bank (ADR 0002).
- **TFRIS model** (`models/tfris.py`): `build_tfris()` → `set_target()` →
  `set_text()` → `segment()`, fully frozen. Candidate Localization (quantile
  rule on Aligned-vs-Text similarity) → agglomerative clustering of Native
  Features → seed selection + cluster aggregation → mask finalization.
- **RefCOCO dataset module** (`datasets/refcoco.py`): one item per Referring
  Expression, reconciled against the eval package's `validation_report.json`.
- **Inference script** (`inference_referring.py`): prediction contract
  (one binary PNG per `sent_id` at source resolution), official-evaluator
  closure for full splits, `--limit` capped runs, `--sample N` seeded
  random-subset runs with script-computed estimates.
- **Repo convergence**: the in-context pathway (few-shot datasets, INSID3
  model, keypoint matching) is removed; the codebase is single-task RIS.
- Behavioral test suite: 36 passed, 1 skipped (CRF not installed).

## Validation verdict

Text-guided training-free segmentation **works end-to-end**: 1,800 sampled
episodes across all 9 refcoco-family splits ran without failure, every
prediction set satisfied the contract, and outputs are structured (simple
non-spatial expressions segment markedly better than chance; the minimal-usage
test asserts correct localization for "a cat").

## v1 numbers (sampled protocol: random 200 expressions per split, seed 0)

All hyperparameters at defaults — `tau=0.6, merge_thresh=0.2,
cand_quantile=0.9, image_size=1024`, no CRF. Script-computed estimates
(the official evaluator requires complete splits); `sampled_metrics.json`
in each run directory. Full table and run dirs: issue #6 closing comment.

| split | mIoU | P@0.5 | P@0.7 | overall IoU |
|---|---|---|---|---|
| refcoco val / testA / testB | 0.106 / 0.104 / 0.125 | 0.060 / 0.050 / 0.090 | 0.035 / 0.020 / 0.035 | 0.106 / 0.095 / 0.125 |
| refcoco+ val / testA / testB | 0.158 / 0.137 / 0.138 | 0.095 / 0.070 / 0.095 | 0.040 / 0.035 / 0.035 | 0.169 / 0.123 / 0.137 |
| refcocog val / test_U / test_G | 0.169 / 0.155 / 0.144 | 0.145 / 0.100 / 0.110 | 0.080 / 0.025 / 0.060 | 0.121 / 0.117 / 0.138 |

Cross-check: the interrupted full refcoco/val run scored mIoU 0.131 over its
first 2630 expressions, consistent with the sampled estimate.

## Failure-mode decomposition (2×40 diagnosed episodes, refcoco/val)

1. **The ceiling is not the clustering.** Oracle cluster-selection IoU is
   ~0.66–0.68 against an actual ~0.13: with the existing Native-Feature
   clusters, perfect cluster selection alone reaches ~0.67.
2. **Wrong target ≈ 50%.** The argmax of the text-vs-patch similarity lands
   inside the GT object only 42–50% of the time. Spatial expressions
   ("left", "closest", "2nd") do far worse than non-spatial ones
   (mIoU 0.09 vs 0.20) — the aligned-space weakness the PRD accepted for v1,
   but at a higher rate than hoped.
3. **Aggregation never expands.** In ~100% of episodes no cluster passes
   `merge_thresh=0.2`: aligned-space text-vs-patch cosines live on a much
   smaller scale than the image-vs-image similarities INSID3's threshold was
   calibrated for. The output degenerates to the seed cluster alone
   (seed recall ~0.17–0.19) — hence the fragmentary masks.

## Standing policies

- The eval package is for evaluation only — no training, no tuning against it
  (any recalibration must use other data).
- Benchmarks run as random 200-expression subsets per split (`--sample 200`),
  covering all datasets; full-split runs are not used.

## Next-step candidates (deferred, human decision)

In order of expected impact: recalibrate the aggregation score scale /
merge threshold on non-eval data; handle spatial language explicitly;
improve Candidate Localization precision. Not started — phase 2 scope.
