# v2 baseline (seed 42) — phase-3 control row

This table is the **phase-3 control baseline**: the `v2.0-soft-merge` configuration
re-evaluated under the project-wide fixed sampling seed **42**. Every phase-3
ablation compares against these numbers.

Configuration: `v2.0-soft-merge` — `tau=0.6`, `cand_quantile=0.9`,
`merge_thresh=0.32`, `image_size=1024`, no CRF (the current `main` defaults).
Protocol: `--sample 200`, `--seed 42`, one run per split across all 9
refcoco-family splits.

> These are **script-computed estimates** over a random 200-expression sample per
> split (`sampled_metrics.json` in each `output/v2base-s42-*` run directory), **not
> official full-split numbers**. They are for relative phase-3 comparison only.

| split | mIoU | P@0.5 | P@0.7 | P@0.9 | overall IoU |
|---|---|---|---|---|---|
| refcoco val | 0.2197 | 0.0950 | 0.0300 | 0.0000 | 0.2141 |
| refcoco testA | 0.3043 | 0.2500 | 0.0850 | 0.0050 | 0.2588 |
| refcoco testB | 0.2797 | 0.1700 | 0.0800 | 0.0000 | 0.2628 |
| refcoco+ val | 0.2714 | 0.2000 | 0.0700 | 0.0050 | 0.2417 |
| refcoco+ testA | 0.3222 | 0.2450 | 0.1150 | 0.0050 | 0.2978 |
| refcoco+ testB | 0.2650 | 0.1750 | 0.0600 | 0.0100 | 0.2369 |
| refcocog val | 0.2752 | 0.1950 | 0.1050 | 0.0100 | 0.2401 |
| refcocog test_U | 0.2895 | 0.2200 | 0.0850 | 0.0150 | 0.2392 |
| refcocog test_G | 0.2976 | 0.1850 | 0.0850 | 0.0250 | 0.2685 |
| **mean** | **0.2805** | **0.1928** | **0.0794** | **0.0083** | **0.2511** |

## Why this supersedes the seed-0 table in `docs/v2-report.md`

The certification table in `docs/v2-report.md` was sampled with **seed 0**. The
sampled protocol draws a *different* random 200 expressions per split for each
seed, so seed-0 and seed-42 estimates are over **different subsets** of the same
splits and are **not directly comparable** — per-split and mean differences
between the two tables reflect sampling variance, not any change in method or
hyperparameters (the configuration is identical). Because the project-wide
evaluation seed is fixed at 42 (`opts.py` default), phase-3 ablations are run at
seed 42; this table is the matching same-seed control so that ablation deltas are
attributable to the ablation rather than to a seed mismatch. For reference, the
seed-0 mean mIoU was 0.265 vs 0.2805 here — within sampling noise of the same
configuration.
