# Data: the RefCOCO eval package

TFRIS evaluates on the RefCOCO family (refcoco, refcoco+, refcocog) through a
self-contained, pre-validated eval package. The package is **not** part of this
repository; it lives one level above the repo root (sibling directory
`refcoco_eval_package/`) and is passed to the inference script via
`--data-root ../refcoco_eval_package`.

Layout (see the package's own `README.md` for full details):

```text
refcoco_eval_package/
├── images/                      # COCO train2014 images
├── refcoco/<split>/             # val, testA, testB
│   ├── expressions.jsonl        # one record per Referring Expression (sent_id)
│   └── masks/                   # one GT PNG per ann_id
├── refcoco+/<split>/            # val, testA, testB
├── refcocog/<split>/            # val, test_U, test_G
├── tools/
│   ├── evaluate_predictions.py  # official evaluator (invoked by inference_referring.py)
│   └── verify_package.py
├── validation_report.json       # authoritative per-split expression/instance/image counts
└── manifest.sha256.json
```

The dataset module (`datasets/refcoco.py`) reads `expressions.jsonl` and the
mask PNGs directly from this root — no conversion or preprocessing step.
`tests/test_refcoco_dataset.py` reconciles the module against
`validation_report.json`.

**Policy:** the eval package is for evaluation only. No training, no
hyperparameter tuning against it.
