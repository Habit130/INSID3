# TFRIS: Training-Free Referring Image Segmentation

Given a target image and a **Referring Expression** (a natural-language phrase
identifying exactly one object instance), TFRIS produces the mask of the
referred object — with **no training**: a frozen DINOv3 backbone plus the
frozen [dino.txt](https://github.com/facebookresearch/dinov3) vision head and
text encoder.

The method derives from [INSID3](https://arxiv.org/abs/2603.28480)
(CVPR 2026, training-free in-context segmentation): the reference image + mask
pathway is replaced by a text pathway. One backbone pass yields two feature
spaces (ADR 0001):

- **Native Features** — patch features straight from the DINOv3 backbone, used
  for intra-image work (clustering, intra-image similarity).
- **Aligned Features** — patch features after the dino.txt vision head, living
  in the joint image-text space, used wherever similarity against text is
  computed.

The Referring Expression is embedded as a **Text Prototype** (dino.txt text
encoder with the Text Bias removed, ADR 0002). **Candidate Localization**
scores Aligned Features against the Text Prototype; agglomerative clustering
of Native Features plus seed selection and cluster aggregation produce the
final mask.

## Environment Setup

### Option 1: Conda

```bash
conda create --name tfris python=3.10 -y
conda activate tfris
pip install -r requirements.txt
```

**Optional:** for CRF-based mask refinement, also install:
```bash
git clone https://github.com/netw0rkf10w/CRF.git
cd CRF
python setup.py install
cd ..
```

### Option 2: uv (Linux x86_64; includes CRF)

```bash
uv sync
source .venv/bin/activate
```

## Weights

All weights load fully offline from `pretrain/` (download from the official
[DINOv3 repository](https://github.com/facebookresearch/dinov3)):

| File | Role |
|---|---|
| `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | ViT-L/16 backbone |
| `dinov3_vitl16_dinotxt_vision_head_and_text_encoder.pth` | dino.txt vision head + text encoder |
| `bpe_simple_vocab_16e6.txt.gz` | BPE vocabulary for the dino.txt tokenizer |

Convention: the weights live in a shared directory one level above the repo,
exposed inside the repo as `pretrain/` — on Windows via a directory junction
(`pretrain -> ..\pretrain`), elsewhere via a symlink or a plain directory.
Construction fails at startup with a descriptive error if any file is missing.

## Minimal Usage

```python
from models import build_tfris
from utils.visualization import visualize_prediction_referring as visualize

target_image_path = "assets/cat_image.jpg"
expression = "a cat"

# Build model (frozen; loads weights from pretrain/)
model = build_tfris()

# Set target image and Referring Expression
model.set_target(target_image_path)
model.set_text(expression)

# Predict — (H, W) boolean mask at source resolution; state resets afterwards
pred_mask = model.segment()

# Save visualization
visualize(target_image_path, expression, pred_mask, "cat_pred.png")
```

For CRF refinement: `build_tfris(mask_refiner="crf")`. For faster inference,
reduce the input resolution (default `1024`), e.g. `build_tfris(image_size=768)`.

To render the raw patch-vs-text similarity heatmap for one image and one
expression:

```bash
python similarity_heatmap.py --image assets/cat_image.jpg --expression "a cat"
```

## Data

Evaluation uses the self-contained RefCOCO eval package (a sibling directory
of the repo). See [docs/data.md](docs/data.md) for its layout and policy.

## Evaluation

```bash
python inference_referring.py --data-root ../refcoco_eval_package --dataset refcoco --split val --exp-name tfris-refcoco-val
```

Main arguments (see `opts.py`):

- `--dataset`: `refcoco`, `refcoco+`, or `refcocog`
- `--split`: `val`/`testA`/`testB` (refcoco, refcoco+); `val`/`test_U`/`test_G` (refcocog)
- `--data-root`: eval package root
- `--image-size` (default `1024`); hyperparameters `--tau` (0.6), `--merge-thresh` (0.2), `--cand-quantile` (0.9)
- `--limit N`: cap to the first N expressions (prediction contract validated, official evaluator skipped)
- `--sample N`: evaluate a random subset of N expressions (drawn with `--seed`); script-computed estimates land in `sampled_metrics.json`
- `--crf-mask-refinement`: enable CRF post-processing

Each run writes one binary PNG per `sent_id` under
`output/<exp-name>_<timestamp>/predictions/<dataset>/<split>/` and, for full
splits, invokes the eval package's official `evaluate_predictions.py`,
appending its report (expression/instance-weighted mIoU, overall IoU,
Precision@0.5/0.7/0.9) to the run's `log.txt`.

## Citation

TFRIS builds directly on INSID3:

```bibtex
@inproceedings{cuttano2026insid3,
  title     = {{INSID3}: Training-Free In-Context Segmentation with {DINOv3}},
  author    = {Claudia Cuttano and Gabriele Trivigno and Christoph Reich and Daniel Cremers and Carlo Masone and Stefan Roth},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## Acknowledgements

- [DINOv3 / dino.txt](https://github.com/facebookresearch/dinov3)
- [INSID3](https://visinf.github.io/INSID3/)
