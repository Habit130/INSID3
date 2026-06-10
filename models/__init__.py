"""Model construction utilities for TFRIS."""

from models.dinotxt import build_dinotxt
from models.tfris import TFRIS


def build_tfris(
    *,
    image_size: int = 1024,
    tau: float = 0.6,
    merge_threshold: float = 0.2,
    cand_quantile: float = 0.9,
    mask_refiner: str = "bilinear",
    resize_to_orig_size: bool = True,
    device: str = "cuda",
):
    dinotxt = build_dinotxt(device=device)
    model = TFRIS(
        dinotxt=dinotxt,
        image_size=image_size,
        tau=tau,
        merge_threshold=merge_threshold,
        cand_quantile=cand_quantile,
        mask_refiner=mask_refiner,
        resize_to_orig_size=resize_to_orig_size,
        device=device,
    )
    for param in model.parameters():
        param.requires_grad = False
    return model


def build_tfris_from_args(args):
    return build_tfris(
        image_size=args.image_size,
        tau=args.tau,
        merge_threshold=args.merge_thresh,
        cand_quantile=args.cand_quantile,
        mask_refiner='crf' if getattr(args, 'crf_mask_refinement', False) else 'bilinear',
        # The official RefCOCO evaluator rejects size mismatches, so
        # predictions stay at source image resolution.
        resize_to_orig_size=True,
        device=args.device,
    )
