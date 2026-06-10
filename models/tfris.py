"""TFRIS: training-free referring image segmentation with a frozen DINOv3 backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image

from models.dinotxt import DinoTxtEncoder
from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import build_transform, load_image
from utils.refinement import upsample_mask, init_crf, crf_refine


class TFRIS(nn.Module):
    """Training-free referring image segmentation using frozen DINOv3 + dino.txt."""

    def __init__(
        self,
        dinotxt: DinoTxtEncoder,
        image_size: int = 1024,
        tau: float = 0.6,
        merge_threshold: float = 0.2,
        cand_quantile: float = 0.9,
        mask_refiner: str = "bilinear",
        resize_to_orig_size: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.dinotxt = dinotxt
        self.device = device
        self.image_size = image_size
        self.tau = tau
        self.merge_threshold = merge_threshold
        self.cand_quantile = cand_quantile
        self.mask_refiner = mask_refiner
        self.resize_to_orig_size = resize_to_orig_size

        if mask_refiner == 'crf':
            self._crf, self._crf_band_px, self._crf_p_core = init_crf(image_size, self.device)

        self._transform = build_transform(image_size)
        self.reset_state()

    def reset_state(self) -> None:
        """Clear the cached target image, its original size, and the Text Prototype."""
        self._tgt_image = None
        self._orig_tgt_size = None
        self._text_prototype = None

    # ──────── State setup ────────

    def set_target(self, image: str | Image.Image | torch.Tensor) -> None:
        """Set target image from a file path, PIL Image, or tensor."""
        img_tensor, self._orig_tgt_size = load_image(image, self._transform, self.device)
        self._tgt_image = img_tensor

    def set_text(self, expression: str) -> None:
        """Set the Referring Expression, encoded as a Text Prototype."""
        self._text_prototype = self.dinotxt.encode_expression(expression)

    # ──────── Public API ────────

    def segment(self) -> torch.Tensor:
        """Segment the referred object using the previously set target and expression.

        Returns:
            pred_mask: (H, W) boolean mask.
        """
        if self._tgt_image is None:
            raise RuntimeError('segment() requires a target image; call set_target() first.')
        if self._text_prototype is None:
            raise RuntimeError('segment() requires a referring expression; call set_text() first.')
        pred = self.predict_mask(self._tgt_image, self._text_prototype)

        self.reset_state()
        return pred

    # ──────── Inference ────────

    @torch.no_grad()
    def predict_mask(self, tgt_image: torch.Tensor, text_prototype: torch.Tensor) -> torch.Tensor:
        """Segment the object referred to by the Text Prototype in the target image.

        Args:
            tgt_image: (1, C, H, W) target image.
            text_prototype: (C,) L2-normalized Text Prototype.

        Returns:
            pred_mask: (H, W) boolean mask.
        """
        # Feature extraction: one backbone pass yields both spaces (ADR 0001)
        feat_native, feat_aligned = self.dinotxt.encode_image_features(tgt_image)
        _, h, w = feat_native.shape

        # Candidate Localization: Aligned Features vs Text Prototype, quantile rule
        sim = torch.einsum('c,chw->hw', text_prototype, feat_aligned)
        candidate_mask = sim > torch.quantile(sim, self.cand_quantile)
        if candidate_mask.sum() == 0:
            return self._finalize_mask(candidate_mask, tgt_image)

        # Fine-grained clustering on Native Features
        feat_native_flat = feat_native.reshape(feat_native.shape[0], -1).permute(1, 0)
        cluster_labels = agglomerative_clustering(feat_native_flat, self.tau).reshape(h, w)
        K = int(cluster_labels.max().item()) + 1

        feat_aligned_flat = feat_aligned.reshape(feat_aligned.shape[0], -1).permute(1, 0)
        cluster_protos = compute_cluster_prototypes(
            feat_aligned_flat, cluster_labels.view(-1), K
        )

        # Seed selection and cluster aggregation
        pred_mask = self._seed_and_aggregate(
            candidate_mask, cluster_labels, cluster_protos, K,
            text_prototype, feat_native_flat, sim, h, w
        )

        return self._finalize_mask(pred_mask, tgt_image)

    # ──────── Seed selection and cluster aggregation ────────

    def _seed_and_aggregate(
        self,
        candidate_mask: torch.Tensor,
        cluster_labels: torch.Tensor,
        cluster_protos: torch.Tensor,
        K: int,
        text_prototype: torch.Tensor,
        feat_native_flat: torch.Tensor,
        sim: torch.Tensor,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """Select the seed cluster and aggregate remaining clusters."""
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if matched_mask.sum() == 0:
            return candidate_mask

        matched_ids, n_pixels = cluster_labels[matched_mask].unique(return_counts=True)

        # Area weighting
        all_areas = cluster_labels[cluster_labels >= 0].unique(return_counts=True)[1]
        per_cluster = torch.zeros(K, device=cluster_labels.device)
        per_cluster[matched_ids] = n_pixels.float()
        area_weights = per_cluster / all_areas

        # Seed selection: cluster with highest similarity to the Text Prototype
        protos_matched = cluster_protos[matched_ids]
        cross_sim_matched = protos_matched @ text_prototype
        seed_idx = int(torch.argmax(cross_sim_matched).item())
        seed_id = matched_ids[seed_idx].item()

        # Intra-image similarity to seed (Native Features)
        native_protos = compute_cluster_prototypes(
            feat_native_flat, cluster_labels.view(-1), K
        )
        intra_sim = torch.einsum('c,kc->k', native_protos[seed_id], native_protos)

        # Cross-image similarity per cluster (Aligned Features vs Text Prototype)
        cross_sim = torch.empty(K, device=sim.device, dtype=sim.dtype)
        for k in range(K):
            idx = (cluster_labels == k)
            cross_sim[k] = sim[idx].mean() if idx.any() else 0.0

        # Combined score
        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined *= area_weights

        final_mask = torch.zeros(h, w, dtype=torch.bool, device=cluster_labels.device)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > self.merge_threshold
        # The aligned space bounds text-vs-patch cosines well below the range
        # INSID3's cross-image similarities lived in, so even the seed's combined
        # score can fall under merge_threshold; restore INSID3's implicit
        # guarantee that the seed cluster is always part of the output (ADR 0001).
        final_mask |= (cluster_labels == seed_id)
        return final_mask

    # ──────── Mask finalization ────────

    def _finalize_mask(self, mask: torch.Tensor, tgt_image: torch.Tensor) -> torch.Tensor:
        """Upsample feature-resolution mask, optionally with CRF refinement."""
        H, W = tgt_image.shape[-2:]
        up = upsample_mask(mask, H, W)
        if self.mask_refiner == 'crf':
            up = crf_refine(self._crf, self._crf_band_px, self._crf_p_core, tgt_image, up)
        # Resize to original target resolution
        if self.resize_to_orig_size:
            up = upsample_mask(up, self._orig_tgt_size[0], self._orig_tgt_size[1])
        return up
