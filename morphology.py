"""Cell morphology descriptors and cross-modal Top-K retrieval utilities.

The descriptor is computed from an instance-label mask and an image channel.
Area and nuclear intensity are normalized within each patch so that the values
are invariant to a global image scale and comparable across patches.  Retrieval
uses only these segmentation-derived quantities and centroid distance; learned
cell embeddings are not used to construct the candidate set.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.measure import regionprops


EPS = 1e-8


def _as_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    if image.ndim == 3:
        return image[..., 0].astype(np.float32, copy=False)
    raise ValueError(f"Expected a 2-D image or 3-D image, got shape {image.shape}")


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize intensities by patch percentiles, avoiding uint8 scale dependence."""
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi - lo < EPS:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def extract_morphology_descriptors(
    image: np.ndarray,
    instance_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract five descriptors for every non-background instance.

    Returns:
        descriptors: ``[N, 5]`` columns are normalized area, eccentricity,
            solidity, axis ratio, and normalized mean nuclear intensity.
        centroids: ``[N, 2]`` centroids in ``(row, column)`` coordinates.
        labels: instance labels corresponding to the rows.

    Area is divided by the patch median instance area.  Mean intensity is
    robustly min-max normalized using the 1st and 99th image percentiles.
    Eccentricity, solidity, and axis ratio are dimensionless region properties.
    """
    image_gray = _as_gray(image)
    mask = np.asarray(instance_mask)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D instance mask, got shape {mask.shape}")
    if image_gray.shape != mask.shape:
        raise ValueError(
            f"Image and mask shapes must match, got {image_gray.shape} and {mask.shape}"
        )

    intensity_image = _robust_normalize(image_gray)
    props = regionprops(mask.astype(np.int32), intensity_image=intensity_image)
    if not props:
        return (
            np.empty((0, 5), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    areas = np.asarray([p.area for p in props], dtype=np.float32)
    area_scale = max(float(np.median(areas)), EPS)
    rows = []
    centroids = []
    labels = []
    for prop in props:
        minor = max(float(prop.minor_axis_length), EPS)
        major = max(float(prop.major_axis_length), minor)
        axis_ratio = major / minor
        rows.append(
            [
                float(prop.area) / area_scale,
                float(prop.eccentricity),
                float(prop.solidity),
                axis_ratio,
                float(prop.mean_intensity),
            ]
        )
        centroids.append(prop.centroid)
        labels.append(prop.label)

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(centroids, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
    )


def compute_topk_retrieval(
    query_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    query_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    k: int = 5,
    radius: float = 32.0,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve Top-K reference cells for each query cell.

    The score is ``alpha * exp(-distance^2 / (2*sigma^2)) +
    (1-alpha) * cosine(descriptor_q, descriptor_r)`` with ``sigma=radius/2``.
    Candidates are first restricted to the spatial radius.  If no reference
    cell falls inside the radius, the nearest reference cells are used so that
    every valid query has a target.  Returned indices are padded with ``-1``.
    """
    q_desc = np.asarray(query_descriptors, dtype=np.float32)
    r_desc = np.asarray(reference_descriptors, dtype=np.float32)
    q_ctr = np.asarray(query_centroids, dtype=np.float32)
    r_ctr = np.asarray(reference_centroids, dtype=np.float32)
    if q_desc.ndim != 2 or r_desc.ndim != 2 or q_desc.shape[1] != r_desc.shape[1]:
        raise ValueError("Query and reference descriptors must be [N,D] and [M,D]")
    if q_ctr.shape != (len(q_desc), 2) or r_ctr.shape != (len(r_desc), 2):
        raise ValueError("Centroids must have shape [N,2] and [M,2]")
    if k < 1 or radius <= 0 or not 0 <= alpha <= 1:
        raise ValueError("Require k>=1, radius>0, and alpha in [0,1]")

    result = np.full((len(q_desc), k), -1, dtype=np.int64)
    scores_out = np.full((len(q_desc), k), -np.inf, dtype=np.float32)
    if len(r_desc) == 0 or len(q_desc) == 0:
        return result, scores_out

    q_norm = np.linalg.norm(q_desc, axis=1, keepdims=True).clip(min=EPS)
    r_norm = np.linalg.norm(r_desc, axis=1, keepdims=True).clip(min=EPS)
    cosine = (q_desc / q_norm) @ (r_desc / r_norm).T
    sigma = radius / 2.0

    for qi in range(len(q_desc)):
        delta = r_ctr - q_ctr[qi]
        distances = np.linalg.norm(delta, axis=1)
        candidates = np.flatnonzero(distances <= radius)
        if len(candidates) == 0:
            candidates = np.argsort(distances)[: min(k, len(r_desc))]
        spatial = np.exp(-(distances[candidates] ** 2) / (2.0 * sigma**2))
        scores = alpha * spatial + (1.0 - alpha) * cosine[qi, candidates]
        order = np.argsort(-scores)[:k]
        selected = candidates[order]
        result[qi, : len(selected)] = selected
        scores_out[qi, : len(selected)] = scores[order]
    return result, scores_out


def gather_topk_prototypes(reference_features, topk_indices):
    """Average reference features selected by ``compute_topk_retrieval``."""
    import torch

    features = reference_features if torch.is_tensor(reference_features) else torch.as_tensor(reference_features)
    indices = topk_indices if torch.is_tensor(topk_indices) else torch.as_tensor(topk_indices)
    if indices.ndim != 2:
        raise ValueError("topk_indices must have shape [N,K]")
    safe = indices.clamp_min(0).to(device=features.device, dtype=torch.long)
    gathered = features[safe]
    valid = (indices >= 0).to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    return (gathered * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
