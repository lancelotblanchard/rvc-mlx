"""
Feature retrieval ("index") for RVC, in MLX.

RVC nudges the source speaker's content features towards the target voice by replacing each frame with a weighted
average of its k=8 nearest neighbours in the voice's training feature bank (weights ∝ 1 / squared_distance²), then
mixing that with the original features by `index_rate`. The WebUI uses an approximate faiss IVF index for this; here we
do an exact search with matrix multiplies, which is fast on Apple GPUs and removes faiss from the runtime.
"""

from __future__ import annotations

from typing import Tuple

import mlx.core as mx
import numpy as np


def knn(queries: mx.array, bank: mx.array, k: int = 8, chunk: int = 512) -> Tuple[mx.array, mx.array]:
    """
    Exact k-nearest-neighbour search under squared L2 distance.

    :param queries: `(N, D)`.
    :param bank: `(M, D)`.
    :returns: `(sq_dist, idx)`, both `(N, k)`, unsorted within a row (callers only take weighted sums).
    """
    k = min(k, bank.shape[0])
    queries = queries.astype(mx.float32)
    bank = bank.astype(mx.float32)
    bank_sq = mx.sum(bank * bank, axis=-1)
    dists, idxs = [], []
    for s in range(0, queries.shape[0], chunk):
        q = queries[s : s + chunk]
        d = mx.sum(q * q, axis=-1, keepdims=True) - 2.0 * (q @ bank.T) + bank_sq[None]
        idx = mx.argpartition(d, kth=k - 1, axis=-1)[:, :k]
        dists.append(mx.maximum(mx.take_along_axis(d, idx, axis=-1), 0.0))
        idxs.append(idx)
    return mx.concatenate(dists, axis=0), mx.concatenate(idxs, axis=0)


def retrieve(feats: mx.array, bank: mx.array, k: int = 8) -> mx.array:
    """
    RVC's retrieval: for each frame, the inverse-squared-distance² weighted mean of its k nearest bank vectors.

    :param feats: `(T, D)` or `(1, T, D)`.
    :returns: same shape as `feats`.
    """
    squeeze = feats.ndim == 3
    x = feats[0] if squeeze else feats
    d, idx = knn(x, bank, k)
    w = 1.0 / mx.maximum(d, 1e-12) ** 2  # faiss returns squared L2, RVC weights by np.square(1 / score)
    w = w / mx.sum(w, axis=-1, keepdims=True)
    out = mx.sum(bank.astype(mx.float32)[idx] * w[..., None], axis=1).astype(feats.dtype)
    return out[None] if squeeze else out


def retrieve_blended(feats: mx.array, banks, weights=None, k: int = 8) -> mx.array:
    """
    Retrieval against several voices' banks at once (voice blending): the weighted mean of each bank's `retrieve`.
    """
    banks = list(banks)
    weights = [1.0] * len(banks) if weights is None else [float(w) for w in weights]
    if len(weights) != len(banks) or sum(weights) <= 0:
        raise ValueError("Need one positive-summing weight per bank")
    total = sum(weights)
    out = None
    for w, bank in zip(weights, banks):
        if w == 0:
            continue
        r = retrieve(feats, bank, k) * (w / total)
        out = r if out is None else out + r
    return out


def kmeans(x: np.ndarray, n_clusters: int, n_iter: int = 20, seed: int = 0) -> np.ndarray:
    """Plain Lloyd k-means on MLX (k-means++-free random init); returns `(n_clusters, D)` float32 centroids."""
    rng = np.random.default_rng(seed)
    data = mx.array(np.asarray(x, dtype=np.float32))
    centroids = data[mx.array(rng.choice(x.shape[0], n_clusters, replace=False))]
    for _ in range(n_iter):
        _, assign = knn(data, centroids, k=1)
        assign = assign[:, 0]
        one_hot = (assign[:, None] == mx.arange(n_clusters)[None]).astype(mx.float32)  # (N, K)
        counts = mx.sum(one_hot, axis=0)
        sums = one_hot.T @ data
        # Empty clusters keep their previous centroid.
        centroids = mx.where(counts[:, None] > 0, sums / mx.maximum(counts, 1)[:, None], centroids)
        mx.eval(centroids)
    return np.array(centroids)
