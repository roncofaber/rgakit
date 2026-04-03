"""
similarity.py
-------------
Spectral similarity metrics for EI mass spectra.

All functions operate on MassSpectrum objects and return a float in [0, 1]
(higher = more similar).

Public API
----------
cosine(a, b)
    Dot-product cosine on shared m/z grid.

jaccard(a, b, threshold=0.05)
    Fraction of peaks shared above an intensity threshold.

pearson(a, b)
    Pearson correlation on the shared m/z grid.

entropy(a, b)
    Spectral entropy similarity (Li et al. 2021, Nature Methods).
    More robust than cosine when spectra are noisy.

score(a, b, method="cosine", **kw)
    Dispatcher: call any metric by name.

pairwise(spectra, method="cosine", **kw)
    Compute the full N×N similarity matrix for a list (or library) of spectra.
    Returns (matrix, names).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _on_shared_grid(
    a: "MassSpectrum", b: "MassSpectrum"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (shared_mz, a_vec, b_vec) projected onto the shared m/z grid."""
    shared = np.intersect1d(a.mz, b.mz)
    return shared, a.on_grid(shared), b.on_grid(shared)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability distribution p (already normalised)."""
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cosine(a: "MassSpectrum", b: "MassSpectrum") -> float:
    """
    Dot-product cosine similarity on the shared m/z grid.

    Both spectra are normalised to unit vectors before scoring.
    Returns 0 when there is no m/z overlap.
    """
    shared, av, bv = _on_shared_grid(a, b)
    if len(shared) == 0:
        return 0.0
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def jaccard(
    a: "MassSpectrum",
    b: "MassSpectrum",
    threshold: float = 0.05,
) -> float:
    """
    Jaccard index: fraction of m/z peaks shared above *threshold*.

    Both spectra are normalised to [0, 1] before applying the threshold.
    Measures peak-presence overlap independent of intensity.

    Parameters
    ----------
    threshold : minimum normalised intensity to count a peak as 'present'
    """
    a_peaks = set(a.mz[a.normalized >= threshold].tolist())
    b_peaks = set(b.mz[b.normalized >= threshold].tolist())
    union = a_peaks | b_peaks
    if not union:
        return 0.0
    return len(a_peaks & b_peaks) / len(union)


def pearson(a: "MassSpectrum", b: "MassSpectrum") -> float:
    """
    Pearson correlation coefficient on the shared m/z grid, mapped to [0, 1].

    Values below zero are clipped to 0 (anti-correlated spectra are treated as
    completely dissimilar for classification purposes).
    """
    shared, av, bv = _on_shared_grid(a, b)
    if len(shared) < 2:
        return 0.0
    av_c = av - av.mean()
    bv_c = bv - bv.mean()
    denom = np.linalg.norm(av_c) * np.linalg.norm(bv_c)
    r = float(np.dot(av_c, bv_c) / denom) if denom > 0 else 0.0
    return max(0.0, r)


def entropy(a: "MassSpectrum", b: "MassSpectrum") -> float:
    """
    Spectral entropy similarity (Li et al. 2021, Nature Methods 18:1876).

    Defined as: 1 - (2·H(merged) - H(a) - H(b)) / log(4)
    where H is Shannon entropy and 'merged' is the mean of both spectra
    projected on the union grid.

    Returns a value in [0, 1].  More robust than cosine for noisy spectra
    because it down-weights dominant peaks.
    """
    union = np.union1d(a.mz, b.mz)
    av = a.on_grid(union)
    bv = b.on_grid(union)

    # Normalise to probability distributions
    av_sum, bv_sum = av.sum(), bv.sum()
    if av_sum == 0 or bv_sum == 0:
        return 0.0
    av_p = av / av_sum
    bv_p = bv / bv_sum

    merged_p = (av_p + bv_p) / 2.0

    h_a      = _entropy(av_p)
    h_b      = _entropy(bv_p)
    h_merged = _entropy(merged_p)

    sim = 1.0 - (2.0 * h_merged - h_a - h_b) / np.log(4.0)
    return float(np.clip(sim, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_METRICS: dict[str, callable] = {
    "cosine":  cosine,
    "jaccard": jaccard,
    "pearson": pearson,
    "entropy": entropy,
}


def score(
    a: "MassSpectrum",
    b: "MassSpectrum",
    method: str = "cosine",
    **kwargs,
) -> float:
    """
    Compute the similarity between two spectra using *method*.

    Parameters
    ----------
    method : one of ``"cosine"``, ``"jaccard"``, ``"pearson"``, ``"entropy"``
    **kwargs : passed to the underlying metric (e.g. ``threshold`` for jaccard)
    """
    fn = _METRICS.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {list(_METRICS)}"
        )
    return fn(a, b, **kwargs)


# ---------------------------------------------------------------------------
# Pairwise matrix
# ---------------------------------------------------------------------------

def pairwise(
    spectra,
    method: str = "cosine",
    **kwargs,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute the N×N pairwise similarity matrix for a collection of spectra.

    Parameters
    ----------
    spectra : iterable of MassSpectrum, or a SpectraLibrary
    method  : similarity metric (``"cosine"``, ``"jaccard"``, ``"pearson"``,
              ``"entropy"``)
    **kwargs : passed to the underlying metric

    Returns
    -------
    matrix : np.ndarray, shape (N, N), symmetric, diagonal = 1
    names  : list[str] — compound names in the same order as matrix rows/cols

    Example
    -------
    >>> mat, names = pairwise(lib, method="cosine")
    >>> import pandas as pd
    >>> pd.DataFrame(mat, index=names, columns=names).round(3)
    """
    spec_list = list(spectra)   # works for SpectraLibrary (iter yields MassSpectrum)
    names = [s.name for s in spec_list]
    n = len(spec_list)
    fn = _METRICS.get(method)
    if fn is None:
        raise ValueError(f"Unknown method {method!r}. Choose from: {list(_METRICS)}")

    mat = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            s = fn(spec_list[i], spec_list[j], **kwargs)
            mat[i, j] = s
            mat[j, i] = s

    return mat, names
