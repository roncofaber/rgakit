"""
elastic_net.py
--------------
Non-negative Elastic Net solver (requires scikit-learn).

Combines L1 (sparsity) and L2 (grouping) penalties.  The L2 term handles
collinear compounds gracefully: instead of arbitrarily picking one of two
near-identical spectra, it shares weight between them proportionally.

Parameters
----------
alpha    : overall regularisation strength, relative to alpha_max (0–1).
l1_ratio : balance between L1 and L2.  1.0 = pure LASSO, 0.5 = equal mix,
           0.0 = pure ridge.  Default 0.7 gives good sparsity with some
           collinearity tolerance.
"""

from __future__ import annotations

import numpy as np


def make_elastic_net(alpha: float = 0.1, l1_ratio: float = 0.7):
    """
    Return an Elastic Net solver callable ``(A, y) -> (weights, residual)``.

    The returned callable is **stateful**: a single ``ElasticNet`` model
    instance is shared across calls with ``warm_start=True``.
    """
    try:
        from sklearn.linear_model import ElasticNet
    except ImportError:
        raise ImportError(
            "scikit-learn is required for method='elastic_net': "
            "pip install scikit-learn"
        )

    model = ElasticNet(
        alpha=1.0,
        l1_ratio=l1_ratio,
        positive=True,
        fit_intercept=False,
        max_iter=10_000,
        warm_start=True,
    )

    def _elastic_net(A, y):
        alpha_max = float(np.max(A.T @ y)) / len(y)
        if alpha_max <= 0:
            return np.zeros(A.shape[1]), float(np.linalg.norm(y))
        model.alpha = alpha * alpha_max
        model.fit(A, y)
        w = np.maximum(model.coef_, 0.0)
        return w, float(np.linalg.norm(A @ w - y))

    return _elastic_net
