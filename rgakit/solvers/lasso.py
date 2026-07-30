"""
lasso.py
--------
Non-negative LASSO solver (requires scikit-learn).

The user-facing ``alpha`` is a relative value in (0, 1).  Internally it is
multiplied by ``alpha_max = max(A.T @ y) / n_channels``, which is the exact
threshold above which all weights become zero.  This makes ``alpha``
meaningful regardless of the number of m/z channels or signal scale.
"""

from __future__ import annotations

import numpy as np


def make_lasso(alpha: float = 0.1):
    """
    Return a LASSO solver callable ``(A, y) -> (weights, residual)``.

    The returned callable is **stateful**: a single ``Lasso`` model instance
    is shared across calls with ``warm_start=True``.
    """
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        raise ImportError(
            "scikit-learn is required for method='lasso': "
            "pip install scikit-learn"
        )

    model = Lasso(
        alpha=1.0,
        positive=True,
        fit_intercept=False,
        max_iter=10_000,
        warm_start=True,
    )

    def _lasso(A, y):
        alpha_max = float(np.max(A.T @ y)) / len(y)
        if alpha_max <= 0:
            return np.zeros(A.shape[1]), float(np.linalg.norm(y))
        model.alpha = alpha * alpha_max
        model.fit(A, y)
        w = np.maximum(model.coef_, 0.0)
        return w, float(np.linalg.norm(A @ w - y))

    return _lasso
