"""
solvers.py
----------
Fitting solvers for SpectraLibrary.

Each solver is a callable ``(A, y) -> (weights, residual)`` where:

  A        : (n_channels, n_compounds)  design matrix (columns normalised to [0, 1])
  y        : (n_channels,)              observed spectrum (normalised)
  weights  : (n_compounds,)             non-negative compound weights
  residual : float                      ||A @ weights - y||_2

Supported methods
-----------------
nnls  : non-negative least squares (scipy, no extra dependencies)
lasso : non-negative LASSO — promotes sparse solutions via L1 regularisation
        (requires scikit-learn)

Alpha scaling for LASSO
-----------------------
The user-facing ``alpha`` is a **relative** value in (0, 1).  Internally it
is multiplied by ``alpha_max = max(A.T @ y) / n_channels``, which is the
exact threshold above which *all* weights become zero.  This makes ``alpha``
meaningful regardless of the number of m/z channels or signal scale:

  alpha ~ 0    dense solution (approaches NNLS)
  alpha = 0.1  10% of maximum regularisation — good starting point
  alpha = 0.5  aggressive sparsity
  alpha ~ 1    all weights driven to zero
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

SUPPORTED_METHODS: tuple[str, ...] = ("nnls", "lasso")


def make_solver(method: str, alpha: float = 0.1):
    """
    Return a callable ``(A, y) -> (weights, residual)`` for the requested solver.

    Parameters
    ----------
    method : ``"nnls"`` — non-negative least squares (scipy, no extra
             dependencies); ``"lasso"`` — non-negative LASSO which promotes a
             sparse solution by adding an L1 penalty (requires scikit-learn).
    alpha  : Relative regularisation strength for ``"lasso"``, in (0, 1).
             Scaled internally by ``alpha_max`` (the per-call L1 threshold that
             drives all weights to zero), so its meaning is independent of the
             data scale or number of m/z channels.  Larger values yield fewer
             non-zero compounds.  Good starting point: 0.1.  Ignored when
             ``method="nnls"``.

    Returns
    -------
    callable  ``(A, y) -> (weights, residual)``

    Raises
    ------
    ValueError  if *method* is not in :data:`SUPPORTED_METHODS`.
    ImportError if ``method="lasso"`` and scikit-learn is not installed.

    Notes
    -----
    The returned callable is **stateful** for ``"lasso"``: a single
    ``Lasso`` model instance is shared across calls with ``warm_start=True``,
    so each call reuses the previous solution as a starting point.  This is
    intentional — it makes repeated calls in a time-series loop significantly
    faster when consecutive spectra are similar.
    """
    if method == "nnls":
        return nnls

    if method == "lasso":
        try:
            from sklearn.linear_model import Lasso
        except ImportError:
            raise ImportError(
                "scikit-learn is required for method='lasso': "
                "pip install scikit-learn"
            )

        # alpha is set per-call after scaling; start with a placeholder.
        model = Lasso(
            alpha=1.0,            # overwritten each call after scaling
            positive=True,        # enforces w >= 0
            fit_intercept=False,
            max_iter=10_000,
            warm_start=True,      # efficient for time-series loops
        )

        def _lasso(A, y):
            # alpha_max is the minimum absolute alpha that zeros all weights
            # (KKT condition for positive Lasso at w=0).
            alpha_max = float(np.max(A.T @ y)) / len(y)
            if alpha_max <= 0:
                # No compound has positive correlation with the observed spectrum.
                return np.zeros(A.shape[1]), float(np.linalg.norm(y))
            model.alpha = alpha * alpha_max
            model.fit(A, y)
            w = np.maximum(model.coef_, 0.0)
            return w, float(np.linalg.norm(A @ w - y))

        return _lasso

    raise ValueError(
        f"Unknown fitting method {method!r}. "
        f"Supported: {', '.join(SUPPORTED_METHODS)}."
    )
