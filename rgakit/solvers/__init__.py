"""
solvers
-------
Fitting solvers for SpectraLibrary.

Each solver is a callable ``(A, y) -> (weights, residual)`` where:

  A        : (n_channels, n_compounds)  design matrix (columns normalised to [0, 1])
  y        : (n_channels,)              observed spectrum (normalised)
  weights  : (n_compounds,)             non-negative compound weights
  residual : float                      ||A @ weights - y||_2

Supported methods: nnls, lasso, elastic_net, omp, romp.
"""

from __future__ import annotations

SUPPORTED_METHODS: tuple[str, ...] = ("nnls", "lasso", "elastic_net", "omp", "romp")


def make_solver(
    method:          str            = "nnls",
    alpha:           float          = 0.1,
    l1_ratio:        float          = 0.7,
    n_compounds:     int | None     = None,
    min_improvement: float          = 0.005,
    prune_tolerance: float          = 0.01,
    n_trials:        int            = 1,
    temperature:     float          = 0.3,
):
    """
    Return a callable ``(A, y) -> (weights, residual)`` for the requested solver.

    Parameters
    ----------
    method          : ``"nnls"`` | ``"lasso"`` | ``"elastic_net"`` | ``"omp"``
                      | ``"romp"``
    alpha           : Relative regularisation strength for LASSO / Elastic Net.
    l1_ratio        : L1 vs L2 balance for Elastic Net (1.0 = LASSO, default 0.7).
    n_compounds     : Max compounds for OMP / ROMP.  None = auto.
    min_improvement : Min relative residual reduction for OMP / ROMP (default 0.5%).
    prune_tolerance : Max relative residual increase to accept a removal
                      during ROMP backward pruning (default 1%).
    n_trials        : Independent OMP / ROMP trajectories (default 1 = greedy).
    temperature     : Softmax temperature for stochastic OMP / ROMP (default 0.3).

    Returns
    -------
    callable  ``(A, y) -> (weights, residual)``
    """
    if method == "nnls":
        from .nnls import make_nnls
        return make_nnls()

    if method == "lasso":
        from .lasso import make_lasso
        return make_lasso(alpha)

    if method == "elastic_net":
        from .elastic_net import make_elastic_net
        return make_elastic_net(alpha, l1_ratio)

    if method == "omp":
        from .omp import make_omp
        return make_omp(n_compounds, min_improvement, n_trials, temperature)

    if method == "romp":
        from .romp import make_romp
        return make_romp(n_compounds, min_improvement, prune_tolerance,
                         n_trials=n_trials, temperature=temperature)

    raise ValueError(
        f"Unknown fitting method {method!r}. "
        f"Supported: {', '.join(SUPPORTED_METHODS)}."
    )
