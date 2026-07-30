"""
omp.py
------
Non-Negative Orthogonal Matching Pursuit with optional stochastic trials.

Greedy forward selection with NNLS on the active set.  Naturally sparse —
no regularisation hyperparameter needed.

When ``n_trials > 1``, multiple independent trajectories are run: the first
is always deterministic (standard greedy OMP), subsequent trials sample from
a softmax over correlations controlled by ``temperature``.  The trajectory
with the lowest residual wins.

Stopping criteria
-----------------
Selection stops when any of the following is true:

  - The best remaining correlation with the residual is <= 0.
  - The newly added compound receives weight 0 from NNLS (redundant given
    the already-selected set) -> it is removed and selection stops.
  - The relative improvement in ||residual|| is below *min_improvement*.
  - *n_compounds* compounds have been selected (when set explicitly).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def make_omp(
    n_compounds:     int | None = None,
    min_improvement: float      = 0.005,
    n_trials:        int        = 1,
    temperature:     float      = 0.3,
):
    """
    Return an OMP solver callable ``(A, y) -> (weights, residual)``.

    Parameters
    ----------
    n_compounds     : Maximum compounds to select.  None = stop on convergence.
    min_improvement : Minimum relative residual reduction to keep adding (0.5%).
    n_trials        : Number of independent trajectories (default 1 = greedy).
    temperature     : Softmax temperature for stochastic trials (default 0.3).
    """
    rng = np.random.default_rng()

    def _omp_single(A, y, stochastic: bool = False):
        n_comp    = A.shape[1]
        max_k     = min(n_compounds, n_comp) if n_compounds else n_comp
        active    = []
        remaining = list(range(n_comp))
        residual  = y.copy()
        prev_norm = float(np.linalg.norm(y)) or 1.0
        w_full    = np.zeros(n_comp)

        for _ in range(max_k):
            if not remaining:
                break

            cors = A[:, remaining].T @ residual
            if stochastic and temperature > 0:
                pos_mask = cors > 0
                if not pos_mask.any():
                    break
                logits = np.where(pos_mask, cors, -np.inf)
                logits = logits - logits.max()
                probs  = np.exp(logits / temperature)
                probs /= probs.sum()
                pick   = int(rng.choice(len(remaining), p=probs))
            else:
                pick = int(np.argmax(cors))
                if float(cors[pick]) <= 0:
                    break

            best_j = remaining[pick]
            active.append(best_j)
            remaining.remove(best_j)

            w_act, _ = nnls(A[:, active], y)

            if w_act[-1] <= 0:
                active.pop()
                break

            new_norm = float(np.linalg.norm(y - A[:, active] @ w_act))
            improvement = (prev_norm - new_norm) / prev_norm
            if improvement < min_improvement:
                active.pop()
                break
            prev_norm = new_norm
            residual  = y - A[:, active] @ w_act

        if active:
            w_act, _ = nnls(A[:, active], y)
            for i, j in enumerate(active):
                w_full[j] = w_act[i]

        return w_full, float(np.linalg.norm(A @ w_full - y))

    def _omp(A, y):
        best_w, best_res = _omp_single(A, y, stochastic=False)

        for _ in range(n_trials - 1):
            w, res = _omp_single(A, y, stochastic=True)
            if res < best_res:
                best_w, best_res = w, res

        return best_w, best_res

    return _omp
