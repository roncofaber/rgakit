"""
romp.py
-------
Refined Orthogonal Matching Pursuit (ROMP).

Three-phase solver:

  1. **Forward selection** — standard OMP: greedily add the compound most
     correlated with the current residual, re-fit with NNLS on the active set.
  2. **Backward pruning** — try removing each active compound one at a time;
     drop it if the residual increase is below *prune_tolerance* (removes
     compounds that were useful early but became redundant after later picks).
  3. **Swap refinement** — try replacing each active compound with each
     inactive compound; accept the swap if it lowers the residual.

Phases 2–3 repeat until no further changes occur.

Each step is a single NNLS solve, so the refinement is fast even with
hundreds of library compounds.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def make_romp(
    n_compounds:     int | None = None,
    min_improvement: float      = 0.005,
    prune_tolerance: float      = 0.01,
    max_refine_iter: int        = 5,
    n_trials:        int        = 1,
    temperature:     float      = 0.3,
):
    """
    Return a ROMP solver callable ``(A, y) -> (weights, residual)``.

    Parameters
    ----------
    n_compounds     : Max compounds for forward selection.  None = auto.
    min_improvement : OMP stopping threshold (default 0.5%).
    prune_tolerance : Max relative residual increase to accept a removal
                      during backward pruning (default 1%).
    max_refine_iter : Max prune+swap cycles (default 5).
    n_trials        : Stochastic OMP trials for the forward phase (default 1).
    temperature     : Softmax temperature for stochastic trials (default 0.3).
    """
    from .omp import make_omp

    omp_solve = make_omp(n_compounds, min_improvement, n_trials, temperature)

    def _romp(A, y):
        n_comp = A.shape[1]

        # Phase 1: forward selection via OMP
        w_init, _ = omp_solve(A, y)
        active = [j for j in range(n_comp) if w_init[j] > 0]
        if not active:
            return w_init, float(np.linalg.norm(y))

        w_act, _ = nnls(A[:, active], y)
        res_best = float(np.linalg.norm(A[:, active] @ w_act - y))

        for _cycle in range(max_refine_iter):
            changed = False

            # Phase 2: backward pruning
            i = 0
            while i < len(active):
                if len(active) <= 1:
                    break
                trial = active[:i] + active[i+1:]
                w_t, _ = nnls(A[:, trial], y)
                res_t  = float(np.linalg.norm(A[:, trial] @ w_t - y))
                if res_t <= res_best * (1 + prune_tolerance):
                    active   = trial
                    w_act    = w_t
                    res_best = res_t
                    changed  = True
                else:
                    i += 1

            # Phase 3: swap refinement
            inactive = sorted(set(range(n_comp)) - set(active))
            for i, j_out in enumerate(list(active)):
                best_swap    = None
                best_swap_r  = res_best
                best_swap_w  = None
                for j_in in inactive:
                    trial = list(active)
                    trial[i] = j_in
                    w_t, _ = nnls(A[:, trial], y)
                    res_t  = float(np.linalg.norm(A[:, trial] @ w_t - y))
                    if res_t < best_swap_r:
                        best_swap   = j_in
                        best_swap_r = res_t
                        best_swap_w = w_t
                if best_swap is not None:
                    inactive.remove(best_swap)
                    inactive.append(j_out)
                    active[i] = best_swap
                    w_act    = best_swap_w
                    res_best = best_swap_r
                    changed  = True

            if not changed:
                break

        w_full = np.zeros(n_comp)
        for i, j in enumerate(active):
            w_full[j] = w_act[i]

        return w_full, float(np.linalg.norm(A @ w_full - y))

    return _romp
