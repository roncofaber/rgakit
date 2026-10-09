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
  3. **Swap refinement** — try replacing each active compound with an inactive
     compound; accept the swap if it lowers the residual. Swap candidates are
     pre-screened: only compounds positively correlated with the current
     residual are considered, and candidates are ranked by the unconstrained
     least-squares residual (a cheap lower bound of the NNLS residual), with
     only the best few verified by NNLS.

Phases 2–3 repeat until no further changes occur.

Each accepted step is a single NNLS solve, so the refinement is fast even
with hundreds of library compounds.
"""

from __future__ import annotations

import numpy as np

from .nnls import solve_nnls

_SWAP_SCREEN = 50
_SWAP_VERIFY = 3


def make_romp(
    n_compounds:     int | None = None,
    min_improvement: float      = 0.005,
    prune_tolerance: float      = 0.01,
    max_refine_iter: int        = 5,
    n_trials:        int        = 1,
    temperature:     float      = 0.3,
    random_state:    int | None = None,
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
    temperature     : Softmax temperature for stochastic OMP (default 0.3).
    random_state    : Seed for the stochastic forward phase.  None = fresh
                      entropy each call; set for reproducible fits.
    """
    from .omp import make_omp

    omp_solve = make_omp(n_compounds, min_improvement, n_trials, temperature,
                         random_state=random_state)

    def _romp(A, y):
        n_comp = A.shape[1]

        # Phase 1: forward selection via OMP
        w_init, _ = omp_solve(A, y)
        active = [j for j in range(n_comp) if w_init[j] > 0]
        if not active:
            return w_init, float(np.linalg.norm(y))

        w_act, _ = solve_nnls(A[:, active], y)
        res_best = float(np.linalg.norm(A[:, active] @ w_act - y))

        for _cycle in range(max_refine_iter):
            changed = False

            # Phase 2: backward pruning
            i = 0
            while i < len(active):
                if len(active) <= 1:
                    break
                trial = active[:i] + active[i+1:]
                w_t, _ = solve_nnls(A[:, trial], y)
                res_t  = float(np.linalg.norm(A[:, trial] @ w_t - y))
                if res_t <= res_best * (1 + prune_tolerance):
                    active   = trial
                    w_act    = w_t
                    res_best = res_t
                    changed  = True
                else:
                    i += 1

            # Phase 3: swap refinement (pre-screened)
            if len(active) > 1:
                resid = y - A[:, active] @ w_act
                inactive = sorted(set(range(n_comp)) - set(active))
                cors = A[:, inactive].T @ resid if inactive else np.zeros(0)
                order = np.argsort(-cors)
                screened = [inactive[t] for t in order[:_SWAP_SCREEN]
                            if cors[t] > 0]

                for i, j_out in enumerate(list(active)):
                    ranked = []
                    for j_in in screened:
                        cols = list(active)
                        cols[i] = j_in
                        Aw = A[:, cols]
                        w_u, *_ = np.linalg.lstsq(Aw, y, rcond=None)
                        res_u = float(np.linalg.norm(Aw @ w_u - y))
                        ranked.append((res_u, j_in))
                    ranked.sort(key=lambda t: t[0])

                    best_swap   = None
                    best_swap_r = res_best
                    best_swap_w = None
                    for _, j_in in ranked[:_SWAP_VERIFY]:
                        trial = list(active)
                        trial[i] = j_in
                        w_t, _ = solve_nnls(A[:, trial], y)
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
