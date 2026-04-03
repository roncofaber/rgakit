"""
background.py
-------------
Per-channel linear background subtraction for RGA pressure data.
"""

from __future__ import annotations

import warnings

import numpy as np


def background_correct(
    shutter,
    shutter_time,
    exp_time,
    pressure,
    margin_before: float = 5.0,
    margin_after:  float = 5.0,
):
    """
    Per-channel linear background subtraction on RGA pressure data.

    Fits a straight line through two beam-off windows (before open and after
    close), then subtracts the extrapolated line from every m/z channel.
    A margin is excluded around each shutter edge to avoid transient artefacts.

    Parameters
    ----------
    shutter       : (n_tey,)         binary shutter signal (0=closed, 1=open)
    shutter_time  : (n_tey,)         TEY time axis (s)
    exp_time      : (n_times,)       RGA time axis (s)
    pressure      : (n_times, n_mz)  raw partial pressures (Torr)
    margin_before : seconds to exclude before shutter-open edge
    margin_after  : seconds to exclude after shutter-close edge

    Returns
    -------
    corrected  : (n_times, n_mz)  background-corrected pressure
    open_time  : float  detected shutter-open time (s)
    close_time : float  detected shutter-close time (s)
    """
    edges     = np.diff(shutter.astype(int))
    open_idx  = np.where(edges > 0)[0]
    close_idx = np.where(edges < 0)[0]

    if len(open_idx) == 0 or len(close_idx) == 0:
        raise ValueError("Could not detect shutter open/close edges.")

    open_time  = shutter_time[open_idx[0]  + 1]   # first open sample
    close_time = shutter_time[close_idx[0]]        # last open sample (edge is between [i] and [i+1])

    # beam-off windows (skip first scan as a transient guard)
    off1_mask = (exp_time >= exp_time[1]) & (exp_time <= open_time  - margin_before)
    off2_mask = (exp_time >= close_time  + margin_after)

    n1, n2 = off1_mask.sum(), off2_mask.sum()
    if n1 + n2 < 2:
        raise ValueError(
            f"Not enough background points for linear fit "
            f"(before={n1}, after={n2}). "
            f"Try reducing margin_before/margin_after."
        )
    if n1 == 0:
        warnings.warn(
            "No beam-off points before shutter open — using post-close window only.",
            stacklevel=2,
        )
    if n2 == 0:
        warnings.warn(
            "No beam-off points after shutter close — using pre-open window only.",
            stacklevel=2,
        )

    bg_mask = off1_mask | off2_mask
    x_bg    = exp_time[bg_mask]

    corrected = np.zeros_like(pressure)
    for mz_idx in range(pressure.shape[1]):
        col        = pressure[:, mz_idx]
        coeffs     = np.polyfit(x_bg, col[bg_mask], 1)
        corrected[:, mz_idx] = col - np.polyval(coeffs, exp_time)

    return corrected, open_time, close_time
