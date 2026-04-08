"""
background.py
-------------
Standalone linear background correction for RGA pressure data.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def background_correct(
    time:         np.ndarray,
    pressure:     np.ndarray,
    shutter:      np.ndarray,
    shutter_time: np.ndarray,
    window:       float = 30.0,
    gap_before:   float = 5.0,
    gap_after:    float = 10.0,
) -> tuple[np.ndarray, float, float]:
    """
    Per-channel linear background subtraction on RGA pressure data.

    Fits a straight line through two beam-off windows (before shutter open and
    after shutter close), then subtracts the extrapolated baseline from every
    m/z channel independently.

    Windows
    -------
    Before : [open_time  - gap_before - window,  open_time  - gap_before]
    After  : [close_time + gap_after,             close_time + gap_after + window]

    Parameters
    ----------
    time         : (n_times,)       RGA time axis (s)
    pressure     : (n_times, n_mz)  raw partial pressures (Torr)
    shutter      : (n_tey,)         binary shutter signal (0=closed, 1=open)
    shutter_time : (n_tey,)         TEY time axis aligned with *shutter* (s)
    window       : duration (s) of each background window
    gap_before   : gap (s) between end of pre-shutter window and shutter open
    gap_after    : gap (s) between shutter close and start of post-shutter window

    Returns
    -------
    corrected  : np.ndarray shape (n_times, n_mz) — background-subtracted pressures
    open_time  : float — shutter open time (s)
    close_time : float — shutter close time (s)
    """
    edges     = np.diff(shutter.astype(int))
    open_idx  = np.where(edges > 0)[0]
    close_idx = np.where(edges < 0)[0]
    if len(open_idx) == 0 or len(close_idx) == 0:
        raise ValueError("Could not detect shutter open/close edges.")

    open_time  = shutter_time[open_idx[0] + 1]
    close_time = shutter_time[close_idx[0]]
    logger.debug(
        "Shutter window: open=%.2f s, close=%.2f s (duration=%.1f s)",
        open_time, close_time, close_time - open_time,
    )

    off1_end   = open_time  - gap_before
    off1_start = off1_end   - window
    off1_mask  = (time >= off1_start) & (time <= off1_end)

    off2_start = close_time + gap_after
    off2_end   = off2_start + window
    off2_mask  = (time >= off2_start) & (time <= off2_end)

    n1, n2 = off1_mask.sum(), off2_mask.sum()
    logger.debug(
        "Background windows: pre-shutter=%d scans [%.1f, %.1f] s, "
        "post-shutter=%d scans [%.1f, %.1f] s",
        n1, off1_start, off1_end, n2, off2_start, off2_end,
    )

    if n1 + n2 < 2:
        raise ValueError(
            f"Not enough background points (before={n1}, after={n2}). "
            "Try increasing 'window' or reducing 'gap_before'/'gap_after'."
        )
    if n1 == 0:
        logger.warning(
            "No RGA scans in pre-shutter window [%.1f, %.1f] s — using post-close only.",
            off1_start, off1_end,
        )
    if n2 == 0:
        logger.warning(
            "No RGA scans in post-shutter window [%.1f, %.1f] s — using pre-open only.",
            off2_start, off2_end,
        )

    bg_mask   = off1_mask | off2_mask
    x_bg      = time[bg_mask]
    corrected = np.empty_like(pressure, dtype=float)
    n_mz      = pressure.shape[1]
    for mz_idx in range(n_mz):
        col = pressure[:, mz_idx].astype(float)
        coeffs = np.polyfit(x_bg, col[bg_mask], 1)
        corrected[:, mz_idx] = col - np.polyval(coeffs, time)

    logger.info(
        "Background correction applied: %d m/z channels, window=%.0f s, "
        "gap_before=%.0f s, gap_after=%.0f s",
        n_mz, window, gap_before, gap_after,
    )
    return corrected, open_time, close_time
