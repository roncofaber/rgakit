"""
background.py
-------------
Standalone per-channel background correction for RGA pressure data.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def background_correct(
    time:           np.ndarray,
    pressure:       np.ndarray,
    shutter:        np.ndarray,
    shutter_time:   np.ndarray,
    window:         float = 30.0,
    gap_before:     float = 5.0,
    gap_after:      float = 10.0,
    method:         str = "linear",
    order:          int | None = None,
    skip_first:     int = 0,
    channel_offset: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """
    Per-channel polynomial background subtraction on RGA pressure data.

    Methods
    -------
    "linear" : straight line through two beam-off windows, before shutter open
               and after shutter close.
    "pre"    : polynomial (default order 0, a constant) through the pre-open
               window only.  Use when the signal decays slowly after the
               shutter closes, which the "linear" post-close window would
               otherwise absorb into the baseline.

    Windows
    -------
    Before : [open_time  - gap_before - window,  open_time  - gap_before]
    After  : [close_time + gap_after,             close_time + gap_after + window]

    Parameters
    ----------
    time           : (n_times,)       RGA time axis (s), start of each scan
    pressure       : (n_times, n_mz)  raw partial pressures (Torr)
    shutter        : (n_tey,)         binary shutter signal (0=closed, 1=open)
    shutter_time   : (n_tey,)         TEY time axis aligned with *shutter* (s)
    window         : duration (s) of each background window
    gap_before     : gap (s) between end of pre-shutter window and shutter open
    gap_after      : gap (s) between shutter close and start of post-shutter window
    method         : "linear" or "pre"
    order          : polynomial order of the baseline; defaults to 1 for
                     "linear" and 0 for "pre"
    skip_first     : number of leading scans never used as background
    channel_offset : (n_mz,) time (s) at which each channel is sampled
                     relative to the scan start; windows are applied to
                     ``time + channel_offset`` per channel

    Returns
    -------
    corrected  : np.ndarray shape (n_times, n_mz) — background-subtracted pressures
    open_time  : float — shutter open time (s)
    close_time : float — shutter close time (s)
    """
    if method not in ("linear", "pre"):
        raise ValueError(f"Unknown method {method!r}; use 'linear' or 'pre'.")
    if order is None:
        order = 1 if method == "linear" else 0

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

    n_times, n_mz = pressure.shape
    offset = np.zeros(n_mz) if channel_offset is None else np.asarray(channel_offset, dtype=float)
    tc     = time[:, None] + offset[None, :]
    usable = np.arange(n_times)[:, None] >= skip_first

    off1_end   = open_time  - gap_before
    off1_start = off1_end   - window
    off1_mask  = (tc >= off1_start) & (tc <= off1_end) & usable

    off2_start = close_time + gap_after
    off2_end   = off2_start + window
    if method == "linear":
        off2_mask = (tc >= off2_start) & (tc <= off2_end) & usable
    else:
        off2_mask = np.zeros_like(off1_mask)

    n1, n2 = off1_mask.sum(axis=0).min(), off2_mask.sum(axis=0).min()
    logger.debug(
        "Background windows: pre-shutter>=%d scans [%.1f, %.1f] s, "
        "post-shutter>=%d scans [%.1f, %.1f] s",
        n1, off1_start, off1_end, n2, off2_start, off2_end,
    )

    if n1 + n2 < order + 1:
        raise ValueError(
            f"Not enough background points (before={n1}, after={n2}) for order {order}. "
            "Try increasing 'window' or reducing 'gap_before'/'gap_after'."
        )
    if method == "linear" and n1 == 0:
        logger.warning(
            "No RGA scans in pre-shutter window [%.1f, %.1f] s — using post-close only.",
            off1_start, off1_end,
        )
    if method == "linear" and n2 == 0:
        logger.warning(
            "No RGA scans in post-shutter window [%.1f, %.1f] s — using pre-open only.",
            off2_start, off2_end,
        )

    bg_mask   = off1_mask | off2_mask
    corrected = np.empty_like(pressure, dtype=float)
    for mz_idx in range(n_mz):
        col    = pressure[:, mz_idx].astype(float)
        m      = bg_mask[:, mz_idx]
        coeffs = np.polyfit(tc[m, mz_idx], col[m], order)
        corrected[:, mz_idx] = col - np.polyval(coeffs, tc[:, mz_idx])

    logger.info(
        "Background correction applied (%s, order %d): %d m/z channels, window=%.0f s, "
        "gap_before=%.0f s, gap_after=%.0f s",
        method, order, n_mz, window, gap_before, gap_after,
    )
    return corrected, open_time, close_time
