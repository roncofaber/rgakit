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


def _resolve_shutter_window(
    shutter: np.ndarray | None,
    shutter_time: np.ndarray | None,
    open_time: float | None,
    close_time: float | None,
) -> tuple[float, float]:
    if open_time is not None and close_time is not None:
        return float(open_time), float(close_time)
    if shutter is None or shutter_time is None:
        raise ValueError(
            "Either (open_time, close_time) or (shutter, shutter_time) must be provided."
        )
    edges = np.diff(np.asarray(shutter, dtype=int))
    open_idx = np.where(edges > 0)[0]
    close_idx = np.where(edges < 0)[0]
    if len(open_idx) == 0 or len(close_idx) == 0:
        raise ValueError("Could not detect shutter open/close edges.")
    st = np.asarray(shutter_time, dtype=float)
    return float(st[open_idx[0] + 1]), float(st[close_idx[0]])


def extract_run_signals(
    time: np.ndarray,
    pressure: np.ndarray,
    shutter: np.ndarray | None = None,
    shutter_time: np.ndarray | None = None,
    open_time: float | None = None,
    close_time: float | None = None,
    window: float | None = None,
    gap_before: float = 0.5,
    settle: float = 0.0,
    duration: float | None = None,
    skip_first: int = 1,
    channel_offset: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Extract per-channel pre-exposure baseline, noise, beam-on rise, standard
    error, and z-score from a raw RGA time series.

    Windows
    -------
    Pre-exposure : by default (``window=None``) every scan from *skip_first*
                   up to ``open_time - gap_before``; a numeric *window*
                   limits it to that many seconds ending at the same point.
    Beam-on      : ``[open_time + settle, close_time]``, or
                   ``[open_time + settle, open_time + duration]`` when
                   *duration* caps the window sooner than *close_time*.

    Standard error
    --------------
    ``noise`` is each channel's pre-exposure sample standard deviation across
    its ``n_pre`` scans (``ddof=1``; NaN if ``n_pre < 2``). ``rise`` is the
    beam-on window mean minus the pre-exposure mean (``level``). The standard
    error of that difference assumes equal variance between the two windows,
    estimated from the pre-exposure noise alone::

        se = noise * sqrt(1 / n_on + 1 / n_pre)
        z  = rise / se

    ``noise`` comes from only ``n_pre`` scans, so ``z`` follows a
    t distribution with ``n_pre - 1`` degrees of freedom, not a standard
    normal, so a fixed cut on ``z`` is not a calibrated detection threshold
    when ``n_pre`` is small.

    ``channel_offset`` (s), when given, shifts each channel's effective
    sample time (``time + channel_offset[j]``) before the window masks are
    applied, so scans are assigned to the pre-exposure/beam-on windows per
    channel rather than per scan. A channel with fewer than 1 pre-exposure
    scan or fewer than 1 beam-on scan gets NaN in every derived field rather
    than a silently biased value; there is no implicit floor on ``n_pre``.

    Parameters
    ----------
    time           : (n_times,) scan start timestamps (s)
    pressure       : (n_times, n_mz) raw partial pressures (Torr)
    shutter        : (n_tey,) binary shutter trace (0=closed, 1=open)
    shutter_time   : (n_tey,) TEY/shutter time axis (s)
    open_time      : explicit shutter open time (s); overrides edge detection
    close_time     : explicit shutter close time (s); overrides edge detection
    window         : pre-exposure baseline window duration (s); None (default)
                     uses every scan from skip_first up to open_time - gap_before
    gap_before     : guard gap before shutter open (s)
    settle         : settling delay after shutter open (s)
    duration       : optional cap on beam-on window length from open_time (s)
    skip_first     : number of leading scans excluded from the pre-window
    channel_offset : (n_mz,) intra-scan sampling offset per channel (s)

    Returns
    -------
    dict with keys ``"level"``, ``"noise"``, ``"rise"``, ``"se"``, ``"z"``,
    ``"n_pre"``, ``"n_on"``, ``"open_time"``, and ``"close_time"``.
    """
    ot, ct = _resolve_shutter_window(shutter, shutter_time, open_time, close_time)
    time_arr = np.asarray(time, dtype=float)
    press_arr = np.asarray(pressure, dtype=float)
    n_times, n_mz = press_arr.shape

    offset = (
        np.zeros(n_mz, dtype=float)
        if channel_offset is None
        else np.asarray(channel_offset, dtype=float)
    )
    tc = time_arr[:, None] + offset[None, :]
    usable = np.arange(n_times)[:, None] >= skip_first

    pre_end = ot - gap_before
    pre_start = -np.inf if window is None else pre_end - window
    pre_mask = (tc >= pre_start) & (tc <= pre_end) & usable

    on_start = ot + settle
    on_end = ct if duration is None else min(ct, ot + duration)
    on_mask = (tc >= on_start) & (tc <= on_end)

    level = np.full(n_mz, np.nan, dtype=float)
    noise = np.full(n_mz, np.nan, dtype=float)
    rise = np.full(n_mz, np.nan, dtype=float)
    se = np.full(n_mz, np.nan, dtype=float)
    z = np.full(n_mz, np.nan, dtype=float)
    n_pre = np.zeros(n_mz, dtype=int)
    n_on = np.zeros(n_mz, dtype=int)

    for j in range(n_mz):
        col = press_arr[:, j]
        m_pre = pre_mask[:, j] & np.isfinite(col)
        m_on = on_mask[:, j] & np.isfinite(col)
        np_j = int(m_pre.sum())
        no_j = int(m_on.sum())
        n_pre[j] = np_j
        n_on[j] = no_j
        if np_j >= 1:
            level[j] = float(col[m_pre].mean())
        if np_j >= 2:
            noise[j] = float(col[m_pre].std(ddof=1))
        if no_j >= 1 and np_j >= 1:
            rise[j] = float(col[m_on].mean()) - level[j]
        if no_j >= 1 and np_j >= 1 and np.isfinite(noise[j]) and noise[j] > 0:
            se[j] = noise[j] * np.sqrt(1.0 / no_j + 1.0 / np_j)
            z[j] = rise[j] / se[j]

    return {
        "level": level,
        "noise": noise,
        "rise": rise,
        "se": se,
        "z": z,
        "n_pre": n_pre,
        "n_on": n_on,
        "open_time": ot,
        "close_time": ct,
    }
