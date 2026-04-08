"""
stack.py
--------
SpectrumStack: a source-agnostic time-resolved mass spectrum dataset.

Public API
----------
SpectrumStack(time, pressure, mz=None, shutter=None, shutter_time=None,
              open_time=None, close_time=None)
    Direct construction from arrays.

SpectrumStack.from_rga(rga)
    Adapter for clabs RGAMeasurement objects.

stack.background_correct(window, gap_before, gap_after) -> SpectrumStack
    Returns a new background-corrected stack.

stack.averaged(time_range=None) -> MassSpectrum
    Average over the shutter-open window (or a custom range).

stack.slice(t_start, t_end) -> SpectrumStack
    Return a new stack restricted to [t_start, t_end].

stack.open_window -> (float, float)
    The (open_time, close_time) window.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SpectrumStack:
    """
    A time-ordered sequence of mass spectra from an RGA measurement.

    This is the canonical time-series container in rgakit.  It is
    source-agnostic: use ``from_rga`` to wrap a clabs ``RGAMeasurement``,
    or pass raw arrays directly.

    Parameters
    ----------
    time         : (n_times,) float — seconds since experiment start
    pressure     : (n_times, n_mz) float — signal (pressure or counts)
    mz           : (n_mz,) int, optional — m/z axis; defaults to 1..n_mz
    shutter      : (n_tey,) array-like, optional — raw binary shutter signal
                   (0 = closed, 1 = open) on the *shutter_time* axis
    shutter_time : (n_tey,) float, optional — time axis for *shutter*
                   (may differ from *time*; required when *shutter* is given)
    open_time    : float, optional — known shutter-open time (s)
    close_time   : float, optional — known shutter-close time (s)
    name         : str, optional — label for the measurement
    """

    def __init__(
        self,
        time:         np.ndarray,
        pressure:     np.ndarray,
        mz:           np.ndarray | None = None,
        shutter:      np.ndarray | None = None,
        shutter_time: np.ndarray | None = None,
        open_time:    float | None      = None,
        close_time:   float | None      = None,
        name:         str               = "",
    ):
        self.time     = np.asarray(time,     dtype=float)
        self.pressure = np.asarray(pressure, dtype=float)
        self.name     = name

        if mz is None:
            self.mz = np.arange(1, self.pressure.shape[1] + 1, dtype=int)
        else:
            self.mz = np.asarray(mz, dtype=int)

        self._shutter      = None if shutter is None else np.asarray(shutter)
        self._shutter_time = None if shutter_time is None else np.asarray(shutter_time, dtype=float)
        self._open_time    = open_time
        self._close_time   = close_time

        # Set after background_correct(); None on uncorrected stacks
        self._raw_pressure: np.ndarray | None = None
        self._bg_off1: tuple[float, float] | None = None   # (t_start, t_end) pre-shutter window
        self._bg_off2: tuple[float, float] | None = None   # (t_start, t_end) post-shutter window

        if self.pressure.ndim != 2:
            raise ValueError("pressure must be a 2-D array (n_times, n_mz).")
        if len(self.time) != self.pressure.shape[0]:
            raise ValueError("time length must match pressure.shape[0].")
        if len(self.mz) != self.pressure.shape[1]:
            raise ValueError("mz length must match pressure.shape[1].")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_rga(cls, rga, name: str | None = None) -> "SpectrumStack":
        """
        Adapter for clabs ``RGAMeasurement`` objects.

        Works whether or not ``rga.background_correct()`` has been called.
        If it has, ``open_time`` and ``close_time`` are read directly from
        the object; otherwise they are left unset until
        :meth:`background_correct` is called on the returned stack.

        Parameters
        ----------
        rga  : RGAMeasurement (clabs)
        name : display label; defaults to ``rga.sample_name``
        """
        open_time  = getattr(rga, "open_time",  None)
        close_time = getattr(rga, "close_time", None)

        stack = cls(
            time         = rga.time,
            pressure     = rga.pressure,
            mz           = rga.mz,
            shutter      = rga.shutter,
            shutter_time = rga.tey_time,
            open_time    = open_time,
            close_time   = close_time,
            name         = name or getattr(rga, "sample_name", ""),
        )
        raw = getattr(rga, "_raw_pressure", None)
        if raw is not None:
            stack._raw_pressure = np.asarray(raw, dtype=float)
        stack._bg_off1 = getattr(rga, "_bg_off1", None)
        stack._bg_off2 = getattr(rga, "_bg_off2", None)
        return stack

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_times(self) -> int:
        return len(self.time)

    @property
    def n_mz(self) -> int:
        return self.pressure.shape[1]

    @property
    def open_window(self) -> tuple[float, float]:
        """
        ``(open_time, close_time)`` in seconds.

        Resolution order:
        1. Explicit ``open_time``/``close_time`` (set by ``background_correct``
           or passed in the constructor).
        2. Edge detection on the stored shutter signal.
        3. Full time axis as fallback.
        """
        if self._open_time is not None and self._close_time is not None:
            return self._open_time, self._close_time

        if self._shutter is not None and self._shutter_time is not None:
            edges     = np.diff(self._shutter.astype(int))
            open_idx  = np.where(edges > 0)[0]
            close_idx = np.where(edges < 0)[0]
            if len(open_idx) and len(close_idx):
                return (
                    float(self._shutter_time[open_idx[0] + 1]),
                    float(self._shutter_time[close_idx[0]]),
                )

        return float(self.time[0]), float(self.time[-1])

    # ------------------------------------------------------------------
    # Transforms (immutable — always return a new SpectrumStack)
    # ------------------------------------------------------------------

    def background_correct(
        self,
        window:     float = 30.0,
        gap_before: float = 5.0,
        gap_after:  float = 10.0,
    ) -> "SpectrumStack":
        """
        Per-channel linear background subtraction.

        Requires that shutter and shutter_time arrays are available (set
        automatically by ``from_rga``).  Returns a new, corrected
        ``SpectrumStack`` with ``open_time`` and ``close_time`` set.

        Parameters
        ----------
        window     : duration (s) of each background window
        gap_before : gap (s) between end of pre-shutter window and shutter open
        gap_after  : gap (s) between shutter close and start of post-shutter window
        """
        if self._shutter is None or self._shutter_time is None:
            raise ValueError(
                "background_correct requires shutter and shutter_time arrays. "
                "Use SpectrumStack.from_rga(rga) to include these, or pass them "
                "in the constructor."
            )

        from .background import background_correct as _bg

        corrected, open_time, close_time = _bg(
            time         = self.time,
            pressure     = self.pressure,
            shutter      = self._shutter,
            shutter_time = self._shutter_time,
            window       = window,
            gap_before   = gap_before,
            gap_after    = gap_after,
        )

        new = SpectrumStack(
            time         = self.time,
            pressure     = corrected,
            mz           = self.mz,
            shutter      = self._shutter,
            shutter_time = self._shutter_time,
            open_time    = open_time,
            close_time   = close_time,
            name         = self.name,
        )
        new._raw_pressure = self.pressure.copy()
        new._bg_off1 = (open_time  - gap_before - window, open_time  - gap_before)
        new._bg_off2 = (close_time + gap_after,            close_time + gap_after + window)
        logger.info(
            "Background-corrected %r: open=%.1f s, close=%.1f s, %d m/z channels",
            self.name, open_time, close_time, self.n_mz,
        )
        return new

    def slice(self, t_start: float, t_end: float) -> "SpectrumStack":
        """
        Return a new ``SpectrumStack`` restricted to ``[t_start, t_end]``.

        Parameters
        ----------
        t_start, t_end : float — time bounds in seconds
        """
        mask = (self.time >= t_start) & (self.time <= t_end)
        if not mask.any():
            raise ValueError(
                f"No scans in [{t_start:.1f}, {t_end:.1f}] s "
                f"(time axis spans [{self.time[0]:.1f}, {self.time[-1]:.1f}] s)."
            )
        ot = max(self._open_time,  t_start) if self._open_time  is not None else None
        ct = min(self._close_time, t_end)   if self._close_time is not None else None
        return SpectrumStack(
            time         = self.time[mask],
            pressure     = self.pressure[mask, :],
            mz           = self.mz,
            shutter      = self._shutter,
            shutter_time = self._shutter_time,
            open_time    = ot,
            close_time   = ct,
            name         = self.name,
        )

    def integrated_pressure(
        self,
        mz_min:  int | None       = None,
        mz_max:  int | None       = None,
        exclude: list[int] | None = None,
    ) -> np.ndarray:
        """
        Sum partial pressures over m/z and return total signal vs time.

        Parameters
        ----------
        mz_min  : only include channels with m/z >= mz_min
        mz_max  : only include channels with m/z <= mz_max
        exclude : list of specific m/z values to drop

        Returns
        -------
        (n_times,) float array — total pressure at each scan
        """
        mask = np.ones(len(self.mz), dtype=bool)
        if mz_min is not None:
            mask &= self.mz >= mz_min
        if mz_max is not None:
            mask &= self.mz <= mz_max
        if exclude is not None:
            mask &= ~np.isin(self.mz, exclude)
        return self.pressure[:, mask].sum(axis=1)

    # ------------------------------------------------------------------
    # Reduction to a single MassSpectrum
    # ------------------------------------------------------------------

    def averaged(self, time_range: tuple[float, float] | None = None) -> "MassSpectrum":
        """
        Average all scans in *time_range* and return a :class:`MassSpectrum`.

        Parameters
        ----------
        time_range : ``(t_start, t_end)`` in seconds.  Defaults to
                     :attr:`open_window` (shutter-open window if known, or
                     full axis otherwise).
        """
        from .spectrum import MassSpectrum

        t_start, t_end = time_range if time_range is not None else self.open_window
        mask = (self.time >= t_start) & (self.time <= t_end)
        if not mask.any():
            raise ValueError(
                f"No scans in [{t_start:.1f}, {t_end:.1f}] s "
                f"(time axis spans [{self.time[0]:.1f}, {self.time[-1]:.1f}] s)."
            )

        mean_pressure = self.pressure[mask, :].mean(axis=0)
        nonzero       = mean_pressure > 0

        metadata = {
            "n_averaged_scans": int(mask.sum()),
            "t_start":          t_start,
            "t_end":            t_end,
        }
        return MassSpectrum(
            mz        = self.mz[nonzero],
            intensity = mean_pressure[nonzero],
            name      = self.name,
            metadata  = metadata,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize this SpectrumStack to disk with dill."""
        import dill
        with open(path, "wb") as f:
            dill.dump(self, f)
        logger.debug("SpectrumStack %r saved to %s", self.name, path)

    @classmethod
    def load(cls, path: str) -> "SpectrumStack":
        """Load a SpectrumStack previously saved with :meth:`save`."""
        import dill
        with open(path, "rb") as f:
            obj = dill.load(f)
        logger.debug("SpectrumStack %r loaded from %s", obj.name, path)
        return obj

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ot, ct = self.open_window
        return (
            f"SpectrumStack({self.name!r}, "
            f"n_times={self.n_times}, n_mz={self.n_mz}, "
            f"window=[{ot:.1f}, {ct:.1f}] s)"
        )
