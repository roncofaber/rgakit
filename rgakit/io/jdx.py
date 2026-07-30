"""
io/jdx.py
---------
JCAMP-DX parsing and generation for EI mass spectra.
"""

from __future__ import annotations

import re

import numpy as np


_PEAK_TABLE_RE = re.compile(
    r"##PEAK TABLE=\(XY\.\.XY\)(.*?)##END=", re.DOTALL | re.IGNORECASE
)
_PAIR_RE    = re.compile(r"(\d+),(\d+)")
_JDX_FIELDS = {
    "name":    re.compile(r"##TITLE[^\S\r\n]*=[^\S\r\n]*(.+)",           re.IGNORECASE),
    "cas":     re.compile(r"##CAS REGISTRY NO[^\S\r\n]*=[^\S\r\n]*(.+)", re.IGNORECASE),
    "formula": re.compile(r"##MOLFORM[^\S\r\n]*=[^\S\r\n]*(.+)",         re.IGNORECASE),
    "mw":      re.compile(r"##MW[^\S\r\n]*=[^\S\r\n]*(.+)",              re.IGNORECASE),
}
_USER_FIELD_RE = re.compile(r"^##\$([^=]+)=(.+)", re.MULTILINE)

# RGA-specific metadata keys preserved in JDX round-trips
RGA_META_KEYS = (
    "x", "y", "pd_ua", "n_open_scans", "background_correct",
    "t_open", "t_close",
)


def _try_cast(value: str):
    """Convert a string to int, float, or bool if possible; otherwise return it."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    return value


def parse_jdx(jdx_text: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse a JCAMP-DX string into (mz, intensity, metadata)."""
    metadata = {}
    for key, pattern in _JDX_FIELDS.items():
        m = pattern.search(jdx_text)
        if m:
            metadata[key] = m.group(1).strip()

    for m in _USER_FIELD_RE.finditer(jdx_text):
        metadata[m.group(1).strip()] = _try_cast(m.group(2).strip())

    match = _PEAK_TABLE_RE.search(jdx_text)
    if not match:
        raise ValueError("No PEAK TABLE found in JCAMP-DX text.")

    pairs = _PAIR_RE.findall(match.group(1))
    if not pairs:
        raise ValueError("Peak table is empty.")

    mz        = np.array([int(m)   for m, _ in pairs], dtype=int)
    intensity = np.array([float(i) for _, i in pairs], dtype=float)
    return mz, intensity, metadata


def generate_jdx(name: str, mz: np.ndarray, intensity: np.ndarray,
                 metadata: dict | None = None) -> str:
    """Generate a JCAMP-DX string from spectrum data."""
    meta = metadata or {}
    scaled = np.round(intensity / intensity.max() * 9999).astype(int)

    lines = [
        f"##TITLE={name}",
        "##JCAMP-DX=4.24",
        "##DATA TYPE=MASS SPECTRUM",
        "##DATA CLASS=PEAK TABLE",
        f"##CAS REGISTRY NO={meta.get('cas', '')}",
        f"##MOLFORM={meta.get('formula', '')}",
        f"##MW={meta.get('mw', '')}",
        f"##XUNITS=M/Z",
        f"##YUNITS=RELATIVE ABUNDANCE",
        f"##XFACTOR=1",
        f"##YFACTOR=1",
        f"##FIRSTX={int(mz.min())}",
        f"##LASTX={int(mz.max())}",
        f"##NPOINTS={len(mz)}",
    ]

    for key in RGA_META_KEYS:
        if key in meta:
            lines.append(f"##${key}={meta[key]}")

    lines.append(f"##PEAK TABLE=(XY..XY)")
    for m, i in zip(mz, scaled):
        lines.append(f"{m},{i}")
    lines.append("##END=")

    return "\n".join(lines)
