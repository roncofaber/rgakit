"""
io/msp.py
---------
NIST MSP format parsing for EI mass spectra.
"""

from __future__ import annotations

import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

_MSP_FIELD_MAP = {
    "name":      "name",
    "cas#":      "cas",
    "cas":       "cas",
    "formula":   "formula",
    "mw":        "mw",
    "exactmass": "exactmass",
    "nist#":     "nist_id",
    "db#":       "nist_id",
    "comments":  "comments",
    "inchikey":  "inchi_key",
}

_MSP_NUM_PEAKS_RE = re.compile(r"^num\s*peaks?\s*[:=]\s*(\d+)", re.IGNORECASE)
_MSP_FIELD_RE     = re.compile(r"^([^:]+?)\s*:\s*(.+)$")


def _try_cast(value: str):
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_msp_blocks(text: str) -> list[dict]:
    """
    Split MSP text into individual spectrum dicts.

    Each dict has ``mz``, ``intensity`` (np.ndarray) plus any metadata
    fields present in the block (``name``, ``cas``, ``formula``, ``mw``, …).
    """
    raw_blocks = re.split(r"\n{2,}", text.strip())
    results = []

    for block in raw_blocks:
        if not block.strip():
            continue
        lines     = block.splitlines()
        metadata  = {}
        mz_list:  list[int]   = []
        int_list: list[float] = []
        in_peaks  = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if in_peaks:
                tokens = line.split()
                for k in range(0, len(tokens) - 1, 2):
                    try:
                        mz_list.append(int(tokens[k]))
                        int_list.append(float(tokens[k + 1]))
                    except (ValueError, IndexError):
                        break
                continue

            m_np = _MSP_NUM_PEAKS_RE.match(line)
            if m_np:
                in_peaks = True
                continue

            m_f = _MSP_FIELD_RE.match(line)
            if m_f:
                key          = m_f.group(1).strip().lower()
                val          = m_f.group(2).strip()
                internal_key = _MSP_FIELD_MAP.get(key, key)
                metadata[internal_key] = _try_cast(val)

        if not mz_list:
            logger.warning("MSP block %r has no peaks - skipped.",
                           metadata.get("name", "<unnamed>"))
            continue

        mz        = np.array(mz_list, dtype=int)
        intensity = np.array(int_list, dtype=float)
        order     = np.argsort(mz)
        results.append({**metadata, "mz": mz[order], "intensity": intensity[order]})

    return results
