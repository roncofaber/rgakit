"""
report.py
---------
Plotting helpers and interactive HTML report generation for FitResult.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .result import FitResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_SPECTRUM_PLOT_HEIGHT   = 420   # px — fixed-height spectrum figures
_RESIDUAL_PLOT_HEIGHT   = 220   # px — residual panel (shorter)
_CONTRIB_HEIGHT_PER_ROW = 32    # px per compound in the contributions chart (matplotlib)


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def plot_contributions(
    result:    FitResult,
    ax=None,
    threshold: float = 1e-4,
):
    """Horizontal bar chart of compound contributions (matplotlib)."""
    import matplotlib.pyplot as plt

    contribs  = {k: v for k, v in result.contributions.items() if v >= threshold}
    names     = list(contribs.keys())
    sc        = getattr(result, "spectral_contributions", None)
    obs_total = float(getattr(result, "obs_total_full", None) or np.sum(result.observed)) or 1.0
    if sc is not None:
        pcts = [100.0 * float(np.sum(sc[n])) / obs_total for n in names]
    else:
        pcts = [100.0 * v for v in contribs.values()]

    # Re-sort by spectral coverage (descending)
    order = sorted(range(len(names)), key=lambda i: pcts[i], reverse=True)
    names = [names[i] for i in order]
    pcts  = [pcts[i]  for i in order]

    if ax is None:
        fig_h = max(3, _CONTRIB_HEIGHT_PER_ROW / 72 * len(names))
        _, ax = plt.subplots(figsize=(8, fig_h))

    ax.barh(names[::-1], pcts[::-1], color="steelblue")
    ax.set_xlabel("Contribution (%)")
    ax.set_title("Compound contributions")
    ax.axvline(0, color="k", linewidth=0.5)
    return ax


def plot_stacked_spectrum(
    result:  FitResult,
    library,
    ax=None,
    top_n:   int = 8,
):
    """
    Stacked bar chart of per-compound spectrum contributions (matplotlib).

    Parameters
    ----------
    result  : FitResult
    library : SpectraLibrary used for fitting
    top_n   : maximum number of compounds to show (highest first)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    contribs = result.contributions
    top      = list(contribs.items())[:top_n]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    colors = cm.tab20.colors
    bottom = np.zeros(len(result.grid))

    for i, (name, weight) in enumerate(top):
        try:
            spectrum = library[name]
        except KeyError:
            logger.warning("Compound %r not found in library; omitting from stacked plot.", name)
            continue
        contrib = spectrum.on_grid(result.grid) * weight
        ax.bar(result.grid, contrib, width=0.8, bottom=bottom,
               label=name, color=colors[i % len(colors)], alpha=0.85)
        bottom += contrib

    ax.bar(result.grid, result.observed, width=0.8, fill=False,
           edgecolor="black", linewidth=0.8, label="Observed")
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.set_title(f"Top-{top_n} stacked contributions")
    ax.legend(loc="upper right", fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# Interactive HTML report
# ---------------------------------------------------------------------------

# Consistent color for the "Observed" data series across all sections
_OBSERVED_COLOR = "rgba(255,127,14,0.85)"

_LAYOUT_BASE = dict(
    template="plotly_white",
    legend=dict(
        orientation="v",
        x=1.01, y=1,
        xanchor="left",
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="lightgray",
        borderwidth=1,
    ),
)

_PAGE_CSS = """
:root {
    --clr-primary:   #1f77b4;
    --clr-observed:  rgba(255,127,14,0.85);
    --clr-border:    #eee;
    --clr-text:      #222;
    --clr-muted:     #888;
    --clr-good:      #2ca02c;
    --clr-warn:      #d4a700;
    --clr-bad:       #d62728;
    --clr-bg:        #f5f5f5;
    --clr-surface:   #fff;
    --clr-hover:     #f0f4ff;
    --clr-stripe:    #f8f8f8;
    --clr-nav-text:  #555;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    font-family: Arial, Helvetica, sans-serif;
    background: var(--clr-bg);
    margin: 0;
    padding: 0;
    color: var(--clr-text);
}

/* Sticky navigation */
.rgakit-nav {
    position: sticky; top: 0; z-index: 200;
    background: var(--clr-surface); border-bottom: 1px solid #e8e8e8;
    padding: 0 32px;
    display: flex; align-items: center; gap: 0;
    box-shadow: 0 1px 4px rgba(0,0,0,.07);
    overflow-x: auto; white-space: nowrap;
}
.rgakit-nav > a {
    display: inline-block; padding: 11px 16px;
    font-size: 0.82em; color: var(--clr-nav-text); text-decoration: none;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
}
.rgakit-nav > a:hover { color: var(--clr-primary); border-bottom-color: var(--clr-primary); }
.rgakit-nav > a.active { color: var(--clr-primary); border-bottom-color: var(--clr-primary); font-weight: 600; }

/* Repository brand block (right side of nav) */
.rgakit-nav-brand {
    flex-shrink: 0;
    display: flex; flex-direction: column; align-items: flex-start;
    padding: 5px 10px 5px 14px;
    text-decoration: none; color: var(--clr-nav-text);
    border-left: 1px solid #eee;
    transition: color 0.15s;
}
.rgakit-nav-brand:hover { color: var(--clr-primary); }
.rgakit-nav-brand-row { display: flex; align-items: center; gap: 6px; line-height: 1; }
.rgakit-github-icon { width: 18px; height: 18px; fill: currentColor; flex-shrink: 0; }
.rgakit-nav-brand-name { font-size: 0.88em; font-weight: 700; }
/* version inline after name; timestamp on its own line below */
.rgakit-nav-brand-ver { font-size: 0.72em; color: #aaa; margin-left: 3px; }
.rgakit-nav-ts { font-size: 0.65em; color: #bbb; padding-left: 24px; margin-top: 2px; white-space: nowrap; }
/* Push the whole brand block to the far right */
.rgakit-nav-brand { margin-left: auto; }

.page-wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }

/* Shared button styles */
.rgakit-btn {
    padding: 5px 14px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: #f8f8f8;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s, border-color 0.15s;
}
.rgakit-btn:hover { background: #e8eef8; border-color: var(--clr-primary); }
.rgakit-btn.active { background: var(--clr-primary); color: #fff; border-color: var(--clr-primary); }
.rgakit-btn-primary {
    padding: 5px 14px;
    border: 1px solid #2c7be5;
    border-radius: 4px;
    background: #2c7be5;
    cursor: pointer;
    font-size: 12px;
    color: #fff;
    font-weight: 600;
    transition: background 0.15s;
}
.rgakit-btn-primary:hover { background: #1a6fd4; border-color: #1a6fd4; }

/* Page title (above the metrics card) */
.page-title {
    padding: 18px 0 10px;
    text-align: center;
}
.page-title h1 { margin: 0 0 3px; font-size: 1.55em; color: #1a1a1a; font-weight: 700; }
.page-title .header-sample { color: #888; font-size: 0.88em; margin: 0; }

/* Compact metrics card */
.header-card {
    background: var(--clr-surface);
    border-radius: 8px;
    box-shadow: 0 1px 5px rgba(0,0,0,.10);
    padding: 14px 24px 12px;
    margin-bottom: 20px;
    text-align: center;
}
.header-metrics {
    display: flex; justify-content: center; align-items: stretch;
    gap: 0;
}
.hmetric { flex: 0 0 auto; padding: 4px 28px; border-right: 1px solid #eee; cursor: help; }
.hmetric:last-child { border-right: none; }
.hmetric-val { display: block; font-size: 1.45em; font-weight: 700; color: var(--clr-primary); line-height: 1.15; }
.hmetric-lbl { display: block; font-size: 0.65em; color: #aaa; text-transform: uppercase; letter-spacing: .05em; margin-top: 2px; }
.fit-info-table { border-collapse: collapse; margin: 8px auto 0; font-size: 0.78em; }
.fit-info-table td { padding: 2px 10px 2px 6px; }
.fit-info-table .fik { color: #aaa; text-align: right; white-space: nowrap; padding-right: 6px; }
.fit-info-table .fiv { color: #444; font-weight: 500; text-align: left; border-left: 1px solid #eee; padding-left: 8px; }

/* Unexplained peaks table */
.unexp-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
.unexp-table th { background: #f8f8f8; text-align: left; padding: 6px 12px;
    border-bottom: 2px solid #e0e0e0; font-weight: 600; color: var(--clr-nav-text); white-space: nowrap; }
.unexp-table td { padding: 6px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
.unexp-table tr:hover td { background: #fafafa; }
.unexp-mz  { font-weight: bold; font-size: 14px; font-family: monospace; }
.unexp-val { font-family: monospace; color: #444; }
.unexp-bar-wrap { height: 7px; background: #ececec; border-radius: 4px; min-width: 80px; margin-top: 3px; }
.unexp-bar { height: 100%; border-radius: 4px; }
.unexp-sev-low  { color: #92710a; }
.unexp-sev-mid  { color: #c2410c; }
.unexp-sev-high { color: #b91c1c; }
.unexp-row-low  td:first-child { border-left: 3px solid #d4a400; }
.unexp-row-mid  td:first-child { border-left: 3px solid #ea580c; }
.unexp-row-high td:first-child { border-left: 3px solid #dc2626; }

/* Collapsible compound groups */
.compound-group { border: 1px solid #eee; border-radius: 6px; margin-bottom: 12px; overflow: hidden; }
.compound-group summary {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px; cursor: pointer; user-select: none;
    font-size: 0.95em; font-weight: 600; color: #444;
    background: #fafafa; list-style: none;
}
.compound-group summary::-webkit-details-marker { display: none; }
.compound-group summary::before {
    content: "▶"; font-size: 0.7em; color: #aaa;
    transition: transform 0.2s; flex-shrink: 0;
}
.compound-group[open] summary::before { transform: rotate(90deg); }
.compound-group summary:hover { background: #f4f4f4; }
.compound-group-body { padding: 12px 14px 14px; }
.compound-group-label { flex: 1; }
.compound-group-count { font-weight: normal; font-size: 0.88em; color: #999; }

/* Undetected library compounds */
.undetected-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 0; }
.undetected-card {
    width: 140px; border: 1px solid #e5e5e5; border-radius: 6px;
    padding: 8px; text-align: center; cursor: default; background: #fafafa;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.undetected-card:hover { border-color: #bbb; box-shadow: 0 2px 8px rgba(0,0,0,.10); }
.undetected-card img { width: 120px; height: 120px; object-fit: contain; display: block; margin: 0 auto; }
.undetected-card-ph { width: 120px; height: 120px; display: flex; align-items: center;
    justify-content: center; color: #ddd; font-size: 40px; margin: 0 auto; }
.undetected-card-name { font-size: 11px; color: #999; margin-top: 5px;
    word-break: break-word; line-height: 1.3;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }

/* Detected compound cards */
.detected-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
.detected-card {
    position: relative;
    border: 1px solid #e5e5e5; border-radius: 6px;
    padding: 8px; text-align: center; cursor: pointer;
    background: var(--clr-surface); transition: box-shadow 0.15s;
}
.undetected-card { cursor: pointer; }
.detected-card:hover   { box-shadow: 0 2px 10px rgba(0,0,0,.12); }
.detected-card img { width: 120px; height: 120px; object-fit: contain; display: block; margin: 0 auto; }
.detected-card-ph { width: 120px; height: 120px; display: flex; align-items: center;
    justify-content: center; color: #ddd; font-size: 40px; margin: 0 auto; }
.detected-card-name { font-size: 11px; color: #444; margin-top: 5px;
    word-break: break-word; line-height: 1.3; font-weight: 600;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.detected-card-pct {
    position: absolute; top: 4px; right: 6px;
    font-size: 11px; font-weight: 700; color: var(--clr-primary);
}

.section {
    background: var(--clr-surface);
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
    padding: 20px 24px;
    margin-bottom: 28px;
    scroll-margin-top: 44px;
}
.section h2 {
    margin: 0 0 14px;
    font-size: 1.1em;
    color: #1a1a1a;
    border-bottom: 2px solid var(--clr-primary);
    padding-bottom: 8px;
}

/* Stacked spectrum section */
.stacked-toolbar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    flex-wrap: wrap;
}
.stacked-toolbar input[type=text] {
    padding: 4px 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 12px;
    width: 160px;
}
.stacked-wrap { display: flex; gap: 16px; align-items: flex-start; }
.stacked-plot-col { flex: 1 1 0; min-width: 0; }
.stacked-legend-col { width: 260px; flex-shrink: 0; }
.stacked-legend-header {
    font-size: 0.8em; font-weight: bold; color: var(--clr-nav-text);
    padding: 6px 8px 6px; border-bottom: 1px solid #eee;
}
.stacked-legend-hint { font-weight: normal; color: #aaa; }
.stacked-legend-scroll {
    max-height: 360px; overflow-y: auto;
    border: 1px solid #eee; border-radius: 4px;
}
.legend-item {
    display: flex; align-items: flex-start; gap: 7px;
    padding: 5px 8px; cursor: pointer;
    transition: background 0.1s, opacity 0.15s;
    font-size: 12px;
}
.legend-item:hover { background: #f0f4ff; }
.legend-item.isolated { outline: 2px solid var(--clr-primary); border-radius: 2px; }
/* When any item is isolated, suppress hover highlight on non-isolated items */
.stacked-legend-scroll.isolating .legend-item:not(.isolated) { cursor: default; }
.stacked-legend-scroll.isolating .legend-item:not(.isolated):hover { background: none; }
.legend-swatch {
    display: inline-block; width: 13px; height: 13px;
    border-radius: 2px; flex-shrink: 0; margin-top: 2px;
}
.legend-label { flex: 1; white-space: normal; word-break: break-word; line-height: 1.3; }
.legend-pct { color: #999; font-size: 0.88em; flex-shrink: 0; padding-top: 1px; }


.compound-card {
    display:flex; align-items:center; gap:20px;
    padding:16px 20px; margin:0 auto 16px;
    max-width:500px;
    background:var(--clr-surface); border:1px solid var(--clr-border);
    border-radius:8px;
}
.compound-card img { height:120px; }
.compound-card-info { }
.compound-card-name { font-weight:700; font-size:1.2em; margin-bottom:4px; color:var(--clr-text); }
.compound-card-detail { color:var(--clr-muted); font-size:0.88em; line-height:1.6; }
.suggestion-grid { display:grid; grid-template-columns:repeat(5, 1fr); gap:10px; }
.suggestion-card {
    position:relative;
    background:var(--clr-surface); border:1px solid #e5e5e5;
    border-radius:6px; padding:8px; text-align:center; cursor:pointer;
    transition: box-shadow 0.15s;
}
.suggestion-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,.12); }
.suggestion-card img { width:120px; height:120px; object-fit:contain; display:block; margin:0 auto; }
.suggestion-card-name { font-size:11px; font-weight:600; color:#444; margin-top:5px;
    word-break:break-word; line-height:1.3;
    display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
.suggestion-card-meta { font-size:10px; color:var(--clr-muted); margin-top:2px; }
.suggestion-card-cos {
    position:absolute; top:4px; right:6px;
    font-size:11px; font-weight:700;
}
.coll-warning {
    background:#fff8e1; border:1px solid #ffe082; border-radius:6px;
    padding:10px 14px; margin-top:4px; font-size:0.84em;
}
.coll-warning-title { font-weight:600; color:#b45309; margin-bottom:6px; }
.coll-pair {
    display:inline-flex; align-items:center; gap:6px; margin:2px 4px 2px 0;
    padding:2px 10px; border-radius:12px; background:#fff0c2; border:1px solid #ffe082;
    font-size:0.82em; cursor:pointer; color:#92400e;
}
.coll-pair:hover { background:#ffe082; }
/* Modal collinearity section */
.rgakit-modal-collinear {
    padding:8px 16px 10px; border-top:1px solid #f0f0f0;
    font-size:0.81em; color:var(--clr-nav-text); flex-shrink:0;
}
.rgakit-modal-collinear-title { font-weight:600; color:#b45309; margin-bottom:5px; }
.rgakit-modal-coll-pill {
    display:inline-flex; align-items:center; gap:4px; margin:2px;
    padding:2px 10px; border-radius:12px; background:#fff8e1;
    border:1px solid #ffe082; cursor:pointer; color:#92400e;
}
.rgakit-modal-coll-pill:hover { background:#ffe082; }
/* Compound spectrum popup modal */
.rgakit-modal-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,0.45); z-index: 500;
    align-items: center; justify-content: center;
}
.rgakit-modal-overlay.open { display: flex; }
.rgakit-modal-box {
    background: var(--clr-surface); border-radius: 12px;
    width: min(700px, 94vw); max-height: 88vh;
    display: flex; flex-direction: column;
    box-shadow: 0 8px 40px rgba(0,0,0,.28);
    overflow: hidden;
}
.rgakit-modal-header {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 18px 20px 14px; border-bottom: 1px solid #eee;
    flex-shrink: 0; position: relative;
}
.rgakit-modal-close {
    position: absolute; top: 10px; right: 12px;
    font-size: 1.5em; color: #bbb; cursor: pointer;
    background: none; border: none; line-height: 1; padding: 0 4px;
}
.rgakit-modal-close:hover { color: #333; }
.rgakit-modal-info { flex: 1; min-width: 0; padding-right: 28px; }
.rgakit-modal-name    { font-size: 1.05em; font-weight: 700; color: #1a1a1a; margin-bottom: 2px;
                        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rgakit-modal-formula { font-size: 0.92em; color: #333; margin-bottom: 4px; }
.rgakit-modal-meta    { font-size: 0.75em; color: #aaa; }
.rgakit-modal-struct-wrap {
    flex-shrink: 0; display: flex; align-items: flex-end; gap: 4px;
}
.rgakit-modal-struct  {
    width: 90px; height: 90px; object-fit: contain;
    border-radius: 6px; background: #fafafa;
    border: 1.5px solid #222; box-sizing: border-box;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    transform-origin: top right;
}
.rgakit-modal-struct-wrap:hover {
    z-index: 100;
}
.rgakit-modal-struct:hover {
    transform: scale(2.5);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    position: relative; z-index: 100;
}
.rgakit-modal-copy-icon {
    width: 22px; height: 22px; padding: 3px;
    border: 1px solid #ddd; border-radius: 4px;
    background: var(--clr-surface); color: #999;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: color 0.15s, border-color 0.15s;
}
.rgakit-modal-copy-icon:hover { border-color: #aaa; color: #333; }
.rgakit-modal-plot-wrap { flex: 1; min-height: 200px; max-height: 340px; padding: 6px 10px 10px; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plotly_palette():
    try:
        import plotly.express as px
        return px.colors.qualitative.Plotly + px.colors.qualitative.D3
    except ImportError:
        return ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
                "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]


def _make_color_map(names: list[str]) -> dict[str, str]:
    palette = _plotly_palette()
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}


def _ymax_without(arr: np.ndarray, grid: np.ndarray,
                  exclude_mz=(2,), headroom: float = 1.15) -> float | None:
    """Max of *arr* ignoring *exclude_mz* channels, with headroom padding.

    Used to set an initial y-axis range that is not dominated by H2 (m/z=2).
    Returns None if no valid channels remain.
    """
    mask = ~np.isin(grid, exclude_mz)
    if not mask.any():
        return None
    return float(arr[mask].max()) * headroom


def _fig_to_json(fig) -> str:
    """Serialize a plotly figure to a compact JSON string."""
    return fig.to_json()


_STRUCTURE_CACHE_DIR = Path.home() / ".cache" / "rgakit" / "structures"
_REPO_URL = "https://github.com/roncofaber/rgakit"


def _smiles_to_svg(smiles: str, size: int = 200, padding: float = 0.02) -> str | None:
    """Render a SMILES string to a base64 SVG data URI using RDKit."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        import base64
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        opts = drawer.drawOptions()
        opts.padding = padding
        opts.clearBackground = False   # transparent background
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        b64 = base64.b64encode(drawer.GetDrawingText().encode()).decode()
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return None


_PUBCHEM_MIN_INTERVAL = 0.25   # seconds between requests (≤ 4 req/s, well under PubChem's 5/s limit)
_pubchem_last_call    = 0.0    # module-level timestamp of the last PubChem request


def _pubchem_get(identifier: str):
    """Rate-limited, retry-on-503 wrapper around pubchempy.get_compounds."""
    import time
    import pubchempy as pcp

    global _pubchem_last_call

    max_retries = 4
    for attempt in range(max_retries):
        # Enforce minimum interval between requests
        elapsed = time.monotonic() - _pubchem_last_call
        wait    = _PUBCHEM_MIN_INTERVAL - elapsed
        if wait > 0:
            time.sleep(wait)

        try:
            _pubchem_last_call = time.monotonic()
            return pcp.get_compounds(identifier, 'name')
        except Exception as exc:
            # Retry on server-busy (503); give up on anything else
            if "503" in str(exc) or "ServerBusy" in type(exc).__name__:
                backoff = _PUBCHEM_MIN_INTERVAL * (2 ** attempt)
                logger.warning(
                    "PubChem busy (attempt %d/%d) for %r — retrying in %.1fs",
                    attempt + 1, max_retries, identifier, backoff,
                )
                time.sleep(backoff)
            else:
                raise
    logger.warning("PubChem request failed after %d attempts for %r", max_retries, identifier)
    return []


def _get_structure(name: str, cas: str | None = None, smiles: str | None = None) -> str | None:
    """
    Return a base64 SVG data URI for a compound structure rendered by RDKit.

    If *smiles* is provided (e.g. from library metadata) it is used directly
    and neither the disk cache nor PubChem are consulted.  Otherwise the disk
    cache is checked first; PubChem is only contacted as a last resort.
    """
    import re

    # Fast path: SMILES already known (library metadata carries it for NIST compounds)
    if smiles:
        logger.debug("Rendering structure from provided SMILES for %r", name)
        svg = _smiles_to_svg(smiles)
        if svg is None:
            logger.warning("RDKit could not render structure for %r (SMILES: %s).", name, smiles)
        return svg

    # Disk cache
    safe         = re.sub(r'[^\w\-]', '_', cas or name)[:80]
    smiles_cache = _STRUCTURE_CACHE_DIR / f"{safe}.smi"

    if smiles_cache.exists():
        smiles = smiles_cache.read_text().strip() or None
        if smiles:
            logger.debug("Structure cache hit: %s", name)
        else:
            logger.debug("Structure cache hit (no SMILES): %s", name)
    else:
        # PubChem fallback — only reached for compounds without library SMILES
        smiles = None
        for identifier in filter(None, [cas, name]):
            logger.debug("Fetching SMILES from PubChem: %r", identifier)
            hits = _pubchem_get(identifier)
            if hits:
                smiles = hits[0].smiles
                logger.debug("SMILES found for %r: %s", name, smiles)
                break
        if not smiles:
            logger.warning("No SMILES found for %r (CAS=%s) - structure will be missing.", name, cas)
        _STRUCTURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        smiles_cache.write_text(smiles or "")

    if not smiles:
        return None
    svg = _smiles_to_svg(smiles)
    if svg is None:
        logger.warning("RDKit could not render structure for %r (SMILES: %s).", name, smiles)
    return svg



def _embed_plotly_div(div_id: str, fig_json: str, height: int) -> str:
    """Return HTML that creates a bare div and calls Plotly.newPlot."""
    return (
        f'<div id="{div_id}" style="width:100%;height:{height}px;"></div>\n'
        f'<script>\n'
        f'(function(){{\n'
        f'  var fig = {fig_json};\n'
        f'  fig.layout.height = {height};\n'
        f'  Plotly.newPlot("{div_id}", fig.data, fig.layout, {{responsive:true}});\n'
        f'}})();\n'
        f'</script>\n'
    )


# ---------------------------------------------------------------------------
# Pressure-map section (SpectrumStack)
# ---------------------------------------------------------------------------

def _build_pressure_map_section(stack) -> tuple[str, str]:
    """
    Build the interactive '2-D Pressure Map + Time Trace' section body.

    Ports the clabs generate_rga_report linked-view JS to a self-contained
    HTML fragment that can be embedded as a section in generate_report().

    Returns (heading, body_html).
    """
    mz         = stack.mz.tolist()
    time       = stack.time.tolist()
    pres       = stack.pressure             # (n_times, n_mz)
    pres_by_mz = pres.T.tolist()            # pres_by_mz[i] = time trace for mz[i]

    praw_by_mz = (
        stack._raw_pressure.T.tolist()
        if stack._raw_pressure is not None else None
    )

    pos  = pres[pres > 0]
    vmin = float(pos.min()) if pos.size else 1e-14

    if stack._raw_pressure is not None:
        pos_raw  = stack._raw_pressure[stack._raw_pressure > 0]
        vmin_raw = float(pos_raw.min()) if pos_raw.size else 1e-14
    else:
        vmin_raw = None

    open_t, close_t = stack.open_window
    bg1 = list(map(float, stack._bg_off1)) if stack._bg_off1 is not None else None
    bg2 = list(map(float, stack._bg_off2)) if stack._bg_off2 is not None else None

    payload = json.dumps({
        "mz":      mz,
        "time":    time,
        "pres":    pres_by_mz,
        "praw":    praw_by_mz,
        "vmin":    vmin,
        "vmin_raw": vmin_raw,
        "open_t":  open_t,
        "close_t": close_t,
        "bg1":     bg1,
        "bg2":     bg2,
    })

    toggle_btn = (
        '<button id="rgakit-map-toggle" '
        'style="float:right;margin-bottom:6px;padding:4px 12px;border:1px solid #bbb;'
        'border-radius:4px;background:#f5f5f5;cursor:pointer;font-size:0.82em;color:var(--clr-nav-text);">'
        'Show raw data</button>'
        if praw_by_mz is not None else ''
    )

    body = f"""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
  <p style="font-size:12px;color:#aaa;margin:0;">Hover to select an m/z channel &nbsp;·&nbsp; click to lock the selection</p>
  {toggle_btn}
</div>
<div id="rgakit-rga-heatmap" style="width:100%;"></div>
<div style="display:flex;align-items:baseline;gap:12px;margin:18px 0 6px;">
  <h3 style="margin:0;font-size:1em;color:#444;">Time Trace</h3>
  <span id="rgakit-mz-badge" style="font-size:0.85em;color:#1f77b4;font-weight:bold;">—</span>
  <span id="rgakit-lock-hint" style="font-size:0.75em;color:#bbb;"></span>
</div>
<div id="rgakit-rga-timetrace" style="width:100%;height:280px;"></div>

<div style="margin-top:28px;border-top:1px solid #eee;padding-top:18px;">
  <style>
    .rgakit-dual-range {{ position:relative; height:32px; display:flex; align-items:center; margin:8px 0 14px; }}
    .rgakit-dual-range input[type=range] {{ position:absolute; width:100%; height:0; pointer-events:none; background:transparent; -webkit-appearance:none; appearance:none; margin:0; padding:0; }}
    .rgakit-dual-range input[type=range]::-webkit-slider-thumb {{ pointer-events:all; -webkit-appearance:none; appearance:none; width:18px; height:18px; border-radius:50%; background:#1f77b4; cursor:pointer; border:2px solid #fff; box-shadow:0 1px 4px rgba(0,0,0,.25); }}
    .rgakit-dual-range input[type=range]::-moz-range-thumb {{ pointer-events:all; width:18px; height:18px; border-radius:50%; background:#1f77b4; cursor:pointer; border:2px solid #fff; box-shadow:0 1px 4px rgba(0,0,0,.25); }}
  </style>
  <div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">
    <h3 style="margin:0;font-size:1em;color:#444;">Integrated Pressure</h3>
    <span id="rgakit-integ-label" style="font-size:0.85em;color:{_OBSERVED_COLOR};font-weight:bold;"></span>
  </div>
  <div class="rgakit-dual-range">
    <div style="position:absolute;width:100%;height:4px;background:#ddd;border-radius:2px;pointer-events:none;"></div>
    <div id="rgakit-range-fill" style="position:absolute;height:4px;background:#1f77b4;border-radius:2px;pointer-events:none;"></div>
    <input type="range" id="rgakit-mz-lo">
    <input type="range" id="rgakit-mz-hi">
  </div>
  <div id="rgakit-rga-integrated" style="width:100%;height:280px;"></div>
</div>
<script>
(function() {{
  var D = {payload};

  var mzToIdx = {{}};
  D.mz.forEach(function(v, i) {{ mzToIdx[v] = i; }});

  function logClip(v) {{ return Math.log10(Math.max(v, D.vmin)); }}

  var zHeat    = D.pres.map(function(row) {{ return row.map(logClip); }});
  var zHeatRaw = D.praw && D.vmin_raw != null
    ? D.praw.map(function(row) {{
        return row.map(function(v) {{ return Math.log10(Math.max(v, D.vmin_raw)); }});
      }})
    : null;
  var showingRaw = false;

  function toggleMap() {{
    if (!zHeatRaw) return;
    showingRaw = !showingRaw;
    Plotly.restyle('rgakit-rga-heatmap', {{ z: [showingRaw ? zHeatRaw : zHeat] }}, [0]);
    var btn = document.getElementById('rgakit-map-toggle');
    if (btn) btn.textContent = showingRaw ? 'Show corrected' : 'Show raw data';
  }}
  var _toggleBtn = document.getElementById('rgakit-map-toggle');
  if (_toggleBtn) _toggleBtn.addEventListener('click', toggleMap);

  var logMin = Math.log10(D.vmin);
  var flatMax = D.pres.reduce(function(mx, row) {{
    return Math.max(mx, Math.max.apply(null, row));
  }}, D.vmin);
  var logMax   = Math.log10(Math.max(flatMax, D.vmin));
  var _supMap = {{'0':'\u2070','1':'\u00b9','2':'\u00b2','3':'\u00b3','4':'\u2074',
                  '5':'\u2075','6':'\u2076','7':'\u2077','8':'\u2078','9':'\u2079','-':'\u207b'}};
  function _toSup(n) {{ return String(n).split('').map(function(c){{ return _supMap[c]||c; }}).join(''); }}
  var tickVals = [], tickText = [];
  for (var tv = Math.ceil(logMin); tv <= Math.floor(logMax); tv++) {{
    tickVals.push(tv);
    tickText.push('10' + _toSup(tv));
  }}

  function shutterShapes(forHeatmap) {{
    var shapes = [];
    var lk = {{ color: 'red', width: 1.5, dash: 'dash' }};
    if (D.open_t  !== null) shapes.push({{ type:'line', x0:D.open_t,  x1:D.open_t,  y0:0, y1:1, yref:'paper', line:lk }});
    if (D.close_t !== null) shapes.push({{ type:'line', x0:D.close_t, x1:D.close_t, y0:0, y1:1, yref:'paper', line:lk }});
    if (D.bg1 && D.bg2) {{
      var bgColor = forHeatmap ? 'rgba(255,255,255,0.13)' : 'rgba(255,165,0,0.13)';
      shapes.push({{ type:'rect', x0:D.bg1[0], x1:D.bg1[1], y0:0, y1:1, yref:'paper', fillcolor:bgColor, line:{{width:0}} }});
      shapes.push({{ type:'rect', x0:D.bg2[0], x1:D.bg2[1], y0:0, y1:1, yref:'paper', fillcolor:bgColor, line:{{width:0}} }});
    }}
    return shapes;
  }}

  var initIdx    = Math.floor(D.mz.length / 2);
  var selLine    = {{ type:'line', x0:D.time[0], x1:D.time[D.time.length-1],
                     y0:D.mz[initIdx], y1:D.mz[initIdx],
                     line:{{ color:'rgba(255,80,80,0.7)', width:1.5 }} }};
  var heatShapes = shutterShapes(true).concat([selLine]);

  // Square the heatmap: match height to rendered container width (cap at 800 px).
  // offsetWidth may be 0 if the browser hasn't laid out yet — fall back to the
  // parent's width, then 700 px.  After the first paint, rAF corrects any mismatch.
  var hmEl    = document.getElementById('rgakit-rga-heatmap');
  var hmSize  = Math.min(
    hmEl.offsetWidth || (hmEl.parentElement ? hmEl.parentElement.clientWidth : 0) || 700,
    800
  );
  if (hmSize < 200) hmSize = 700;
  hmEl.style.height = hmSize + 'px';

  Plotly.newPlot('rgakit-rga-heatmap', [{{
    type: 'heatmap',
    x: D.time, y: D.mz, z: zHeat,
    colorscale: 'Viridis', zsmooth: false,
    hovertemplate: 'Time: %{{x:.1f}} s<br>m/z: %{{y}}<br>log\u2081\u2080(P): %{{z:.2f}}<extra></extra>',
    colorbar: {{ title: {{ text: 'Partial pressure [Torr]', side:'right' }},
                 tickvals: tickVals, ticktext: tickText }},
  }}], {{
    template: 'plotly_white',
    xaxis: {{ title:{{ text:'Time [s]' }}, showspikes:true, spikemode:'across', spikecolor:'#aaa', spikethickness:1 }},
    yaxis: {{ title:{{ text:'m/z' }} }},
    margin: {{ l:60, r:130, t:20, b:50 }}, height: hmSize,
    shapes: heatShapes,
  }}, {{ responsive:true }});

  // Correct size on first paint in case offsetWidth was 0 at script-run time
  requestAnimationFrame(function() {{
    var w = Math.min(document.getElementById('rgakit-rga-heatmap').offsetWidth || 700, 800);
    if (w !== hmSize) {{ hmSize = w; Plotly.relayout('rgakit-rga-heatmap', {{height: w}}); }}
  }});

  var ttTraces = [{{
    type:'scatter', mode:'lines',
    x: D.time, y: D.pres[initIdx],
    name:'Corrected',
    line: {{ color:'steelblue', width:1.8 }},
    hovertemplate: '%{{y:.3e}} Torr<extra>Corrected</extra>',
  }}];
  if (D.praw) {{
    ttTraces.push({{
      type:'scatter', mode:'lines',
      x: D.time, y: D.praw[initIdx],
      name:'Raw',
      line: {{ color:'steelblue', width:1.2, dash:'dot' }},
      opacity: 0.5,
      hovertemplate: '%{{y:.3e}} Torr<extra>Raw</extra>',
    }});
  }}

  Plotly.newPlot('rgakit-rga-timetrace', ttTraces, {{
    template: 'plotly_white',
    xaxis: {{ title:{{ text:'Time [s]' }}, showspikes:true, spikemode:'across', spikecolor:'#aaa', spikethickness:1 }},
    yaxis: {{ title:{{ text:'Partial pressure [Torr]' }}, exponentformat:'e' }},
    legend: {{ x:1.01, y:1, xanchor:'left' }},
    hovermode: 'x unified',
    margin: {{ l:80, r:130, t:10, b:50 }}, height:280,
    shapes: shutterShapes(false),
  }}, {{ responsive:true }});

  var locked  = false;
  var curIdx  = initIdx;
  var mzBadge  = document.getElementById('rgakit-mz-badge');
  var lockHint = document.getElementById('rgakit-lock-hint');
  mzBadge.textContent = 'm/z = ' + D.mz[initIdx];

  function updateTrace(idx) {{
    if (idx === curIdx) return;
    curIdx = idx;
    var updates = D.praw
      ? {{ y: [D.pres[idx], D.praw[idx]] }}
      : {{ y: [D.pres[idx]] }};
    var idxs = D.praw ? [0, 1] : [0];
    Plotly.restyle('rgakit-rga-timetrace', updates, idxs);
    var newShapes = shutterShapes(true).concat([{{
      type:'line', x0:D.time[0], x1:D.time[D.time.length-1],
      y0:D.mz[idx], y1:D.mz[idx],
      line:{{ color:'rgba(255,80,80,0.7)', width:1.5 }},
    }}]);
    Plotly.relayout('rgakit-rga-heatmap', {{ shapes: newShapes }});
    mzBadge.textContent = 'm/z = ' + D.mz[idx];
  }}

  document.getElementById('rgakit-rga-heatmap').on('plotly_hover', function(data) {{
    if (locked) return;
    var idx = mzToIdx[data.points[0].y];
    if (idx !== undefined) updateTrace(idx);
  }});

  document.getElementById('rgakit-rga-heatmap').on('plotly_click', function(data) {{
    var idx = mzToIdx[data.points[0].y];
    if (idx === undefined) return;
    if (locked && idx === curIdx) {{
      locked = false; lockHint.textContent = '';
    }} else {{
      locked = true;
      lockHint.textContent = '(locked \u2014 click same m/z to release)';
      updateTrace(idx);
    }}
  }});

  // ── Integrated pressure ──────────────────────────────────────────────────
  var nMz = D.mz.length, nT = D.time.length;

  // Prefix sums over m/z: prefix[i][t] = sum(D.pres[0..i-1][t])
  var prefix = new Array(nMz + 1);
  prefix[0] = new Array(nT).fill(0);
  for (var mi = 0; mi < nMz; mi++) {{
    prefix[mi + 1] = new Array(nT);
    for (var ti = 0; ti < nT; ti++) {{
      prefix[mi + 1][ti] = prefix[mi][ti] + D.pres[mi][ti];
    }}
  }}

  function getIntegrated(lo, hi) {{
    var y = new Array(nT);
    for (var ti = 0; ti < nT; ti++) y[ti] = prefix[hi + 1][ti] - prefix[lo][ti];
    return y;
  }}

  var loSlider   = document.getElementById('rgakit-mz-lo');
  var hiSlider   = document.getElementById('rgakit-mz-hi');
  var integLabel = document.getElementById('rgakit-integ-label');
  var rangeFill  = document.getElementById('rgakit-range-fill');

  loSlider.min = 0; loSlider.max = nMz - 1; loSlider.value = 0;
  hiSlider.min = 0; hiSlider.max = nMz - 1; hiSlider.value = nMz - 1;
  loSlider.style.zIndex = 1; hiSlider.style.zIndex = 2;

  function updateIntegLabel(lo, hi) {{
    integLabel.textContent = 'm/z ' + D.mz[lo] + ' \u2013 ' + D.mz[hi];
    var pLo = lo / (nMz - 1) * 100, pHi = hi / (nMz - 1) * 100;
    rangeFill.style.left  = pLo + '%';
    rangeFill.style.width = (pHi - pLo) + '%';
  }}
  updateIntegLabel(0, nMz - 1);

  Plotly.newPlot('rgakit-rga-integrated', [{{
    type: 'scatter', mode: 'lines',
    x: D.time, y: getIntegrated(0, nMz - 1),
    name: '\u03a3 pressure',
    line: {{ color: '{_OBSERVED_COLOR}', width: 1.8 }},
    hovertemplate: '%{{y:.3e}} Torr<extra>\u03a3 Pressure</extra>',
  }}], {{
    template: 'plotly_white',
    xaxis: {{ title: {{ text: 'Time [s]' }}, showspikes: true, spikemode: 'across', spikecolor: '#aaa', spikethickness: 1 }},
    yaxis: {{ title: {{ text: '\u03a3 Partial pressure [Torr]' }}, exponentformat: 'e' }},
    hovermode: 'x unified',
    margin: {{ l: 80, r: 40, t: 10, b: 50 }}, height: 280,
    shapes: shutterShapes(false),
  }}, {{ responsive: true }});

  loSlider.addEventListener('input', function() {{
    var lo = parseInt(loSlider.value), hi = parseInt(hiSlider.value);
    if (lo > hi) {{ loSlider.value = hi; lo = hi; }}
    loSlider.style.zIndex = (lo === nMz - 1) ? 3 : 1;
    updateIntegLabel(lo, hi);
    Plotly.restyle('rgakit-rga-integrated', {{ y: [getIntegrated(lo, hi)] }}, [0]);
  }});
  hiSlider.addEventListener('input', function() {{
    var lo = parseInt(loSlider.value), hi = parseInt(hiSlider.value);
    if (hi < lo) {{ hiSlider.value = lo; hi = lo; }}
    hiSlider.style.zIndex = (hi === 0) ? 3 : 2;
    updateIntegLabel(lo, hi);
    Plotly.restyle('rgakit-rga-integrated', {{ y: [getIntegrated(lo, hi)] }}, [0]);
  }});
}})();
</script>"""

    return "2-D Pressure Map", body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    result:                       FitResult,
    library=None,
    output_path:                  str | Path = "rga_fit_report.html",
    title:                        str        = "RGA Fit Report",
    threshold:                    float      = 1e-4,
    top_n:                        int | None = None,
    spectrum=None,
    stack=None,
    time_result=None,
    compound=None,
    decomposition=None,
    fetch_structures:             bool       = True,
    fetch_undetected_structures:  bool       = True,
    unexplained_threshold:        float      = 0.05,
) -> Path:
    """
    Generate a self-contained, scrollable HTML report from a FitResult.

    Produces interactive sections:
      1. Summary card (residual, coverage, R², grid points, compound count)
      2. Observed vs Fitted Spectrum
      3. Residual Spectrum (observed − fitted)
      4. Compound Contributions (%)
      5. Stacked Contributions — all compounds above threshold with scrollable,
         hover-to-highlight, click-to-select (multi) legend, log-scale toggle,
         legend filter, and CSV download (requires *library*)

    Requires plotly (included in the standard rgakit install).

    Parameters
    ----------
    result      : FitResult from SpectraLibrary.fit()
    library     : SpectraLibrary used for fitting; enables stacked-spectrum section
    output_path : destination .html file path
    title       : page title shown at the top of the report
    threshold   : minimum weight for a compound to appear
    top_n       : limit the stacked section to the top-N contributors;
                  ``None`` (default) shows all compounds above *threshold*
    spectrum               : the MassSpectrum that was fitted; when provided, its
                             metadata (x, y, pd_ua, n_open_scans) is shown in the header
    fetch_structures       : if True (default), attempt to fetch structure images from
                             PubChem for each compound; requires internet at report
                             generation time; silently skipped on failure
    unexplained_threshold  : m/z channels where (observed − fitted) exceeds this
                             fraction of the spectrum's max observed signal are listed
                             as unexplained peaks (default 0.05 = 5%)

    Returns
    -------
    Path to the saved HTML file.
    """
    import html as _html_mod
    try:
        import plotly.graph_objects as go
        from plotly.offline import get_plotlyjs
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("plotly is required: pip install rgakit")

    contribs  = {k: v for k, v in result.contributions.items() if v >= threshold}
    names     = list(contribs.keys())
    weights   = list(contribs.values())

    sc        = getattr(result, "spectral_contributions", None)
    # Use full-spectrum total so percentages are wrt everything the instrument
    # measured, not just the portion covered by the library grid.
    obs_total = float(getattr(result, "obs_total_full", None) or np.sum(result.observed)) or 1.0
    if sc is not None:
        pcts = [100.0 * float(np.sum(sc[n])) / obs_total for n in names]
    else:
        w_sum = sum(weights) or 1.0
        pcts  = [100.0 * w / w_sum for w in weights]

    # Re-sort by spectral coverage (descending)
    order   = sorted(range(len(names)), key=lambda i: pcts[i], reverse=True)
    names   = [names[i]   for i in order]
    weights = [weights[i] for i in order]
    pcts    = [pcts[i]    for i in order]

    color_map = _make_color_map(names)

    undetected = (
        sorted(n for n in library.names() if n not in contribs)
        if library is not None else []
    )

    # Fetch/render structure images
    structures: dict[str, str | None] = {}
    if fetch_structures:
        all_fetch = names + (undetected if fetch_undetected_structures else [])
        if all_fetch:
            for n in all_fetch:
                meta   = library[n].metadata if library is not None else {}
                cas    = meta.get("cas")
                smiles = meta.get("smiles")
                structures[n] = _get_structure(n, cas, smiles)
            found = sum(1 for v in structures.values() if v)
            logger.info("Structures: %d/%d rendered successfully.", found, len(all_fetch))

    # Collinearity map: {compound_name: [{name, cosine}, ...]} for modal display.
    # Threshold of 0.97: only flag near-identical spectra.
    _collinearity_map: dict = {}
    _collinearity_pairs: list = []   # raw (a, b, cos) list for diagnostics section
    if library is not None:
        try:
            _collinearity_pairs = library.check_collinearity(
                grid=result.grid, threshold=0.97
            )
            for _a, _b, _cos in _collinearity_pairs:
                _collinearity_map.setdefault(_a, []).append(
                    {"name": _b, "cosine": round(float(_cos), 3)}
                )
                _collinearity_map.setdefault(_b, []).append(
                    {"name": _a, "cosine": round(float(_cos), 3)}
                )
        except Exception:
            pass

    # Embed reference spectra for the compound popup modal.
    # Single helper so detected, undetected, and suggested compounds all
    # populate the same fields consistently.
    def _spec_entry(sp, svg: str = "", collinear: list | None = None) -> dict:
        meta = getattr(sp, "metadata", {}) or {}
        formula = (meta.get("formula") or meta.get("molecular_formula") or "")
        return {
            "mz":            sp.mz.tolist(),
            "intensity":     sp.normalized.tolist(),
            "formula":       formula,
            "cas":           meta.get("cas") or "",
            "mw":            str(meta.get("mw") or meta.get("molecular_weight") or ""),
            "smiles":        meta.get("smiles") or "",
            "svg":           svg,
            "collinear_with": collinear or [],
        }

    _lib_spectra: dict = {}
    if library is not None:
        for _n in names + undetected:
            try:
                _sp = library[_n]
                _lib_spectra[_n] = _spec_entry(
                    _sp,
                    svg=structures.get(_n) or "",
                    collinear=_collinearity_map.get(_n, []),
                )
            except Exception:
                pass

    _suggestions = getattr(result, "suggestions", None)
    if _suggestions:
        for _sug_spec, _sug_dist in _suggestions:
            _sn = getattr(_sug_spec, "name", "") or ""
            if _sn and _sn not in _lib_spectra:
                _sug_smiles = (getattr(_sug_spec, "metadata", {}) or {}).get("smiles") or ""
                _sug_svg = ""
                if _sug_smiles:
                    try:
                        from rdkit.Chem import MolFromSmiles
                        from rdkit.Chem.Draw import MolDraw2DSVG
                        import base64 as _b64
                        _mol = MolFromSmiles(_sug_smiles)
                        if _mol is not None:
                            _dd = MolDraw2DSVG(250, 140)
                            _dd.drawOptions().clearBackground = False
                            _dd.DrawMolecule(_mol)
                            _dd.FinishDrawing()
                            _sug_svg = (
                                "data:image/svg+xml;base64,"
                                + _b64.b64encode(_dd.GetDrawingText().encode()).decode()
                            )
                    except Exception:
                        pass
                _lib_spectra[_sn] = _spec_entry(_sug_spec, svg=_sug_svg)

    _lib_spectra_json = json.dumps(_lib_spectra)

    sections: list[tuple[str, str]] = []   # (heading, div_html)

    # Pressure map + time trace — shown first when a SpectrumStack is provided
    if stack is not None:
        sections.append(_build_pressure_map_section(stack))

    # ------------------------------------------------------------------ #
    # Fit Diagnostics section (solver cards + collinearity warnings)
    # ------------------------------------------------------------------ #
    _cond_val   = getattr(result, "condition_number", None)
    _cond_color = ("var(--clr-good)" if _cond_val is not None and _cond_val < 100
                   else "var(--clr-warn)" if _cond_val is not None and _cond_val < 1000
                   else "var(--clr-bad)")
    _cond_str   = f"{_cond_val:.0f}" if _cond_val is not None else "—"

    _fp0        = getattr(result, "fit_params", {}) or {}
    _method_disp = _fp0.get("method", "nnls").upper()
    _method_details = []
    if _fp0.get("alpha"):
        _method_details.append(f"α={_fp0['alpha']}")
    if _fp0.get("l1_ratio") is not None:
        _method_details.append(f"L1={_fp0['l1_ratio']}")
    if _fp0.get("n_trials") and _fp0["n_trials"] > 1:
        _method_details.append(f"{_fp0['n_trials']} trials")
    if _method_details:
        _method_disp += (
            "<br><small style='font-weight:400;font-size:.65em'>"
            + ", ".join(_method_details) + "</small>"
        )
    _weighted_disp = _fp0.get("weighted", False)

    _n_searched = len(names) + len(undetected)

    # Collinearity warnings among detected compounds only
    _det_set = set(names)
    _det_coll = [(a, b, c) for a, b, c in _collinearity_pairs
                 if a in _det_set and b in _det_set]
    _coll_block = ""
    if _det_coll:
        _pills = "".join(
            f'<span class="coll-pair" data-coll-spec="{_html_mod.escape(a, quote=True)}">{_html_mod.escape(a)}</span>'
            f'<span style="color:#aaa;font-size:.8em">\u2194</span>'
            f'<span class="coll-pair" data-coll-spec="{_html_mod.escape(b, quote=True)}">{_html_mod.escape(b)}</span>'
            f'<span style="color:#bbb;font-size:.8em;margin:0 6px">cos={c:.2f}</span>'
            for a, b, c in sorted(_det_coll, key=lambda x: -x[2])
        )
        _coll_block = f"""
<div class="coll-warning">
  <div class="coll-warning-title">\u26a0 Near-collinear detected compounds \u2014 weights may be unreliable</div>
  {_pills}
  <div style="margin-top:6px;color:#92400e;font-size:.8em">
    Cosine similarity &gt; 0.97 \u2014 click a compound to inspect its reference spectrum.
  </div>
</div>
<script>
document.querySelectorAll('.coll-pair[data-coll-spec]').forEach(function(el) {{
  el.addEventListener('click', function() {{
    if (window.openSpecModal) window.openSpecModal(this.dataset.collSpec);
  }});
}});
</script>"""

    # Diagnostics info is displayed in the header card directly.

    # ------------------------------------------------------------------ #
    # Summary metrics
    # ------------------------------------------------------------------ #
    obs_mean     = float(np.mean(result.observed))
    ss_total     = float(np.sum((result.observed - obs_mean) ** 2))
    ss_res       = float(result.residual ** 2)
    r2           = max(0.0, 1.0 - ss_res / ss_total) if ss_total > 0 else 0.0
    coverage     = float(np.sum(np.minimum(result.fitted, result.observed))) / (
        float(np.sum(result.observed)) or 1.0
    ) * 100
    obs_norm      = float(np.linalg.norm(result.observed)) or 1.0
    norm_residual = result.residual / obs_norm

    # Quality color coding for key metrics
    def _metric_color(val, good_thresh, warn_thresh, higher_is_better=True):
        """Return a CSS color string based on quality thresholds."""
        if higher_is_better:
            if val >= good_thresh: return "var(--clr-good)"
            if val >= warn_thresh: return "var(--clr-warn)"
            return "var(--clr-bad)"
        else:
            if val <= good_thresh: return "var(--clr-good)"
            if val <= warn_thresh: return "var(--clr-warn)"
            return "var(--clr-bad)"

    r2_color       = _metric_color(r2,            0.95, 0.80, higher_is_better=True)
    nr_color       = _metric_color(norm_residual, 0.05, 0.15, higher_is_better=False)
    cov_color      = _metric_color(coverage,      95.0, 80.0, higher_is_better=True)

    # Report provenance
    try:
        from importlib.metadata import version as _pkg_version
        _rgakit_ver = _pkg_version("rgakit")
    except Exception:
        _rgakit_ver = "unknown"
    _report_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # m/z channels excluded from the fit — used to set y-axis range sensibly
    _fp           = getattr(result, "fit_params", {}) or {}
    fit_exclude_mz = tuple(_fp.get("exclude_mz", (2,)))

    # Unexplained peaks: channels where residual > threshold * observed.max()
    residual_arr  = result.observed - result.fitted
    obs_max       = result.observed.max() or 1.0
    unexp_mask    = residual_arr > unexplained_threshold * obs_max
    unexp_indices = np.argsort(residual_arr)[::-1]   # descending
    unexp_peaks   = [
        (int(result.grid[i]), residual_arr[i], result.observed[i])
        for i in unexp_indices
        if unexp_mask[i]
    ]

    # ------------------------------------------------------------------ #
    # Section 1 — Observed vs Fitted + Residual (merged, shared x-axis)
    # ------------------------------------------------------------------ #
    pos_mask = residual_arr >= 0
    neg_mask = ~pos_mask

    fig_spec = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
    )
    # Overlay plot: both traces on the same axis.
    # Observed below (added first), Fitted on top (added last, semi-transparent).
    # Alpha lets the viewer read over/under-explanation directly:
    #   fitted > observed → blue extends above orange (over-fitted)
    #   fitted < observed → orange peeks above the shorter blue bar (under-fitted)
    fig_spec.add_trace(go.Bar(
        x=result.grid, y=result.observed,
        name="Observed",
        marker_color="rgba(255,127,14,0.70)",
        hovertemplate="Observed: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig_spec.add_trace(go.Bar(
        x=result.grid, y=result.fitted,
        name="Fitted",
        marker_color="rgba(31,119,180,0.60)",
        hovertemplate="Fitted: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    fig_spec.add_trace(go.Bar(
        x=result.grid[pos_mask], y=residual_arr[pos_mask],
        name="Unexplained",
        marker_color="rgba(214,39,40,0.7)",
        hovertemplate="Unexplained: %{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig_spec.add_trace(go.Bar(
        x=result.grid[neg_mask], y=residual_arr[neg_mask],
        name="Over-fitted",
        marker_color="rgba(31,119,180,0.55)",
        hovertemplate="Over-fitted: %{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig_spec.add_hline(y=0, line_width=1, line_color="black", row=2, col=1)

    # Y-axis bounds come from the observed spectrum only — the measured data
    # defines the scale, not the fit.  fit_exclude_mz controls the fit only.
    ymax1    = _ymax_without(result.observed, result.grid, exclude_mz=())
    ymax_res = _ymax_without(np.abs(residual_arr), result.grid, exclude_mz=fit_exclude_mz)
    spec_height = _SPECTRUM_PLOT_HEIGHT + _RESIDUAL_PLOT_HEIGHT

    # x limits: span the full observed m/z range (result.grid == all observed channels)
    x_lo = float(result.grid[0])  - 1.0
    x_hi = float(result.grid[-1]) + 1.0

    fig_spec.update_layout(
        **_LAYOUT_BASE,
        barmode="overlay",
        hovermode="x unified",
        height=spec_height,
        margin=dict(l=60, r=40, t=30, b=50),
    )
    fig_spec.update_yaxes(title_text="Relative intensity", row=1, col=1)
    fig_spec.update_yaxes(title_text="Observed − Fitted",  row=2, col=1)
    fig_spec.update_xaxes(title_text="m/z", row=2, col=1)
    fig_spec.update_xaxes(
        range=[x_lo, x_hi],
        showspikes=True, spikemode="across",
        spikecolor="#aaa", spikethickness=1, spikesnap="cursor",
    )
    if ymax1:
        fig_spec.update_yaxes(range=[0, ymax1], row=1, col=1)
    if ymax_res:
        fig_spec.update_yaxes(range=[-ymax_res, ymax_res], row=2, col=1)

    sections.append((
        "Observed vs Fitted Spectrum",
        _embed_plotly_div("rgakit-spectrum", _fig_to_json(fig_spec), spec_height),
    ))

    # ------------------------------------------------------------------ #
    # Section 2 — Unexplained peaks (shown right after spectrum)
    # ------------------------------------------------------------------ #
    if unexp_peaks:
        # Sort by unexplained fraction descending (most proportionally problematic first)
        _unexp_sorted = sorted(
            [(mz, res, obs) for mz, res, obs in unexp_peaks if obs > 0],
            key=lambda t: t[1] / t[2], reverse=True,
        )
        rows = []
        for mz, res, obs in _unexp_sorted:
            pct     = res / obs * 100
            bar_w   = min(100, pct)
            if pct < 30:
                sev, bar_color = "low",  "#d4a400"
            elif pct < 60:
                sev, bar_color = "mid",  "#ea580c"
            else:
                sev, bar_color = "high", "#dc2626"
            rows.append(
                f'<tr class="unexp-row-{sev}">'
                f'<td class="unexp-mz unexp-sev-{sev}">{mz}</td>'
                f'<td class="unexp-val">{obs:.4f}</td>'
                f'<td class="unexp-val">{res:.4f}</td>'
                f'<td>'
                f'<span class="unexp-sev-{sev}" style="font-weight:600">{pct:.0f}%</span>'
                f'<div class="unexp-bar-wrap">'
                f'<div class="unexp-bar" style="width:{bar_w:.0f}%;background:{bar_color}"></div>'
                f'</div>'
                f'</td>'
                f'</tr>'
            )
        _n = len(rows)
        unexp_body = (
            f'<p style="font-size:13px;color:#888;margin:0 0 10px;">'
            f'<b>{_n}</b> m/z channel{"s" if _n != 1 else ""} with unexplained signal above '
            f'{unexplained_threshold*100:.0f}% of the observed maximum. '
            f'These likely correspond to compounds absent from the library.</p>'
            f'<table class="unexp-table">'
            f'<thead><tr>'
            f'<th>m/z</th><th>Observed</th><th>Residual</th><th>Unexplained</th>'
            f'</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody>'
            f'</table>'
        )
        sections.append(("Unexplained Peaks", unexp_body))

    # ------------------------------------------------------------------ #
    # Section 3 — Compound contributions: horizontal bar chart
    # ------------------------------------------------------------------ #
    def _csv_field(s: str) -> str:
        """RFC 4180: quote strings that contain commas, quotes, or newlines."""
        s = str(s).replace('"', '""')
        return f'"{s}"' if (',' in s or '"' in s or '\n' in s) else s

    csv_lines = ['"Compound","Weight","Percentage"'] + [
        f"{_csv_field(n)},{w:.6f},{p:.4f}"
        for n, w, p in zip(names, weights, pcts)
    ]
    csv_content = json.dumps("\n".join(csv_lines))

    bar_colors = [color_map[n] for n in names]

    # Horizontal bar chart — reversed so largest is on top.
    fig_contrib = go.Figure(go.Bar(
        y=list(reversed(names)),
        x=list(reversed(pcts)),
        orientation="h",
        marker=dict(color=list(reversed(bar_colors))),
        text=[f"{p:.1f}%" for p in reversed(pcts)],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    ))
    _bar_h = max(280, 28 * len(names) + 60)
    fig_contrib.update_layout(
        **_LAYOUT_BASE,
        xaxis=dict(title="Spectral coverage (%)", fixedrange=True),
        yaxis=dict(fixedrange=True, automargin=True),
        margin=dict(l=10, r=60, t=10, b=40),
        height=_bar_h,
    )

    contrib_body = f"""
<div style="display:flex;justify-content:flex-end;gap:6px;margin-bottom:4px;">
  <button id="rgakit-contrib-csv" class="rgakit-btn-primary">Export CSV</button>
</div>
{_embed_plotly_div("rgakit-contribs-bar", _fig_to_json(fig_contrib), _bar_h)}
<script>
(function() {{
  document.getElementById('rgakit-contrib-csv').addEventListener('click', function() {{
    var csv  = {csv_content};
    var blob = new Blob([csv], {{type: 'text/csv'}});
    var url  = URL.createObjectURL(blob);
    var a    = document.createElement('a');
    a.href = url; a.download = 'rga_contributions.csv';
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
  }});
}})();
</script>"""
    sections.append(("Compound Contributions", contrib_body))

    # ------------------------------------------------------------------ #
    # Compound card (optional: shows the original molecule if provided)
    # ------------------------------------------------------------------ #
    _compound_html = ""
    if compound is not None:
        _cmpd_name = ""
        _cmpd_smiles = None
        _cmpd_formula = ""
        _cmpd_mw = ""
        if isinstance(compound, str):
            _cmpd_smiles = compound
        else:
            _cmpd_name    = getattr(compound, "name", "") or ""
            _cmpd_formula = getattr(compound, "formula", "") or ""
            _mw           = getattr(compound, "molecular_weight", None)
            _cmpd_mw      = f"{_mw:.2f} g/mol" if _mw else ""
            try:
                _cmpd_smiles = compound.smiles
            except Exception:
                pass
        _cmpd_svg = ""
        if _cmpd_smiles:
            try:
                from rdkit.Chem import MolFromSmiles
                from rdkit.Chem.Draw import MolDraw2DSVG
                _m = MolFromSmiles(_cmpd_smiles)
                if _m is not None:
                    d = MolDraw2DSVG(360, 180)
                    d.drawOptions().clearBackground = False
                    d.DrawMolecule(_m)
                    d.FinishDrawing()
                    import base64 as _b64
                    _cmpd_svg = (
                        '<img src="data:image/svg+xml;base64,'
                        + _b64.b64encode(d.GetDrawingText().encode()).decode()
                        + '">'
                    )
            except Exception:
                pass
        _details = []
        if _cmpd_formula:
            _details.append(_cmpd_formula)
        if _cmpd_mw:
            _details.append(_cmpd_mw)
        if _cmpd_smiles:
            _details.append(f'<code style="font-size:.85em">{_html_mod.escape(_cmpd_smiles)}</code>')
        _compound_html = f"""
<div class="compound-card">
  {_cmpd_svg}
  <div class="compound-card-info">
    <div class="compound-card-name">{_html_mod.escape(_cmpd_name) if _cmpd_name else 'Compound'}</div>
    <div class="compound-card-detail">{' &middot; '.join(_details)}</div>
  </div>
</div>"""

    # ------------------------------------------------------------------ #
    # Section 4 — Compound structures (detected + not detected)
    # ------------------------------------------------------------------ #

    def _det_card(n: str, pct: float, color: str, formula: str = "") -> str:
        src  = structures.get(n)
        attr = _html_mod.escape(n, quote=True)
        inner = (
            f'<img src="{src}" alt="{attr}">'
            if src else
            '<div class="detected-card-ph">?</div>'
        )
        return (
            f'<div class="detected-card"'
            f' title="{attr}" data-spec-name="{attr}">'
            f'<div class="detected-card-pct">{pct:.3g}%</div>'
            f'{inner}'
            f'<div class="detected-card-name">{n}</div>'
            f'</div>'
        )

    def _undet_card(n: str) -> str:
        src  = structures.get(n)
        attr = _html_mod.escape(n, quote=True)
        inner = (
            f'<img src="{src}" alt="{attr}">'
            if src else
            '<div class="undetected-card-ph">?</div>'
        )
        return (
            f'<div class="undetected-card" title="{attr}" data-spec-name="{attr}">'
            f'{inner}'
            f'<div class="undetected-card-name">{n}</div>'
            f'</div>'
        )

    det_cards   = "".join(
        _det_card(
            n, p, color_map[n],
            library[n].metadata.get("formula", "") if library is not None else "",
        )
        for n, p in zip(names, pcts)
    )
    undet_cards = "".join(_undet_card(n) for n in undetected)

    n_lib_total  = len(names) + len(undetected)
    undet_block  = f"""
<details class="compound-group">
  <summary>
    <span class="compound-group-label">Not detected</span>
    <span class="compound-group-count">{len(undetected)} of {n_lib_total} library compounds — weight below {threshold:.0e}</span>
  </summary>
  <div class="compound-group-body">
    <div class="undetected-grid">{undet_cards}</div>
  </div>
</details>""" if undetected else ""

    # Build suggestion cards (collapsible sub-section)
    _suggestions = getattr(result, "suggestions", None)
    _sug_block = ""
    if _suggestions:
        def _suggest_card(spec, dist: float) -> str:
            meta    = getattr(spec, "metadata", {}) or {}
            smiles  = meta.get("smiles") or ""
            mw      = meta.get("mw")
            cos     = 1.0 - dist
            name    = getattr(spec, "name", "") or smiles or "(unnamed)"
            cos_clr = "var(--clr-good)" if cos >= 0.8 else "var(--clr-warn)" if cos >= 0.6 else "var(--clr-muted)"

            img_html = ""
            if smiles:
                try:
                    from rdkit.Chem import MolFromSmiles
                    from rdkit.Chem.Draw import MolDraw2DSVG
                    import base64 as _b64
                    _m = MolFromSmiles(smiles)
                    if _m is not None:
                        d = MolDraw2DSVG(180, 100)
                        d.drawOptions().clearBackground = False
                        d.DrawMolecule(_m)
                        d.FinishDrawing()
                        img_html = (
                            '<img src="data:image/svg+xml;base64,'
                            + _b64.b64encode(d.GetDrawingText().encode()).decode()
                            + '">'
                        )
                except Exception:
                    pass

            attr = _html_mod.escape(name, quote=True)
            return (
                f'<div class="suggestion-card" data-spec-name="{attr}">'
                f'<div class="suggestion-card-cos" style="color:{cos_clr}">cos {cos:.2f}</div>'
                f'{img_html}'
                f'<div class="suggestion-card-name">{_html_mod.escape(name)}</div>'
                f'</div>'
            )

        _sug_cards = "".join(_suggest_card(s, d) for s, d in _suggestions)
        _sug_block = f"""
<details class="compound-group">
  <summary>
    <span class="compound-group-label">Suggested</span>
    <span class="compound-group-count">{len(_suggestions)} candidate{"s" if len(_suggestions) != 1 else ""} from residual — click to inspect spectrum</span>
  </summary>
  <div class="compound-group-body">
    <div class="suggestion-grid">{_sug_cards}</div>
  </div>
</details>"""

    structures_body = f"""
{_compound_html}
<details class="compound-group" open>
  <summary>
    <span class="compound-group-label">Detected</span>
    <span class="compound-group-count">{len(names)} compound{"s" if len(names) != 1 else ""}</span>
  </summary>
  <div class="compound-group-body">
    <div class="detected-grid">{det_cards}</div>
  </div>
</details>
{undet_block}
{_sug_block}"""
    sections.append(("Compound Structures", structures_body))

    # ------------------------------------------------------------------ #
    # Section 5 — Stacked per-compound spectrum + scrollable legend
    # ------------------------------------------------------------------ #
    if library is not None:
        # names/weights are already sorted descending by spectral coverage pct
        all_items = list(zip(names, weights))
        if top_n is not None:
            all_items = all_items[:top_n]
        bottom     = np.zeros(len(result.grid))

        fig3          = go.Figure()
        legend_entries = []   # (name, color, pct)
        contribs_data  = []   # full-grid contribution per trace (for JS re-stacking)

        for i, (name, weight) in enumerate(all_items):
            try:
                ref_spec = library[name]
            except KeyError:
                logger.warning("Compound %r not found in library; omitting from stacked plot.", name)
                continue
            contrib = ref_spec.on_grid(result.grid) * weight
            nz      = contrib > 0        # only pass non-zero positions to the trace
            color   = color_map[name]
            fig3.add_trace(go.Bar(
                x=result.grid[nz],
                y=contrib[nz],
                base=bottom[nz],
                name=name,
                marker_color=color,
                showlegend=False,
                customdata=contrib[nz],
                hovertemplate=f"{name}: %{{customdata:.4f}}<extra></extra>",
            ))
            contribs_data.append(contrib.tolist())
            entry_pct = (
                100.0 * float(np.sum(sc[name])) / obs_total
                if sc is not None else 100.0 * weight
            )
            legend_entries.append((name, color, entry_pct))
            bottom += contrib            # update full bottom for next trace

        # Observed outline — always last trace, always fully visible
        fig3.add_trace(go.Bar(
            x=result.grid,
            y=result.observed,
            name="Observed",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color="black", width=1.2)),
            showlegend=False,
            hoverinfo="skip",
        ))
        n_compound_traces = len(legend_entries)

        ymax3    = _ymax_without(result.observed, result.grid, exclude_mz=fit_exclude_mz)
        ymax3_js = f"[0, {ymax3}]" if ymax3 else "null"
        fig3.update_layout(
            **_LAYOUT_BASE,
            barmode="overlay",
            showlegend=False,
            hovermode="x unified",
            xaxis=dict(
                title="m/z",
                showspikes=True,
                spikemode="across",
                spikecolor="#aaa",
                spikethickness=1,
                spikedash="dot",
            ),
            yaxis=dict(title="Relative intensity", **(dict(range=[0, ymax3]) if ymax3 else {})),
            margin=dict(l=60, r=20, t=30, b=50),
            height=_SPECTRUM_PLOT_HEIGHT,
        )

        # Build scrollable legend items
        legend_items_html = "".join(
            f'<div class="legend-item" data-idx="{i}" data-name="{name}"'
            f' onmouseenter="stkHL({i},{n_compound_traces})"'
            f' onmouseleave="stkReset({n_compound_traces})"'
            f' onclick="stkIsolate(this,{i},{n_compound_traces})">'
            f'<span class="legend-swatch" style="background:{color}"></span>'
            f'<span class="legend-label" title="{name}">{name}</span>'
            f'<span class="legend-pct">{pct:.1f}%</span>'
            f'</div>\n'
            for i, (name, color, pct) in enumerate(legend_entries)
        )

        stacked_toolbar = f"""
<div class="stacked-toolbar">
  <button id="stkLogBtn" class="rgakit-btn" onclick="stkToggleLog()">Log scale</button>
  <button class="rgakit-btn" onclick="stkIsolateReset({n_compound_traces})">Reset view</button>
  <input type="text" id="stkFilter" placeholder="Filter compounds…"
         oninput="stkFilterLegend(this.value)">
</div>"""

        section4_body = f"""
{stacked_toolbar}
<div class="stacked-wrap">
  <div class="stacked-plot-col" style="flex:1 1 0;min-width:0;">
    <div id="rgakit-stacked" style="width:100%;height:{_SPECTRUM_PLOT_HEIGHT}px;"></div>
  </div>
  <div class="stacked-legend-col">
    <div class="stacked-legend-header">
      Compound <span class="stacked-legend-hint">— hover / click</span>
    </div>
    <div class="stacked-legend-scroll" id="stkLegendScroll">{legend_items_html}<div class="legend-item" style="cursor:default;border-top:1px solid #eee;margin-top:4px;pointer-events:none;">
      <span class="legend-swatch" style="background:transparent;border:1.5px solid #333;box-sizing:border-box;"></span>
      <span class="legend-label" style="color:var(--clr-nav-text);">Observed</span>
    </div></div>
  </div>
</div>
<script>
(function() {{
  var GD           = 'rgakit-stacked';
  var LOG          = false;
  var SELECTED     = new Set();   // indices of clicked (pinned) compounds
  var legendScroll = document.getElementById('stkLegendScroll');

  var GRID     = {json.dumps(result.grid.tolist())};
  var CONTRIBS = {json.dumps(contribs_data)};   // [n_compounds][n_grid] raw contributions

  var fig = {_fig_to_json(fig3)};
  fig.layout.height = {_SPECTRUM_PLOT_HEIGHT};
  Plotly.newPlot(GD, fig.data, fig.layout, {{responsive:true}});

  var traceIdx = Array.from({{length: {n_compound_traces}}}, function(_,k){{return k;}});

  // Recompute x/y/base for every trace given the current SELECTED set and
  // update Plotly in one restyle call.  When SELECTED is empty all traces
  // are restored to their original (fully-stacked) values.
  function restack() {{
    var n       = CONTRIBS.length;
    var gLen    = GRID.length;
    var newX    = [], newY = [], newBase = [], vis = [], ops = [];
    var bottom  = new Array(gLen).fill(0);
    var inSel   = SELECTED.size > 0;

    for (var i = 0; i < n; i++) {{
      var c    = CONTRIBS[i];
      var show = !inSel || SELECTED.has(i);
      vis.push(show);
      ops.push(1.0);
      if (show) {{
        var x = [], y = [], base = [];
        for (var j = 0; j < gLen; j++) {{
          if (c[j] > 0) {{ x.push(GRID[j]); y.push(c[j]); base.push(bottom[j]); }}
        }}
        newX.push(x); newY.push(y); newBase.push(base);
        for (var j = 0; j < gLen; j++) bottom[j] += c[j];
      }} else {{
        newX.push([]); newY.push([]); newBase.push([]);
      }}
    }}
    Plotly.restyle(GD, {{ x: newX, y: newY, base: newBase, visible: vis, opacity: ops }}, traceIdx);
    legendScroll.classList.toggle('isolating', SELECTED.size > 0);
    document.querySelectorAll('.legend-item').forEach(function(el, j) {{
      // j >= n is the "Observed" footer row — always keep it at full opacity
      el.style.opacity = (j >= n || !inSel || SELECTED.has(j)) ? '1' : '0.35';
    }});
  }}

  function stkHL(idx, n) {{
    if (SELECTED.size > 0) return;
    var ops = [];
    for (var i = 0; i < n; i++) ops.push(i === idx ? 1.0 : 0.07);
    Plotly.restyle(GD, {{'opacity': ops}}, traceIdx);
    document.querySelectorAll('.legend-item').forEach(function(el, j) {{
      // j >= n is the "Observed" footer row — always keep it at full opacity
      el.style.opacity = (j >= n || j === idx) ? '1' : '0.35';
    }});
  }}

  function stkReset(n) {{
    if (SELECTED.size > 0) return;
    Plotly.restyle(GD, {{
      visible: Array(n).fill(true),
      opacity: Array(n).fill(1.0),
    }}, traceIdx);
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      el.style.opacity = '1';
    }});
  }}

  function stkIsolate(el, idx, n) {{
    if (SELECTED.has(idx)) {{
      SELECTED.delete(idx);
      el.classList.remove('isolated');
    }} else {{
      SELECTED.add(idx);
      el.classList.add('isolated');
    }}
    restack();
  }}

  function stkIsolateReset(n) {{
    SELECTED.clear();
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      el.classList.remove('isolated');
      el.style.opacity = '1';
    }});
    restack();
  }}

  function stkToggleLog() {{
    LOG = !LOG;
    var btn = document.getElementById('stkLogBtn');
    btn.classList.toggle('active', LOG);
    btn.textContent = LOG ? 'Linear scale' : 'Log scale';
    if (LOG) {{
      Plotly.relayout(GD, {{'yaxis.type': 'log', 'yaxis.autorange': true}});
    }} else {{
      Plotly.relayout(GD, {{'yaxis.type': 'linear', 'yaxis.range': {ymax3_js}}});
    }}
  }}

  function stkFilterLegend(q) {{
    q = q.toLowerCase();
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      var name = (el.getAttribute('data-name') || '').toLowerCase();
      el.style.display = name.includes(q) ? '' : 'none';
    }});
  }}

  window.stkHL           = stkHL;
  window.stkReset        = stkReset;
  window.stkIsolate      = stkIsolate;
  window.stkIsolateReset = stkIsolateReset;
  window.stkToggleLog    = stkToggleLog;
  window.stkFilterLegend = stkFilterLegend;
}})();

</script>"""

        heading = f"Stacked Contributions — top {top_n}" if top_n else "Stacked Contributions"
        sections.append((heading, section4_body))

    # ------------------------------------------------------------------ #
    # Section 6 — Time-resolved contributions (optional)
    # ------------------------------------------------------------------ #
    if time_result is not None:
        # Use the same compound order as the rest of the report (names,
        # sorted by spectral coverage %).  Compounds present in the time
        # series but not in the averaged fit go at the end.
        _ts_weights = time_result.contributions()
        _ts_order   = [n for n in names if n in _ts_weights]
        _ts_extra   = [n for n in _ts_weights if n not in set(names)]
        ts_items    = [(n, _ts_weights[n]) for n in _ts_order + _ts_extra]
        if top_n is not None:
            ts_items = ts_items[:top_n]

        # Reuse the bar/stacked palette so each compound keeps the same colour
        # across all sections; only assign a new colour for compounds that appear
        # in the time series but not in the averaged fit.
        ts_color_map = {n: color_map.get(n, p)
                        for n, p in _make_color_map([n for n, _ in ts_items]).items()}

        # Scale each compound's weight by its spectral area on the fit grid so
        # that relative percentages match the spectral-coverage definition used
        # in the bar chart (sum(ref_on_grid * weight) / sum(observed)).
        ts_areas: dict[str, float] = {}
        if library is not None:
            for name, _ in ts_items:
                try:
                    ts_areas[name] = float(np.sum(library[name].on_grid(result.grid)))
                except KeyError:
                    ts_areas[name] = 1.0
        else:
            for name, _ in ts_items:
                ts_areas[name] = 1.0

        weighted = {name: w * ts_areas[name] for name, w in ts_items}

        # Convert NNLS weights to absolute pressure units.
        # fit_time_series normalises each scan by its max intensity, so the
        # correct back-conversion is:
        #     partial_pressure_j(t) = w_j(t) × area_j × max_per_scan(t)
        # where max_per_scan(t) is the maximum raw intensity on the fit grid
        # at time t.  Using stack.pressure.sum() instead would inflate values
        # because the weights are relative to the max, not the sum.
        if stack is not None:
            _mz_idx      = np.array([np.searchsorted(stack.mz, m) for m in time_result.grid])
            _mz_idx      = _mz_idx[_mz_idx < len(stack.mz)]
            max_per_scan = stack.pressure[:, _mz_idx].max(axis=1)  # (n_times,)
            if len(max_per_scan) != len(time_result.time):
                max_per_scan = np.interp(
                    time_result.time, stack.time, max_per_scan
                )
            scale       = max_per_scan
            yaxis_title = "Partial pressure (a.u.)"
            hover_fmt   = "%{y:.3e}"
        else:
            total_w    = sum(weighted.values())
            safe_total = np.where(total_w > 0, total_w, 1.0)
            scale       = 100.0 / safe_total
            yaxis_title = "Relative composition (%)"
            hover_fmt   = "%{y:.1f}%"

        fig_ts = go.Figure()
        for name, w in ts_items:
            color = ts_color_map[name]
            y_vals = weighted[name] * scale
            fig_ts.add_trace(go.Scatter(
                x=time_result.time, y=y_vals,
                name=name,
                mode="lines",
                stackgroup="one",
                fillcolor=color,
                line=dict(color=color, width=0.5),
                hovertemplate=f"{name}: {hover_fmt}<extra></extra>",
            ))
        # Shutter markers (reuse stack open/close times if available)
        ts_shapes = []
        shutter_lk = dict(color="red", width=1.5, dash="dash")
        if stack is not None:
            open_t, close_t = stack.open_window
            if open_t  is not None:
                ts_shapes.append(dict(type="line", x0=open_t,  x1=open_t,  y0=0, y1=1,
                                      yref="paper", line=shutter_lk))
            if close_t is not None:
                ts_shapes.append(dict(type="line", x0=close_t, x1=close_t, y0=0, y1=1,
                                      yref="paper", line=shutter_lk))

        fig_ts.update_layout(
            **_LAYOUT_BASE,
            hovermode="x unified",
            xaxis_title="Time (s)",
            yaxis=dict(title=yaxis_title),
            height=_SPECTRUM_PLOT_HEIGHT,
            margin=dict(l=60, r=40, t=30, b=50),
            shapes=ts_shapes,
        )
        ts_note_body = (
            'Each compound\'s NNLS weight is converted to partial pressure by '
            'multiplying by its spectral area and the per-scan normalisation factor '
            '(max intensity on the fit grid). '
            + ('Dashed red lines mark beam-on / beam-off events.' if ts_shapes else '')
        )
        ts_note = f'<p style="font-size:12px;color:#aaa;margin:0 0 8px;">{ts_note_body}</p>'
        sections.append((
            "Time-Resolved Composition",
            ts_note + _embed_plotly_div("rgakit-timeseries", _fig_to_json(fig_ts), _SPECTRUM_PLOT_HEIGHT),
        ))

    # ------------------------------------------------------------------ #
    # Section 7 — NMF Decomposition (optional)
    # ------------------------------------------------------------------ #
    if decomposition is not None:
        nmf_r = decomposition
        # Time-profile line plot
        fig_nmf = go.Figure()
        for i in range(nmf_r.n_components):
            comp  = nmf_r.components[i]
            pct   = 100 * nmf_r.explained[i]
            label = comp.name
            if nmf_r.matches and nmf_r.matches[i]:
                best = nmf_r.matches[i][0]
                label = f"{best[0]} (cos={best[1]:.2f})"
            fig_nmf.add_trace(go.Scatter(
                x=nmf_r.time, y=nmf_r.profiles[:, i],
                name=f"{label} [{pct:.1f}%]",
                mode="lines", line=dict(width=1.5),
            ))

        _nmf_shapes = []
        if stack is not None:
            shutter_lk = dict(color="red", width=1.5, dash="dash")
            try:
                open_t, close_t = stack.open_window
                if open_t is not None:
                    _nmf_shapes.append(dict(type="line", x0=open_t, x1=open_t,
                                            y0=0, y1=1, yref="paper", line=shutter_lk))
                if close_t is not None:
                    _nmf_shapes.append(dict(type="line", x0=close_t, x1=close_t,
                                            y0=0, y1=1, yref="paper", line=shutter_lk))
            except Exception:
                pass

        fig_nmf.update_layout(
            **_LAYOUT_BASE,
            hovermode="x unified",
            xaxis_title="Time (s)",
            yaxis=dict(title="Component intensity (a.u.)"),
            height=_SPECTRUM_PLOT_HEIGHT,
            margin=dict(l=60, r=40, t=30, b=50),
            shapes=_nmf_shapes,
        )

        # Component summary table
        _nmf_rows = ""
        for i in range(nmf_r.n_components):
            comp = nmf_r.components[i]
            pct  = 100 * nmf_r.explained[i]
            match_str = ""
            if nmf_r.matches and nmf_r.matches[i]:
                best = nmf_r.matches[i][0]
                match_str = f'{_html_mod.escape(best[0])} <span style="color:var(--clr-muted)">(cos={best[1]:.2f})</span>'
            _nmf_rows += (
                f'<tr>'
                f'<td style="text-align:center;font-weight:600">{i+1}</td>'
                f'<td style="text-align:right">{pct:.1f}%</td>'
                f'<td>{match_str or "<span style=\"color:#ccc\">no match</span>"}</td>'
                f'</tr>'
            )

        nmf_body = f"""
<p style="font-size:0.85em;color:var(--clr-muted);margin:0 0 10px;">
  Non-negative Matrix Factorisation (NMF) decomposes the time × m/z pressure
  matrix into independent components without using a reference library.
  Each component's resolved spectrum is matched against the database for
  identification.  Reconstruction error: {nmf_r.reconstruction_error:.4f}.
</p>
{_embed_plotly_div("rgakit-nmf-profiles", _fig_to_json(fig_nmf), _SPECTRUM_PLOT_HEIGHT)}
<table class="fit-info-table" style="margin-top:12px;width:100%;">
  <thead><tr>
    <th style="text-align:center;padding:4px 10px;">#</th>
    <th style="text-align:right;padding:4px 10px;">Signal</th>
    <th style="padding:4px 10px;">Best database match</th>
  </tr></thead>
  <tbody>{_nmf_rows}</tbody>
</table>"""
        sections.append(("NMF Decomposition", nmf_body))

    # ------------------------------------------------------------------ #
    # Assemble HTML page
    # ------------------------------------------------------------------ #

    # Sample name from spectrum
    sample_name = ""
    if spectrum is not None:
        sample_name = getattr(spectrum, "name", "") or ""

    # Compact fit-summary table (spectrum metadata + fit parameters)
    _fit_info_pairs: list[tuple[str, str]] = []
    if spectrum is not None:
        meta = getattr(spectrum, "metadata", {}) or {}
        x_val = meta.get("x")
        y_val = meta.get("y")
        if x_val is not None and y_val is not None:
            _fit_info_pairs.append(("Position", f"x={x_val:.2f} mm, y={y_val:.2f} mm"))
        pd_val = meta.get("pd_ua")
        if pd_val is not None:
            _fit_info_pairs.append(("Photocurrent", f"{pd_val:.2f} µA"))
        n_scans = meta.get("n_open_scans")
        if n_scans is not None:
            _fit_info_pairs.append(("Open scans", str(n_scans)))
        if meta.get("background_correct"):
            _fit_info_pairs.append(("Background", "corrected"))

    # Only show filter params here; solver/condition/library are in the header metrics.
    fp = getattr(result, "fit_params", {}) or {}
    if "mz_min" in fp:
        _fit_info_pairs.append(("m/z min", str(fp["mz_min"])))
    if "mz_max" in fp:
        _fit_info_pairs.append(("m/z max", str(fp["mz_max"])))
    if "exclude_mz" in fp:
        _fit_info_pairs.append(("Excluded m/z", ", ".join(str(m) for m in fp["exclude_mz"])))
    if "min_intensity" in fp:
        _fit_info_pairs.append(("Min intensity", f"{fp['min_intensity']*100:.1f}%"))

    if _fit_info_pairs:
        _rows = ""
        for i in range(0, len(_fit_info_pairs), 2):
            pair_a = _fit_info_pairs[i]
            pair_b = _fit_info_pairs[i + 1] if i + 1 < len(_fit_info_pairs) else None
            _rows += "<tr>"
            _rows += f'<td class="fik">{pair_a[0]}</td><td class="fiv">{pair_a[1]}</td>'
            if pair_b:
                _rows += f'<td class="fik" style="padding-left:20px">{pair_b[0]}</td><td class="fiv">{pair_b[1]}</td>'
            _rows += "</tr>"
        fit_summary_html = f'<table class="fit-info-table"><tbody>{_rows}</tbody></table>'
    else:
        fit_summary_html = ""

    # Page title + compact metrics card
    header_card = f"""
<div class="page-title">
  <h1>{title}</h1>
  {'<p class="header-sample">' + sample_name + '</p>' if sample_name else ''}
</div>
<div class="header-card">
  <div class="header-metrics">
    <div class="hmetric" title="Normalized residual: ‖observed − fitted‖₂ / ‖observed‖₂. Lower is better; 0 = perfect fit. Green &lt; 0.05, amber 0.05–0.15, red &gt; 0.15.">
      <span class="hmetric-val" style="color:{nr_color};">{norm_residual:.4f}</span>
      <span class="hmetric-lbl">Norm. Residual</span>
    </div>
    <div class="hmetric" title="Coefficient of determination: 1 − SS_res / SS_tot, where SS_tot = Σ(yᵢ − ȳ)². Values near 1 indicate a good fit. Green ≥ 0.95, amber 0.80–0.95, red &lt; 0.80.">
      <span class="hmetric-val" style="color:{r2_color};">{r2:.3f}</span>
      <span class="hmetric-lbl">R²</span>
    </div>
    <div class="hmetric" title="Signal coverage: fraction of total observed intensity captured by the fit — Σ min(fitted, observed) / Σ observed × 100. Green ≥ 95%, amber 80–95%, red &lt; 80%.">
      <span class="hmetric-val" style="color:{cov_color};">{coverage:.1f}%</span>
      <span class="hmetric-lbl">Signal Coverage</span>
    </div>
    <div class="hmetric" title="Number of library compounds detected (above threshold) vs. total searched.">
      <span class="hmetric-val">{len(contribs)}<span style="font-weight:400;font-size:.65em;color:#999"> / {_n_searched}</span></span>
      <span class="hmetric-lbl">Compounds</span>
    </div>
    <div class="hmetric" title="Number of m/z channels included in the fit grid.">
      <span class="hmetric-val">{len(result.grid)}</span>
      <span class="hmetric-lbl">m/z Grid</span>
    </div>
    <div class="hmetric" title="Fitting solver used. OMP = orthogonal matching pursuit (sparse); NNLS = non-negative least squares; LASSO = L1-regularised.">
      <span class="hmetric-val">{_method_disp}</span>
      <span class="hmetric-lbl">Solver</span>
    </div>
    <div class="hmetric" title="Condition number of the design matrix. High values (&gt; 1000) indicate near-collinear reference spectra that the solver cannot reliably distinguish.">
      <span class="hmetric-val" style="color:{_cond_color};">{_cond_str}</span>
      <span class="hmetric-lbl">Condition №</span>
    </div>
  </div>
  {_coll_block}
  {fit_summary_html}
</div>"""

    import re as _re
    def _sec_id(h: str) -> str:
        return "sec-" + _re.sub(r"[^a-z0-9]+", "-", h.lower()).strip("-")

    section_html = "\n".join(
        f'<div class="section" id="{_sec_id(heading)}"><h2>{heading}</h2>{div}</div>'
        for heading, div in sections
    )

    nav_links = "".join(
        f'<a href="#{_sec_id(h)}">{h}</a>'
        for h, _ in sections
    )
    _github_icon = (
        '<svg class="rgakit-github-icon" viewBox="0 0 16 16"'
        ' xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59'
        '.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94'
        '-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82'
        '.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95'
        ' 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82'
        '.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82'
        '.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95'
        '.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38'
        'A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>'
        '</svg>'
    )
    nav_html = (
        f'<nav class="rgakit-nav">{nav_links}'
        f'<a class="rgakit-nav-brand" href="{_REPO_URL}"'
        f' target="_blank" rel="noopener" title="rgakit on GitHub">'
        f'<span class="rgakit-nav-brand-row">'
        f'{_github_icon}'
        f'<span class="rgakit-nav-brand-name">rgakit</span>'
        f'<span class="rgakit-nav-brand-ver">v{_rgakit_ver}</span>'
        f'</span>'
        f'<span class="rgakit-nav-ts">Created at: {_report_ts}</span>'
        f'</a>'
        f'</nav>'
    )

    # Metadata section (collapsible) — only when FitResult carries metadata
    meta_html = ""
    meta = getattr(result, "metadata", None)
    if meta:
        rows = "".join(
            f"<tr><td style='padding:3px 12px 3px 0;color:#888;font-size:12px;'>{k}</td>"
            f"<td style='font-size:12px;'>{v}</td></tr>"
            for k, v in meta.items()
        )
        meta_html = f"""
<details style="margin-bottom:28px;">
  <summary style="cursor:pointer;font-size:0.9em;color:#666;padding:6px 10px;
      background:#f0f4ff;border-radius:6px;border:1px solid #dde4f5;
      list-style:none;display:flex;align-items:center;gap:6px;">
    <span style="font-size:0.8em;color:var(--clr-primary);">&#9658;</span>
    <span>Sample metadata</span>
    <span style="color:#bbb;font-size:0.85em;font-weight:normal;">(click to expand)</span>
  </summary>
  <div class="section" style="margin-top:8px;">
    <table>{rows}</table>
  </div>
</details>"""

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script>{get_plotlyjs()}</script>
  <style>{_PAGE_CSS}</style>
</head>
<body>
  {nav_html}
  <div class="page-wrap">
    {header_card}
    {meta_html}
    {section_html}
  </div>
  <script>
  (function() {{
    var navLinks = {{}};
    document.querySelectorAll('.rgakit-nav > a:not(.rgakit-nav-brand)').forEach(function(a) {{
      navLinks[a.getAttribute('href').slice(1)] = a;
    }});
    var secs = Array.from(document.querySelectorAll('.section[id]'));
    var NAV_H = 48; // approximate sticky nav height in px
    function updateActive() {{
      var scrollY = window.scrollY + NAV_H + 8;
      var current = null;
      for (var i = 0; i < secs.length; i++) {{
        if (secs[i].offsetTop <= scrollY) current = secs[i].id;
      }}
      Object.values(navLinks).forEach(function(a) {{ a.classList.remove('active'); }});
      if (current && navLinks[current]) navLinks[current].classList.add('active');
    }}
    window.addEventListener('scroll', updateActive, {{ passive: true }});
    updateActive();
  }})();
  </script>

  <!-- Compound spectrum modal -->
  <div id="rgakit-spec-modal" class="rgakit-modal-overlay">
    <div class="rgakit-modal-box">
      <div class="rgakit-modal-header">
        <div class="rgakit-modal-info">
          <div id="rgakit-modal-name"    class="rgakit-modal-name"></div>
          <div id="rgakit-modal-formula" class="rgakit-modal-formula"></div>
          <div id="rgakit-modal-meta" class="rgakit-modal-meta"></div>
        </div>
        <div id="rgakit-modal-struct-wrap" class="rgakit-modal-struct-wrap" style="display:none;">
          <button id="rgakit-modal-copy-img" class="rgakit-modal-copy-icon" title="Copy structure to clipboard">
            <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V2zm2-1a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1V2a1 1 0 0 0-1-1H6zM2 5a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-1h1v1a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h1v1H2z"/></svg>
          </button>
          <img id="rgakit-modal-struct" class="rgakit-modal-struct" src="" alt="">
        </div>
        <button class="rgakit-modal-close" id="rgakit-modal-close" aria-label="Close">&times;</button>
      </div>
      <div class="rgakit-modal-plot-wrap">
        <div id="rgakit-modal-plot" style="width:100%;height:100%;"></div>
      </div>
      <div id="rgakit-modal-collinear" class="rgakit-modal-collinear" style="display:none;">
        <div class="rgakit-modal-collinear-title">⚠ Similar library spectra — may alias with this compound</div>
        <div id="rgakit-modal-coll-list"></div>
      </div>
    </div>
  </div>
  <script>
  (function() {{
    var SPECS    = {_lib_spectra_json};
    var modal    = document.getElementById('rgakit-spec-modal');
    var plotEl   = document.getElementById('rgakit-modal-plot');
    var structEl = document.getElementById('rgakit-modal-struct');
    var collDiv  = document.getElementById('rgakit-modal-collinear');
    var collList = document.getElementById('rgakit-modal-coll-list');
    var structWrap = document.getElementById('rgakit-modal-struct-wrap');
    var copyImg    = document.getElementById('rgakit-modal-copy-img');
    var rendered   = false;

    function formulaToHtml(f) {{
      return f.replace(/(\\d+)/g, '<sub>$1</sub>');
    }}

    function openSpecModal(name) {{
      var spec = SPECS[name];
      if (!spec) return;

      document.getElementById('rgakit-modal-name').textContent = name;
      document.getElementById('rgakit-modal-formula').innerHTML =
        spec.formula ? formulaToHtml(spec.formula) : '';

      var parts = [];
      if (spec.cas)    parts.push('CAS\u00a0' + spec.cas);
      if (spec.mw)     parts.push('MW\u00a0'  + spec.mw);
      if (spec.smiles) parts.push(spec.smiles);
      document.getElementById('rgakit-modal-meta').textContent =
        parts.join('\u2002\u00b7\u2002');

      if (spec.svg) {{
        structEl.src = spec.svg;
        structWrap.style.display = '';
      }} else {{
        structWrap.style.display = 'none';
      }}

      // Collinearity section
      var coll = spec.collinear_with || [];
      if (coll.length > 0) {{
        collList.innerHTML = coll
          .sort(function(a,b){{ return b.cosine - a.cosine; }})
          .map(function(c) {{
            return '<span class="rgakit-modal-coll-pill" data-coll-name="' +
                   c.name.replace(/"/g, '&quot;') + '">' +
                   c.name + ' <span style="opacity:.55;font-size:.88em">cos\u2009=\u2009' +
                   c.cosine.toFixed(2) + '</span></span>';
          }}).join(' ');
        collList.querySelectorAll('[data-coll-name]').forEach(function(el) {{
          el.addEventListener('click', function() {{
            openSpecModal(this.dataset.collName);
          }});
        }});
        collDiv.style.display = '';
      }} else {{
        collDiv.style.display = 'none';
      }}

      var plotHeight = Math.max(180,
        document.querySelector('.rgakit-modal-plot-wrap').clientHeight - 16);
      var trace = {{
        type: 'bar',
        x: spec.mz,
        y: spec.intensity,
        marker: {{ color: 'rgba(30,30,30,0.82)' }},
        hovertemplate: 'm/z\u00a0%{{x}}:  %{{y:.3f}}<extra></extra>',
      }};
      var layout = {{
        template: 'plotly_white',
        hovermode: 'x unified',
        xaxis: {{ title: {{ text: 'm/z' }}, fixedrange: false,
                  showspikes: true, spikemode: 'across',
                  spikecolor: '#aaa', spikethickness: 1, spikesnap: 'cursor' }},
        yaxis: {{ title: {{ text: 'Rel. intensity' }}, range: [0, 1.08], fixedrange: false }},
        margin: {{ l: 55, r: 20, t: 10, b: 50 }},
        height: plotHeight,
        autosize: true,
      }};
      if (rendered) {{
        Plotly.react(plotEl, [trace], layout);
      }} else {{
        Plotly.newPlot(plotEl, [trace], layout, {{responsive: true}});
        rendered = true;
      }}
      modal.classList.add('open');
    }}

    function closeSpecModal() {{
      modal.classList.remove('open');
    }}

    document.getElementById('rgakit-modal-close').addEventListener('click', closeSpecModal);
    modal.addEventListener('click', function(e) {{
      if (e.target === modal) closeSpecModal();
    }});
    document.addEventListener('keydown', function(e) {{
      if (e.key === 'Escape') closeSpecModal();
    }});

    document.querySelectorAll('[data-spec-name]').forEach(function(el) {{
      el.addEventListener('click', function() {{
        openSpecModal(this.dataset.specName);
      }});
    }});

    // Expose globally so inline onclick handlers (collinearity pills, etc.) can call it
    copyImg.addEventListener('click', function(e) {{
      e.stopPropagation();
      var src = structEl.src;
      if (!src) return;
      var img = new Image();
      img.onload = function() {{
        var canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth * 2;
        canvas.height = img.naturalHeight * 2;
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(function(blob) {{
          navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]).then(function() {{
            copyImg.style.color = 'var(--clr-good)';
            setTimeout(function() {{ copyImg.style.color = ''; }}, 1500);
          }});
        }}, 'image/png');
      }};
      img.src = src;
    }});

    window.openSpecModal = openSpecModal;
  }})();
  </script>
</body>
</html>"""

    dest = Path(output_path)
    dest.write_text(page)
    logger.info("Report saved to %s", dest.resolve())
    return dest
