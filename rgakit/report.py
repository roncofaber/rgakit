"""
report.py
---------
Plotting helpers and interactive HTML report generation for FitResult.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from .result import FitResult

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_SPECTRUM_PLOT_HEIGHT   = 420   # px — fixed-height spectrum figures
_RESIDUAL_PLOT_HEIGHT   = 220   # px — residual panel (shorter)
_CONTRIB_HEIGHT_PER_ROW = 32    # px per compound in the contributions chart
_CONTRIB_HEIGHT_MIN     = 360   # px — minimum height for contributions chart
_CONTRIB_MARGIN_LEFT    = 180   # px — left margin to accommodate long names


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

    contribs = {k: v for k, v in result.contributions.items() if v >= threshold}
    total    = sum(contribs.values()) or 1.0
    names    = list(contribs.keys())
    pcts     = [100 * v / total for v in contribs.values()]

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
            warnings.warn(
                f"Compound '{name}' not found in library; omitting from stacked plot.",
                stacklevel=2,
            )
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
* { box-sizing: border-box; }
body {
    font-family: Arial, Helvetica, sans-serif;
    background: #f5f5f5;
    margin: 0;
    padding: 24px;
    color: #222;
}
.page-wrap {
    max-width: 1200px;
    margin: 0 auto;
}
.report-header {
    text-align: center;
    margin-bottom: 20px;
}
.report-header h1 { margin: 0 0 6px; font-size: 1.8em; }
.report-header .subtitle { color: #666; font-size: 0.95em; margin: 0 0 10px; }
.report-header .meta-row {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 8px; margin: 8px 0;
}
.meta-badge {
    background: #eef2fb; border-radius: 20px; padding: 3px 12px;
    font-size: 0.8em; color: #555;
}
.top-contribs {
    display: flex; flex-wrap: wrap; justify-content: center;
    align-items: center; gap: 6px; margin-top: 10px;
}
.top-contribs-label { font-size: 0.8em; color: #888; }
.contrib-pill {
    border-radius: 20px; padding: 3px 11px;
    font-size: 0.8em; color: #fff; font-weight: 500;
    opacity: 0.9;
}

/* Summary card */
.summary-card {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    margin-bottom: 28px;
}
.stat-chip {
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
    padding: 10px 18px;
    text-align: center;
    min-width: 120px;
}
.stat-chip .stat-value { font-size: 1.4em; font-weight: bold; color: #1f77b4; }
.stat-chip .stat-label { font-size: 0.75em; color: #888; margin-top: 2px; }

.section {
    background: #fff;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,.12);
    padding: 20px 24px;
    margin-bottom: 28px;
}
.section h2 {
    margin: 0 0 14px;
    font-size: 1.1em;
    color: #444;
    border-bottom: 1px solid #eee;
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
.stacked-toolbar button {
    padding: 5px 14px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: #f8f8f8;
    cursor: pointer;
    font-size: 12px;
    transition: background 0.15s;
}
.stacked-toolbar button:hover { background: #e8eef8; border-color: #1f77b4; }
.stacked-toolbar button.active { background: #1f77b4; color: #fff; border-color: #1f77b4; }
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
    font-size: 0.8em; font-weight: bold; color: #555;
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
.legend-item.isolated { outline: 2px solid #1f77b4; border-radius: 2px; }
.legend-swatch {
    display: inline-block; width: 13px; height: 13px;
    border-radius: 2px; flex-shrink: 0; margin-top: 2px;
}
.legend-label { flex: 1; white-space: normal; word-break: break-word; line-height: 1.3; }
.legend-pct { color: #999; font-size: 0.88em; flex-shrink: 0; padding-top: 1px; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plotly_palette():
    try:
        import plotly.express as px
        return (px.colors.qualitative.Plotly
                + px.colors.qualitative.D3
                + px.colors.qualitative.Pastel)
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
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    result:      FitResult,
    library=None,
    output_path: str | Path = "rga_fit_report.html",
    title:       str        = "RGA Fit Report",
    threshold:   float      = 1e-4,
    top_n:       int | None = None,
    spectrum=None,
) -> Path:
    """
    Generate a self-contained, scrollable HTML report from a FitResult.

    Produces interactive sections:
      1. Summary card (residual, coverage, R², grid points, compound count)
      2. Observed vs Fitted Spectrum
      3. Residual Spectrum (observed − fitted)
      4. Compound Contributions (%)
      5. Stacked Contributions — all compounds above threshold with scrollable,
         hover-to-highlight, click-to-isolate legend, log-scale toggle,
         legend filter, and CSV download (requires *library*)

    Requires plotly: ``pip install plotly`` (or ``pip install rgakit[report]``).

    Parameters
    ----------
    result      : FitResult from SpectraLibrary.fit()
    library     : SpectraLibrary used for fitting; enables stacked-spectrum section
    output_path : destination .html file path
    title       : page title shown at the top of the report
    threshold   : minimum weight for a compound to appear
    top_n       : limit the stacked section to the top-N contributors;
                  ``None`` (default) shows all compounds above *threshold*
    spectrum    : the MassSpectrum that was fitted; when provided, its metadata
                  (x, y, pd_ua, n_open_scans) is shown in the report header

    Returns
    -------
    Path to the saved HTML file.
    """
    try:
        import plotly.graph_objects as go
        from plotly.offline import get_plotlyjs
    except ImportError:
        raise ImportError(
            "plotly is required for HTML reports:\n"
            "  pip install plotly\n"
            "  or: pip install 'rgakit[report]'"
        )

    contribs = {k: v for k, v in result.contributions.items() if v >= threshold}
    total    = sum(contribs.values()) or 1.0
    names    = list(contribs.keys())
    weights  = list(contribs.values())
    pcts     = [100 * w / total for w in weights]

    color_map = _make_color_map(names)

    sections: list[tuple[str, str]] = []   # (heading, div_html)

    # ------------------------------------------------------------------ #
    # Summary metrics
    # ------------------------------------------------------------------ #
    ss_total = float(np.sum(result.observed ** 2))
    ss_res   = float(result.residual ** 2)
    r2       = max(0.0, 1.0 - ss_res / ss_total) if ss_total > 0 else 0.0
    coverage = float(np.sum(np.minimum(result.fitted, result.observed))) / (
        float(np.sum(result.observed)) or 1.0
    ) * 100

    chips = [
        (f"{result.residual:.4f}", "Residual (L2)"),
        (f"{r2:.4f}",             "R²"),
        (f"{coverage:.1f}%",      "Signal coverage"),
        (str(len(result.grid)),   "m/z grid points"),
        (str(len(contribs)),      "Compounds fitted"),
    ]
    summary_card = '<div class="summary-card">\n' + "".join(
        f'<div class="stat-chip"><div class="stat-value">{val}</div>'
        f'<div class="stat-label">{lbl}</div></div>\n'
        for val, lbl in chips
    ) + '</div>\n'

    # ------------------------------------------------------------------ #
    # Section 1 — Observed vs Fitted
    # ------------------------------------------------------------------ #
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=result.grid, y=result.observed,
        name="Observed",
        marker_color="rgba(255,127,14,0.75)",
        hovertemplate="%{y:.4f}<extra>Observed</extra>",
    ))
    fig1.add_trace(go.Bar(
        x=result.grid, y=result.fitted,
        name="Fitted",
        marker_color="rgba(31,119,180,0.75)",
        hovertemplate="%{y:.4f}<extra>Fitted</extra>",
    ))
    ymax1 = _ymax_without(np.maximum(result.observed, result.fitted), result.grid)
    fig1.update_layout(
        **_LAYOUT_BASE,
        barmode="overlay",
        xaxis_title="m/z",
        yaxis_title="Relative intensity",
        yaxis=dict(range=[0, ymax1]) if ymax1 else {},
        margin=dict(l=60, r=40, t=30, b=50),
        height=_SPECTRUM_PLOT_HEIGHT,
    )
    sections.append((
        "Observed vs Fitted Spectrum",
        _embed_plotly_div("rgakit-overlay", _fig_to_json(fig1), _SPECTRUM_PLOT_HEIGHT),
    ))

    # ------------------------------------------------------------------ #
    # Section 2 — Residual spectrum
    # ------------------------------------------------------------------ #
    residual_vals = result.observed - result.fitted
    pos_mask = residual_vals >= 0
    neg_mask = ~pos_mask

    fig_res = go.Figure()
    # Positive residual (unexplained signal) — red
    fig_res.add_trace(go.Bar(
        x=result.grid[pos_mask],
        y=residual_vals[pos_mask],
        name="Unexplained",
        marker_color="rgba(214,39,40,0.7)",
        hovertemplate="%{y:.4f}<extra>Unexplained</extra>",
    ))
    # Negative residual (over-fitted) — blue
    fig_res.add_trace(go.Bar(
        x=result.grid[neg_mask],
        y=residual_vals[neg_mask],
        name="Over-fitted",
        marker_color="rgba(31,119,180,0.55)",
        hovertemplate="%{y:.4f}<extra>Over-fitted</extra>",
    ))
    fig_res.add_hline(y=0, line_width=1, line_color="black")
    ymax_res = _ymax_without(np.abs(residual_vals), result.grid)
    fig_res.update_layout(
        **_LAYOUT_BASE,
        barmode="overlay",
        xaxis_title="m/z",
        yaxis_title="Observed − Fitted",
        yaxis=dict(range=[-ymax_res, ymax_res]) if ymax_res else {},
        margin=dict(l=60, r=40, t=30, b=50),
        height=_RESIDUAL_PLOT_HEIGHT,
    )
    sections.append((
        "Residual Spectrum (Observed − Fitted)",
        _embed_plotly_div("rgakit-residual", _fig_to_json(fig_res), _RESIDUAL_PLOT_HEIGHT),
    ))

    # ------------------------------------------------------------------ #
    # Section 3 — Compound contributions
    # ------------------------------------------------------------------ #
    bar_colors = [color_map[n] for n in names]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=pcts[::-1],
        y=names[::-1],
        orientation="h",
        marker_color=bar_colors[::-1],
        text=[f"{p:.1f}%" for p in pcts[::-1]],
        textposition="outside",
        hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
    ))
    fig2.update_layout(
        **_LAYOUT_BASE,
        showlegend=False,
        xaxis_title="Contribution (%)",
        yaxis=dict(tickfont_size=12, automargin=True),
        margin=dict(l=_CONTRIB_MARGIN_LEFT, r=60, t=30, b=50),
    )
    contrib_height = max(_CONTRIB_HEIGHT_MIN, _CONTRIB_HEIGHT_PER_ROW * len(contribs) + 80)
    sections.append((
        "Compound Contributions",
        _embed_plotly_div("rgakit-contribs", _fig_to_json(fig2), contrib_height),
    ))

    # ------------------------------------------------------------------ #
    # Section 4 — Stacked per-compound spectrum + scrollable legend
    # ------------------------------------------------------------------ #
    if library is not None:
        all_items  = list(contribs.items())
        if top_n is not None:
            all_items = all_items[:top_n]
        bottom     = np.zeros(len(result.grid))

        fig3          = go.Figure()
        legend_entries = []   # (name, color, pct)

        for i, (name, weight) in enumerate(all_items):
            try:
                spectrum = library[name]
            except KeyError:
                warnings.warn(
                    f"Compound '{name}' not found in library; omitting from stacked plot.",
                    stacklevel=2,
                )
                continue
            contrib = spectrum.on_grid(result.grid) * weight
            nz      = contrib > 0        # only pass non-zero positions to the trace
            color   = color_map[name]
            fig3.add_trace(go.Bar(
                x=result.grid[nz],
                y=contrib[nz],
                base=bottom[nz],
                name=name,
                marker_color=color,
                showlegend=False,
                hoverinfo="skip",
            ))
            legend_entries.append((name, color, 100 * weight / total))
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

        ymax3    = _ymax_without(result.observed, result.grid)
        ymax3_js = f"[0, {ymax3}]" if ymax3 else "null"
        fig3.update_layout(
            **_LAYOUT_BASE,
            barmode="overlay",
            showlegend=False,
            hovermode="x",
            xaxis=dict(
                title="m/z",
                showspikes=True,
                spikemode="across",
                spikecolor="#aaa",
                spikethickness=1,
                spikedash="dot",
            ),
            yaxis_title="Relative intensity",
            yaxis=dict(range=[0, ymax3]) if ymax3 else {},
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

        # CSV data for download
        csv_rows = ["Compound,Weight,Pct"] + [
            f"{name},{weight:.6f},{pct:.2f}"
            for (name, weight), pct in zip(contribs.items(), pcts)
        ]
        csv_b64_data = json.dumps("\n".join(csv_rows))  # embed as JSON string in JS

        stacked_toolbar = f"""
<div class="stacked-toolbar">
  <button id="stkLogBtn" onclick="stkToggleLog()">Log scale</button>
  <button onclick="stkIsolateReset({n_compound_traces})">Reset view</button>
  <button onclick="stkDownloadCSV()">Download CSV</button>
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
    <div class="stacked-legend-scroll" id="stkLegendScroll">{legend_items_html}</div>
  </div>
</div>
<script>
(function() {{
  var GD  = 'rgakit-stacked';
  var LOG = false;
  var ISOLATED = -1;

  var fig = {_fig_to_json(fig3)};
  fig.layout.height = {_SPECTRUM_PLOT_HEIGHT};
  Plotly.newPlot(GD, fig.data, fig.layout, {{responsive:true}});

  // Store original stacked bases and per-trace zero arrays (each trace
  // may have a different number of x points after non-zero filtering).
  var bases      = fig.data.slice(0, {n_compound_traces}).map(function(t) {{
    var b = t.base;
    if (Array.isArray(b)) return b.slice();
    return new Array(t.x.length).fill(typeof b === 'number' ? b : 0);
  }});
  var traceZeros = fig.data.slice(0, {n_compound_traces}).map(function(t) {{
    return new Array(t.x.length).fill(0);
  }});
  var traceIdx = Array.from({{length: {n_compound_traces}}}, function(_,k){{return k;}});

  function stkHL(idx, n) {{
    if (ISOLATED >= 0) return;
    var ops      = [];
    var newBases = [];
    for (var i = 0; i < n; i++) {{
      ops.push(i === idx ? 1.0 : 0.07);
      newBases.push(i === idx ? traceZeros[i] : bases[i]);
    }}
    Plotly.restyle(GD, {{'opacity': ops, 'base': newBases}}, traceIdx);
    document.querySelectorAll('.legend-item').forEach(function(el, j) {{
      el.style.opacity = (j === idx) ? '1' : '0.35';
    }});
  }}

  function stkReset(n) {{
    if (ISOLATED >= 0) return;
    var ops = Array(n).fill(1.0);
    Plotly.restyle(GD, {{'opacity': ops, 'base': bases}}, traceIdx);
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      el.style.opacity = '1';
    }});
  }}

  function stkIsolate(el, idx, n) {{
    if (ISOLATED === idx) {{
      ISOLATED = -1;
      el.classList.remove('isolated');
      stkIsolateReset(n);
    }} else {{
      ISOLATED = idx;
      document.querySelectorAll('.legend-item').forEach(function(item) {{
        item.classList.remove('isolated');
        item.style.opacity = '0.35';
      }});
      el.classList.add('isolated');
      el.style.opacity = '1';
      var ops      = [];
      var newBases = [];
      for (var i = 0; i < n; i++) {{
        ops.push(i === idx ? 1.0 : 0.04);
        newBases.push(i === idx ? traceZeros[i] : bases[i]);
      }}
      Plotly.restyle(GD, {{'opacity': ops, 'base': newBases}}, traceIdx);
    }}
  }}

  function stkIsolateReset(n) {{
    ISOLATED = -1;
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      el.classList.remove('isolated');
      el.style.opacity = '1';
    }});
    Plotly.restyle(GD, {{'opacity': Array(n).fill(1.0), 'base': bases}}, traceIdx);
  }}

  function stkToggleLog() {{
    LOG = !LOG;
    var btn = document.getElementById('stkLogBtn');
    btn.classList.toggle('active', LOG);
    if (LOG) {{
      Plotly.relayout(GD, {{'yaxis.type': 'log', 'yaxis.autorange': true}});
    }} else {{
      Plotly.relayout(GD, {{'yaxis.type': 'linear', 'yaxis.range': {ymax3_js}}});
    }}
  }}

  function stkFilterLegend(q) {{
    q = q.toLowerCase();
    document.querySelectorAll('.legend-item').forEach(function(el) {{
      var name = el.getAttribute('data-name').toLowerCase();
      el.style.display = name.includes(q) ? '' : 'none';
    }});
  }}

  function stkDownloadCSV() {{
    var csv = {csv_b64_data};
    var blob = new Blob([csv], {{type: 'text/csv'}});
    var url  = URL.createObjectURL(blob);
    var a    = document.createElement('a');
    a.href = url; a.download = 'rga_contributions.csv';
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
  }}

  window.stkHL           = stkHL;
  window.stkReset        = stkReset;
  window.stkIsolate      = stkIsolate;
  window.stkIsolateReset = stkIsolateReset;
  window.stkToggleLog    = stkToggleLog;
  window.stkFilterLegend = stkFilterLegend;
  window.stkDownloadCSV  = stkDownloadCSV;
}})();
</script>"""

        heading = "Stacked Contributions" + (f" (top {top_n})" if top_n else "")
        sections.append((heading, section4_body))

    # ------------------------------------------------------------------ #
    # Assemble HTML page
    # ------------------------------------------------------------------ #
    subtitle = (
        f"residual = {result.residual:.4f}"
        f" &nbsp;|&nbsp; {len(result.grid)} m/z points"
        f" &nbsp;|&nbsp; {len(contribs)} compounds detected"
    )

    # Optional metadata badges from the spectrum object
    meta_row_html = ""
    if spectrum is not None:
        meta = getattr(spectrum, "metadata", {}) or {}
        badges = []
        x_val = meta.get("x")
        y_val = meta.get("y")
        if x_val is not None and y_val is not None:
            badges.append(f"x = {x_val:.2f} mm, y = {y_val:.2f} mm")
        pd_val = meta.get("pd_ua")
        if pd_val is not None:
            badges.append(f"PD = {pd_val:.2f} µA")
        n_scans = meta.get("n_open_scans")
        if n_scans is not None:
            badges.append(f"{n_scans} open scans")
        bg = meta.get("background_correct")
        if bg:
            badges.append("background corrected")
        if badges:
            meta_row_html = (
                '<div class="meta-row">'
                + "".join(f'<span class="meta-badge">{b}</span>' for b in badges)
                + "</div>"
            )

    # Top-N contributor pills
    top_pills_html = ""
    top_items = list(contribs.items())[:6]
    if top_items:
        pills = "".join(
            f'<span class="contrib-pill" style="background:{color_map[n]}">'
            f'{n} &nbsp;{100*w/total:.0f}%</span>'
            for n, w in top_items
        )
        top_pills_html = (
            f'<div class="top-contribs">'
            f'<span class="top-contribs-label">Top:</span>{pills}</div>'
        )

    section_html = "\n".join(
        f'<div class="section"><h2>{heading}</h2>{div}</div>'
        for heading, div in sections
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
  <summary style="cursor:pointer;font-size:0.9em;color:#666;padding:4px 0;">
    Sample metadata
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
  <div class="page-wrap">
    <div class="report-header">
      <h1>{title}</h1>
      <p class="subtitle">{subtitle}</p>
      {meta_row_html}
      {top_pills_html}
    </div>
    {summary_card}
    {meta_html}
    {section_html}
  </div>
</body>
</html>"""

    dest = Path(output_path)
    dest.write_text(page)
    print(f"Report saved → {dest.resolve()}")
    return dest
