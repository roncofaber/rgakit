"""
draw.py
-------
Molecular structure visualisation for Compound and its fragments.

Provides
--------
generate_fragment_wheel(compound, ...)
    Radial "wheel" figure: parent molecule at centre, stable fragments
    arranged around it.  Figure auto-sizes to fit the content.

Sizing model
------------
All sizes are in **inches** (``parent_in``, ``frag_in``, etc.).  Figure
dimensions are derived automatically from the number of fragments and the
sizing parameters — no manual ``figsize`` tuning needed.  Pixel resolution
is controlled by ``dpi``.
"""
from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PARENT_EDGE   = "#1565C0"   # dark blue  — parent box border
_MATCHED_EDGE  = "#000000"   # dark green — library-matched fragment border
_MISS_EDGE     = "#9E9E9E"   # medium grey — unmatched fragment border
_LINE_MATCHED  = "#000000"   # light green — spoke to matched fragment
_LINE_MISS     = "#BDBDBD"   # light grey  — spoke to unmatched fragment

# Unicode subscript digits for chemical formula rendering.
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# matplotlib OffsetImage convention: zoom=1 means 1 image pixel = 1 point = 1/72".
# To display an image at a physical size of s inches, use:
#   image_px = s * render_dpi
#   zoom     = _MPL_PTS_PER_INCH / render_dpi
# so that  image_px * zoom / _MPL_PTS_PER_INCH = s  ✓
_MPL_PTS_PER_INCH = 72.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fmt_formula(formula: str) -> str:
    """Return *formula* with digit runs replaced by Unicode subscripts.

    Example: ``"C4H8O"`` → ``"C₄H₈O"``
    """
    return formula.translate(_SUB) if formula else ""


def _mol_svg_str(smiles: str, px: int, bond_px: int = 0) -> "str | None":
    """Render SMILES → SVG string via RDKit MolDraw2DSVG.

    Returns the raw SVG text (with outer ``<svg>`` tag) or *None* on failure.
    """
    try:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(px, px)
        if bond_px > 0:
            drawer.drawOptions().fixedBondLength = float(bond_px)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def _mol_img(smiles: str, px: int, bond_px: int = 0) -> "np.ndarray | None":
    """Render SMILES → square RGBA numpy array (``px × px``) via RDKit.

    Render pipeline (tried in order):
      1. MolDraw2DSVG → cairosvg → numpy   best quality; vector paths → raster
      2. MolDraw2DCairo → numpy            Cairo PNG; requires RDKit Cairo build
      3. MolToImage → numpy                Agg fallback

    Note: when saving the figure as SVG, matplotlib still embeds the result
    as a base64 PNG inside an <image> tag — fully vector molecule output would
    require bypassing matplotlib's rendering pipeline.
    """
    try:
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = MolFromSmiles(smiles)
        if mol is None:
            return None

        def _configure(drawer):
            if bond_px > 0:
                drawer.drawOptions().fixedBondLength = float(bond_px)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()

        # ── 1. SVG → cairosvg → numpy (best) ────────────────────────────
        try:
            import cairosvg
            from PIL import Image
            from io import BytesIO
            drawer = rdMolDraw2D.MolDraw2DSVG(px, px)
            _configure(drawer)
            png = cairosvg.svg2png(
                bytestring=drawer.GetDrawingText().encode(),
                output_width=px, output_height=px,
            )
            return np.array(Image.open(BytesIO(png)))
        except ImportError:
            pass

        # ── 2. Cairo PNG ─────────────────────────────────────────────────
        try:
            from PIL import Image
            from io import BytesIO
            drawer = rdMolDraw2D.MolDraw2DCairo(px, px)
            _configure(drawer)
            return np.array(Image.open(BytesIO(drawer.GetDrawingText())))
        except Exception:
            pass

        # ── 3. Agg fallback ──────────────────────────────────────────────
        from rdkit.Chem.Draw import MolToImage
        return np.array(MolToImage(mol, size=(px, px)))

    except Exception:
        return None


def _place_image(ax, img_arr, xy_frac, zoom, edge_color="black", lw=1.2,
                 zorder=3, shadow=False):
    """Place *img_arr* centred at *xy_frac* (axes-fraction coordinates).

    *zoom* must be ``_MPL_PTS_PER_INCH / render_dpi`` so that the image
    displays at the correct physical size regardless of figure DPI.
    If *shadow* is True, a soft drop shadow is rendered behind the box.
    """
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    ab = AnnotationBbox(
        OffsetImage(img_arr, zoom=zoom),
        xy_frac,
        xycoords="axes fraction",
        frameon=True,
        zorder=zorder,
        bboxprops=dict(
            boxstyle="round,pad=0.15",
            edgecolor=edge_color,
            linewidth=lw,
            facecolor="white",
        ),
    )
    ax.add_artist(ab)
    if shadow:
        from matplotlib import patheffects
        ab.patch.set_path_effects([
            patheffects.withSimplePatchShadow(
                offset=(2, -2),
                shadow_rgbFace=(0, 0, 0),
                alpha=0.18,
            ),
        ])


def _match_fragments(frags, library):
    """Return ``(matched [(frag, spec)…], unmatched [frag…])`` by InChIKey."""
    from rgakit.molecule.utils import smiles_to_inchikey

    ik_to_spec   = {}
    name_to_spec = {}
    for spec in library:
        ik = spec.metadata.get("inchikey")
        if ik:
            ik_to_spec[ik] = spec
        name_to_spec[spec.name] = spec

    matched, unmatched = [], []
    for frag in frags:
        ik   = smiles_to_inchikey(frag.smiles) if frag.smiles else None
        spec = ik_to_spec.get(ik) or name_to_spec.get(frag.formula)
        (matched if spec is not None else unmatched).append(
            (frag, spec) if spec is not None else frag
        )
    return matched, unmatched


def _build_display(all_frags, library, result, show_unmatched,
                   min_contribution, max_frags):
    """Ordered ``(frag, spec|None)`` list + fit-contribution dict."""
    if library is not None:
        matched, unmatched = _match_fragments(all_frags, library)
    else:
        matched, unmatched = [(f, None) for f in all_frags], []

    display: list[tuple] = list(matched)
    if show_unmatched:
        display += [(f, None) for f in unmatched]
    display.sort(key=lambda x: (-x[0].n_heavy, x[1] is None))

    fit_contribs: dict = {}
    if result is not None:
        fit_contribs = result.contributions
        display = [
            (f, spec) for f, spec in display
            if spec is not None
            and fit_contribs.get(spec.name, 0) >= min_contribution
        ]
        display.sort(key=lambda x: -fit_contribs.get(x[1].name, 0))

    return display[:max_frags], matched, unmatched, fit_contribs


def _frag_label(frag, spec, fit_contribs) -> str:
    """Two-line label: name (+ %) on line 1, formula + MW on line 2."""
    name = spec.name if spec else _fmt_formula(frag.formula)
    if fit_contribs and spec is not None:
        pct   = fit_contribs.get(spec.name, 0) * 100
        line1 = f"{name} ({pct:.0f}%)"
    else:
        line1 = name
    line2 = f"{_fmt_formula(frag.formula)}  {frag.monoisotopic_mass:.1f} Da"
    return f"{line1}\n{line2}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_fragment_wheel(
    compound,
    library          = None,
    result           = None,
    output_path      = None,
    title            = None,
    # --- molecule sizes (inches) ---
    parent_in        = 0.70,
    frag_in          = 0.50,
    # --- orbit / spoke geometry (inches) ---
    gap_in           = 0.10,
    orbit_in         = None,
    spoke_frac       = 1.0,
    arrow_pad_in     = 0.04,
    # --- canvas geometry (inches) ---
    label_in         = 0.38,
    margin_in        = 0.12,
    # --- content filters ---
    max_frags        = 20,
    show_unmatched   = True,
    show_labels      = True,
    min_contribution = 1e-4,
    # --- rendering ---
    bond_length      = 0.09,
    dpi              = 150,
    supersample      = 2,
    shadow           = True,
):
    """
    Generate a radial wheel image of a compound and its fragments.

    The parent molecule sits at the centre; stable fragments are arranged
    in a circle around it connected by arrow spokes.  The figure size is
    computed automatically — no manual ``figsize`` tuning needed.

    If *library* is provided:
      - matched fragments → green border and spoke
      - unmatched fragments → grey border and spoke

    If *result* is also provided, only fit-identified fragments are shown
    (sorted by contribution), each labelled with its percentage.

    Parameters
    ----------
    compound         : Compound  (must have called ``do_fragmentation()``)
    library          : optional SpectraLibrary; enables match colour-coding
    result           : optional FitResult; filters to fit-identified fragments
    output_path      : save path (PNG / SVG / PDF); None returns the figure
    title            : figure title string; None (default) omits the title

    parent_in        : parent molecule box size in inches (default 0.70)
    frag_in          : fragment box size in inches (default 0.50)

    gap_in           : minimum edge-to-edge gap between parent and fragment
                       boxes when computing the orbit automatically (default 0.10).
                       Ignored when *orbit_in* is given explicitly.
    orbit_in         : centre-to-centre orbit radius in inches.
                       None (default) → auto-computed to prevent box overlap.
    spoke_frac       : fraction of the available gap drawn as an arrow (default 1.0).
                       1.0 = full span parent→fragment; 0.3 = short tick near fragment.
    arrow_pad_in     : clearance between the arrowhead / tail and each box
                       edge in inches (default 0.04)

    label_in         : radial space reserved for text labels in inches (default 0.38);
                       ignored (set to 0) when *show_labels* is False
    margin_in        : whitespace between the outermost element and the figure
                       edge, in inches (default 0.12)

    max_frags        : maximum number of fragments to display (default 20)
    show_unmatched   : show fragments with no library match (default True);
                       ignored when *result* is given
    show_labels      : show name / formula / MW labels beside each fragment
    min_contribution : minimum fit contribution to include a fragment (default 1e-4);
                       only relevant when *result* is given

    bond_length      : C–C bond display length in inches (default 0.09).
                       Shared across all molecules so their relative sizes are
                       preserved.  Set 0 to let each molecule auto-fill its box.
    dpi              : output resolution in dots per inch (default 150)
    supersample      : internal render multiplier for crisper molecule images
                       (default 2); molecule pixels = size_in × dpi × supersample
    shadow           : draw a drop shadow behind each molecule box (default True)

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # ------------------------------------------------------------------
    # Fragment selection
    # ------------------------------------------------------------------
    all_frags = [f for f in compound.fragments if f.smiles]
    display, matched, unmatched, fit_contribs = _build_display(
        all_frags, library, result, show_unmatched, min_contribution, max_frags
    )
    n = len(display)

    # ------------------------------------------------------------------
    # Rendering parameters
    # Molecules are rendered at (dpi × supersample) for crisp lines, then
    # displayed via OffsetImage with zoom = 72 / mol_dpi so that
    #   image_px × zoom / 72 = frag_in  (correct physical size).
    # ------------------------------------------------------------------
    mol_dpi   = dpi * supersample
    zoom      = _MPL_PTS_PER_INCH / mol_dpi
    bond_px   = max(0, round(bond_length * mol_dpi))
    parent_px = max(1, round(parent_in * mol_dpi))
    frag_px   = max(1, round(frag_in   * mol_dpi))

    # Font sizes derived from physical box dimensions (in points = 1/72 inch)
    # so they scale correctly when frag_in / parent_in are changed.
    # Proportionality constant 0.22 reproduces ~8pt at the default frag_in=0.50".
    _fs_label  = max(6, round(frag_in   * _MPL_PTS_PER_INCH * 0.22))  # fragment labels
    _fs_parent = max(6, round(parent_in * _MPL_PTS_PER_INCH * 0.18))  # parent name
    _fs_title  = max(8, round(parent_in * _MPL_PTS_PER_INCH * 0.22))  # figure title

    # ------------------------------------------------------------------
    # Layout geometry — everything derived from inch measurements
    # ------------------------------------------------------------------
    # Orbit radius (wheel centre → fragment centre).
    # Auto-mode: enforce a no-overlap chord condition.
    #   chord = 2·r·sin(π/n) ≥ √2·frag_in  (worst-case diagonal × 1.2 safety)
    if orbit_in is None:
        if n > 1:
            orbit_no_overlap = 1.2 * math.sqrt(2) * frag_in / (2 * math.sin(math.pi / n))
        else:
            orbit_no_overlap = 0.0
        orbit_in = max(
            parent_in / 2 + gap_in + frag_in / 2,
            orbit_no_overlap,
        )

    # Canvas radius = orbit + fragment half-size + label space (0 if hidden) + margin.
    _label_in = label_in if show_labels else 0.0
    canvas_r  = orbit_in + frag_in / 2 + _label_in + margin_in
    canvas_in = 2 * canvas_r

    # Unit conversion: inches → axes fraction (axes is square [0,1]×[0,1]).
    def _f(x): return x / canvas_in

    r_parent_f = _f(parent_in / 2)
    r_frag_f   = _f(frag_in   / 2)
    r_orbit_f  = _f(orbit_in)

    # _ap is constant; tip_r / r_spoke_f / tail_r are computed per-fragment
    # inside the loop because they depend on the spoke angle.
    _ap = _f(arrow_pad_in)

    # ------------------------------------------------------------------
    # Figure
    # Title (if any) occupies a fixed strip above the square wheel canvas.
    # The axes fills exactly the canvas square; the strip sits above it.
    # This keeps the wheel square and the title inside the figure boundary.
    # ------------------------------------------------------------------
    _title_h_in = 0.28 if title is not None else 0.0
    figw = canvas_in
    figh = canvas_in + _title_h_in

    fig, ax = plt.subplots(figsize=(figw, figh))
    fig.subplots_adjust(left=0, right=1, top=canvas_in / figh, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Parent molecule
    # ------------------------------------------------------------------
    parent_arr = _mol_img(compound.smiles, parent_px, bond_px)
    if parent_arr is not None:
        _place_image(ax, parent_arr, (0.5, 0.5), zoom=zoom,
                     edge_color=_PARENT_EDGE, lw=1.5, zorder=5, shadow=shadow)
    ax.text(0.5, 0.5 - r_parent_f - _f(arrow_pad_in),
            compound.name or _fmt_formula(compound.formula),
            ha="center", va="top", fontsize=_fs_parent-12, fontweight="bold",
            color=_PARENT_EDGE, transform=ax.transAxes)

    # ------------------------------------------------------------------
    # Spokes + fragment boxes
    # ------------------------------------------------------------------
    for i, (frag, spec) in enumerate(display):
        angle        = 2 * math.pi * i / n - math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        fx = 0.5 + r_orbit_f * cos_a
        fy = 0.5 + r_orbit_f * sin_a

        is_matched   = spec is not None
        spoke_color  = _LINE_MATCHED if is_matched else _LINE_MISS
        border_color = _MATCHED_EDGE if is_matched else _MISS_EDGE

        # For an axis-aligned square of half-side s, the spoke-direction distance
        # from box centre to box edge is  s / max(|cos θ|, |sin θ|)  — up to √2·s
        # at 45°.  Using the constant r_frag_f = s would place the arrow tip inside
        # the box at diagonal angles, so we compute this per fragment.
        _mcs          = max(abs(cos_a), abs(sin_a))
        frag_edge_f   = _f((frag_in   / 2) / _mcs)
        parent_edge_f = _f((parent_in / 2) / _mcs)
        tip_r         = r_orbit_f - frag_edge_f - _ap
        r_spoke_f     = max(0.0, r_orbit_f - parent_edge_f - frag_edge_f - 2 * _ap)
        tail_r        = tip_r - spoke_frac * r_spoke_f

        # Arrow (tip just outside fragment box edge, tail pulled in by spoke_frac)
        ax.annotate(
            "",
            xy     = (fx        - (frag_edge_f + _ap) * cos_a,
                      fy        - (frag_edge_f + _ap) * sin_a),
            xytext = (0.5 + tail_r * cos_a,
                      0.5 + tail_r * sin_a),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=spoke_color,
                            lw=1.2, mutation_scale=10),
            zorder=1,
        )

        frag_arr = _mol_img(frag.smiles, frag_px, bond_px)
        if frag_arr is not None:
            _place_image(ax, frag_arr, (fx, fy), zoom=zoom,
                         edge_color=border_color, lw=1.2, zorder=4, shadow=shadow)

        if show_labels:
            lx = 0.5 + (r_orbit_f + r_frag_f + _f(2 * arrow_pad_in)) * cos_a
            ly = 0.5 + (r_orbit_f + r_frag_f + _f(2 * arrow_pad_in)) * sin_a
            ha = "left" if cos_a > 0.1 else ("right" if cos_a < -0.1 else "center")
            va = "bottom" if sin_a > 0.1 else ("top" if sin_a < -0.1 else "center")
            ax.text(lx, ly, _frag_label(frag, spec, fit_contribs),
                    ha=ha, va=va, fontsize=_fs_label, color=border_color,
                    transform=ax.transAxes, zorder=6)

    # ------------------------------------------------------------------
    # Title + legend
    # ------------------------------------------------------------------
    if title is not None:
        ax.set_title(title, fontsize=_fs_title, fontweight="bold", pad=4)

    if library is not None and result is None:
        ax.legend(handles=[
            mpatches.Patch(color=_MATCHED_EDGE, label=f"In library ({len(matched)})"),
            mpatches.Patch(color=_MISS_EDGE,    label=f"No match ({len(unmatched)})"),
        ], loc="lower right", fontsize=_fs_label, framealpha=0.9)

    fig.patch.set_facecolor("white")
    if output_path is not None:
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight",
                    pad_inches=0, facecolor="white")
        logger.info("Fragment wheel saved → %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# SVG backend — generate_fragment_wheel_svg
# Produces truly vector output: molecule SVGs are embedded as data-URI <image>
# elements, so they remain scalable in the saved SVG file.
# ---------------------------------------------------------------------------

def _svg_img_tag(data_uri: str, x: float, y: float, size: float) -> str:
    """Return a raw SVG <image> tag centred at (x, y) with the given *size*."""
    return (
        f'<image x="{x - size/2:.3f}" y="{y - size/2:.3f}" '
        f'width="{size:.3f}" height="{size:.3f}" '
        f'href="{data_uri}" preserveAspectRatio="xMidYMid meet"/>'
    )


def _svg_mol_uri(smiles: str, px: int, bond_px: int = 0) -> "str | None":
    """Return a ``data:image/svg+xml;base64,...`` URI for *smiles*, or *None*."""
    import base64
    svg = _mol_svg_str(smiles, px, bond_px)
    if svg is None:
        return None
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _svg_multiline_text(lines: list, x: float, y: float, font_size: float,
                        fill: str, text_anchor: str, line_height: float,
                        font_weight: str = "normal",
                        font_family: str = "sans-serif") -> str:
    """Return a raw <text> block with one <tspan> per line."""
    dy_first = -(len(lines) - 1) * line_height / 2
    tspans = []
    for k, line in enumerate(lines):
        dy = dy_first if k == 0 else line_height
        tspans.append(
            f'<tspan x="{x:.3f}" dy="{dy:.3f}">{line}</tspan>'
        )
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" '
        f'font-size="{font_size:.2f}" fill="{fill}" '
        f'text-anchor="{text_anchor}" dominant-baseline="central" '
        f'font-family="{font_family}" font-weight="{font_weight}" '
        f'xml:space="preserve">'
        + "".join(tspans)
        + "</text>"
    )


def generate_fragment_wheel_svg(
    compound,
    library          = None,
    result           = None,
    output_path      = None,
    title            = None,
    # --- molecule sizes (inches) ---
    parent_in        = 0.70,
    frag_in          = 0.50,
    # --- orbit / spoke geometry (inches) ---
    gap_in           = 0.10,
    orbit_in         = None,
    spoke_frac       = 1.0,
    arrow_pad_in     = 0.04,
    # --- canvas geometry (inches) ---
    label_in         = 0.38,
    margin_in        = 0.12,
    # --- content filters ---
    max_frags        = 20,
    show_unmatched   = True,
    show_labels      = True,
    min_contribution = 1e-4,
    # --- rendering ---
    bond_length      = 0.09,
    dpi              = 150,
    supersample      = 2,
    shadow           = True,
):
    """
    Generate a radial wheel SVG of a compound and its fragments using *drawsvg*.

    Produces truly vector output: molecules are embedded as base64-encoded SVG
    ``<image>`` elements, so they remain scalable.  Shadows use a real SVG
    Gaussian-blur ``<filter>``; arrowheads are proper SVG ``<marker>`` elements.

    Parameters are identical to :func:`generate_fragment_wheel` (matplotlib).

    Returns
    -------
    drawsvg.Drawing
    """
    import drawsvg as dsvg

    # ------------------------------------------------------------------
    # Fragment selection — identical logic to matplotlib version
    # ------------------------------------------------------------------
    all_frags = [f for f in compound.fragments if f.smiles]
    display, matched, unmatched, fit_contribs = _build_display(
        all_frags, library, result, show_unmatched, min_contribution, max_frags
    )
    n = len(display)

    # ------------------------------------------------------------------
    # Pixel conversion.  All layout math stays in inches; multiply by
    # dpi to get SVG user units (1 uu = 1 px at the given dpi).
    # ------------------------------------------------------------------
    p = dpi  # alias: inches → SVG user units

    mol_dpi  = dpi * supersample
    bond_px  = max(0, round(bond_length * mol_dpi))
    parent_px = max(1, round(parent_in * mol_dpi))
    frag_px   = max(1, round(frag_in   * mol_dpi))

    # Font sizes in SVG user units (1pt = dpi/72 uu)
    _pt = dpi / 72.0
    fs_label  = max(6, round(frag_in   * 72 * 0.22)) * _pt
    fs_parent = max(6, round(parent_in * 72 * 0.18)) * _pt
    fs_title  = max(8, round(parent_in * 72 * 0.22)) * _pt
    lh_label  = fs_label * 1.3   # line height for multiline labels

    # ------------------------------------------------------------------
    # Layout geometry — same formulae as matplotlib version
    # ------------------------------------------------------------------
    if orbit_in is None:
        if n > 1:
            orbit_no_overlap = 1.2 * math.sqrt(2) * frag_in / (2 * math.sin(math.pi / n))
        else:
            orbit_no_overlap = 0.0
        orbit_in = max(parent_in / 2 + gap_in + frag_in / 2, orbit_no_overlap)

    _label_in = label_in if show_labels else 0.0
    canvas_r  = orbit_in + frag_in / 2 + _label_in + margin_in
    canvas_in = 2 * canvas_r

    canvas_px = canvas_in * p
    _title_h  = 0.32 * p if title is not None else 0.0
    fig_w     = canvas_px
    fig_h     = canvas_px + _title_h

    # Wheel centre in SVG coordinates
    cx = canvas_px / 2
    cy = _title_h + canvas_px / 2

    # Inch → pixel helpers
    def _px(x): return x * p

    r_orbit_px  = _px(orbit_in)
    r_parent_px = _px(parent_in / 2)
    r_frag_px   = _px(frag_in   / 2)
    ap          = _px(arrow_pad_in)

    parent_box  = _px(parent_in)
    frag_box    = _px(frag_in)

    # ------------------------------------------------------------------
    # Drawing + defs (shadow filter + arrow markers)
    # ------------------------------------------------------------------
    d = dsvg.Drawing(fig_w, fig_h)
    d.append(dsvg.Raw('<rect width="100%" height="100%" fill="white"/>'))

    _shadow_id   = "mol-shadow"
    _arrow_match = "arr-match"
    _arrow_miss  = "arr-miss"

    shadow_filter = (
        f'<filter id="{_shadow_id}" x="-25%" y="-25%" width="150%" height="150%">'
        f'<feGaussianBlur in="SourceAlpha" stdDeviation="2.2" result="blur"/>'
        f'<feOffset in="blur" dx="2" dy="2" result="offsetBlur"/>'
        f'<feComponentTransfer in="offsetBlur" result="shadow">'
        f'<feFuncA type="linear" slope="0.20"/>'
        f'</feComponentTransfer>'
        f'<feMerge><feMergeNode in="shadow"/><feMergeNode in="SourceGraphic"/></feMerge>'
        f'</filter>'
    )

    def _arrow_marker(mid: str, fill: str) -> str:
        return (
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{fill}"/>'
            f'</marker>'
        )

    d.append(dsvg.Raw(
        "<defs>"
        + shadow_filter
        + _arrow_marker(_arrow_match, _LINE_MATCHED)
        + _arrow_marker(_arrow_miss,  _LINE_MISS)
        + "</defs>"
    ))

    # ------------------------------------------------------------------
    # Spokes + fragment boxes
    # ------------------------------------------------------------------
    def _mol_box(cx_box, cy_box, size, border_color, mol_uri, lw=1.2):
        """Draw a white rounded rect + optional shadow + molecule image."""
        x0, y0 = cx_box - size / 2, cy_box - size / 2
        rx = max(3, size * 0.06)
        flt = f'filter="url(#{_shadow_id})"' if shadow else ""
        d.append(dsvg.Raw(
            f'<rect x="{x0:.3f}" y="{y0:.3f}" width="{size:.3f}" height="{size:.3f}" '
            f'rx="{rx:.2f}" ry="{rx:.2f}" fill="white" '
            f'stroke="{border_color}" stroke-width="{lw}" {flt}/>'
        ))
        if mol_uri is not None:
            d.append(dsvg.Raw(_svg_img_tag(mol_uri, cx_box, cy_box, size)))

    for i, (frag, spec) in enumerate(display):
        angle        = 2 * math.pi * i / n - math.pi / 2
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        fx = cx + r_orbit_px * cos_a
        fy = cy - r_orbit_px * sin_a          # SVG y is downward → negate sin

        is_matched   = spec is not None
        spoke_color  = _LINE_MATCHED if is_matched else _LINE_MISS
        border_color = _MATCHED_EDGE if is_matched else _MISS_EDGE
        marker_id    = _arrow_match  if is_matched else _arrow_miss

        # Angle-corrected box-edge distances (same formula as matplotlib version)
        _mcs          = max(abs(cos_a), abs(sin_a))
        frag_edge_px  = (frag_in   / 2 / _mcs) * p
        par_edge_px   = (parent_in / 2 / _mcs) * p
        tip_r         = r_orbit_px - frag_edge_px - ap
        r_spoke       = max(0.0, r_orbit_px - par_edge_px - frag_edge_px - 2 * ap)
        tail_r        = tip_r - spoke_frac * r_spoke

        # Arrow line: tail → tip (just outside fragment box)
        x1 = cx + tail_r * cos_a
        y1 = cy - tail_r * sin_a
        x2 = cx + tip_r  * cos_a
        y2 = cy - tip_r  * sin_a
        d.append(dsvg.Raw(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{spoke_color}" stroke-width="1.2" '
            f'marker-end="url(#{marker_id})"/>'
        ))

        frag_uri = _svg_mol_uri(frag.smiles, frag_px, bond_px)
        _mol_box(fx, fy, frag_box, border_color, frag_uri, lw=1.2)

        if show_labels:
            lx = cx + (r_orbit_px + r_frag_px + ap * 2) * cos_a
            ly = cy - (r_orbit_px + r_frag_px + ap * 2) * sin_a
            # text-anchor: left / right / middle depending on angle
            ta = "start" if cos_a > 0.1 else ("end" if cos_a < -0.1 else "middle")
            label = _frag_label(frag, spec, fit_contribs)
            d.append(dsvg.Raw(
                _svg_multiline_text(
                    label.split("\n"), lx, ly,
                    font_size=fs_label, fill=border_color,
                    text_anchor=ta, line_height=lh_label,
                )
            ))

    # ------------------------------------------------------------------
    # Parent molecule
    # ------------------------------------------------------------------
    parent_uri = _svg_mol_uri(compound.smiles, parent_px, bond_px)
    _mol_box(cx, cy, parent_box, _PARENT_EDGE, parent_uri, lw=1.8)

    # Parent name just below the box
    name_label = compound.name or _fmt_formula(compound.formula)
    d.append(dsvg.Raw(
        f'<text x="{cx:.3f}" y="{cy + parent_box/2 + ap + fs_parent:.3f}" '
        f'font-size="{fs_parent:.2f}" fill="{_PARENT_EDGE}" '
        f'text-anchor="middle" dominant-baseline="central" '
        f'font-family="sans-serif" font-weight="bold">{name_label}</text>'
    ))

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------
    if library is not None and result is None:
        leg_x  = fig_w - _px(margin_in)
        leg_y  = fig_h - _px(margin_in)
        swatch = fs_label * 0.9
        for k, (color, label) in enumerate([
            (_MATCHED_EDGE, f"In library ({len(matched)})"),
            (_MISS_EDGE,    f"No match ({len(unmatched)})"),
        ]):
            ey = leg_y - k * (swatch + swatch * 0.4)
            d.append(dsvg.Raw(
                f'<rect x="{leg_x - swatch - fs_label*4:.3f}" y="{ey - swatch*0.5:.3f}" '
                f'width="{swatch:.3f}" height="{swatch:.3f}" fill="{color}" rx="2"/>'
                f'<text x="{leg_x - swatch*0.2:.3f}" y="{ey:.3f}" '
                f'font-size="{fs_label:.2f}" fill="{color}" text-anchor="end" '
                f'dominant-baseline="central" font-family="sans-serif">{label}</text>'
            ))

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    if title is not None:
        d.append(dsvg.Raw(
            f'<text x="{fig_w/2:.3f}" y="{_title_h/2:.3f}" '
            f'font-size="{fs_title:.2f}" fill="#222222" '
            f'text-anchor="middle" dominant-baseline="central" '
            f'font-family="sans-serif" font-weight="bold">{title}</text>'
        ))

    # ------------------------------------------------------------------
    # Save / return
    # ------------------------------------------------------------------
    if output_path is not None:
        path_str = str(output_path)
        if path_str.lower().endswith(".png"):
            d.save_png(path_str)
        else:
            d.save_svg(path_str)
        logger.info("Fragment wheel (SVG) saved → %s", output_path)
    return d
