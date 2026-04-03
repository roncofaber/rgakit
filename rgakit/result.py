"""
result.py
---------
FitResult: output container for SpectraLibrary.fit().
"""

from __future__ import annotations

import numpy as np


class FitResult:
    """Container for the output of SpectraLibrary.fit()."""

    def __init__(
        self,
        weights:  dict[str, float],
        residual: float,
        fitted:   np.ndarray,
        observed: np.ndarray,
        grid:     np.ndarray,
    ):
        self.weights  = weights
        self.residual = residual
        self.fitted   = fitted
        self.observed = observed
        self.grid     = grid

    @property
    def contributions(self) -> dict[str, float]:
        """Non-zero weights only, sorted by descending contribution."""
        return dict(
            sorted(
                ((k, v) for k, v in self.weights.items() if v > 0),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    def plot(self, ax=None):
        """Plot observed vs fitted spectrum."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.bar(self.grid, self.observed, width=0.8, label="Observed", alpha=0.5, color="gray")
        ax.bar(self.grid, self.fitted,   width=0.8, label="Fitted",   alpha=0.6, color="steelblue", linewidth=0)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Relative intensity")
        ax.legend()
        ax.set_title(f"Fit  |  residual = {self.residual:.4f}")
        return ax

    def summary(self, threshold: float = 1e-4, bar_width: int = 28, print_output: bool = True) -> str:
        """
        Return (and optionally print) a formatted table of contributions.

        Parameters
        ----------
        threshold    : minimum weight to include a compound
        bar_width    : width of the in-table bar chart
        print_output : if True (default), also print to stdout
        """
        contribs = {k: v for k, v in self.contributions.items() if v >= threshold}
        total    = sum(contribs.values())
        max_w    = max(contribs.values()) if contribs else 1.0

        col_name = max((len(n) for n in contribs), default=8)
        col_name = max(col_name, 8)

        # Row layout: "  " + name(col_name) + "  " + weight(8) + "   " + pct(6) + "   " + bar(bar_width)
        w_total = 2 + col_name + 2 + 8 + 3 + 6 + 3 + bar_width

        div  = "─" * w_total
        hdiv = "═" * w_total

        lines = []
        lines.append(f"╒{hdiv}╕")
        info = f"  Fit result   residual = {self.residual:.4f}   grid = {len(self.grid)} m/z points"
        lines.append(f"│{info:<{w_total}}│")
        lines.append(f"╞{hdiv}╡")
        lines.append(f"│{'  ' + 'Compound':<{col_name + 2}}  {'Weight':>8}   {'%':>6}   {'':>{bar_width}}│")
        lines.append(f"├{div}┤")

        for name, w in contribs.items():
            pct    = 100 * w / total
            filled = int(round(w / max_w * bar_width))
            bar    = "█" * filled + "░" * (bar_width - filled)
            lines.append(f"│  {name:<{col_name}}  {w:>8.4f}   {pct:>5.1f}%   {bar}│")

        lines.append(f"├{div}┤")
        footer = f"  {'Total':<{col_name}}  {total:>8.4f}   {'100.0%':>6}"
        lines.append(f"│{footer:<{w_total}}│")
        lines.append(f"╘{hdiv}╛")

        output = "\n".join(lines)
        if print_output:
            print(output)
        return output

    def __repr__(self) -> str:
        top     = list(self.contributions.items())[:3]
        top_str = ", ".join(f"{n}: {w:.3f}" for n, w in top)
        return f"FitResult(residual={self.residual:.4f}, top=[{top_str}])"
