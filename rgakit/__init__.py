"""
rgakit — RGA mass spectrum decomposition and library fitting.
"""

from .spectrum     import MassSpectrum
from .library      import SpectraLibrary
from .result       import FitResult
from .report       import generate_report, plot_contributions, plot_stacked_spectrum
from .nomenclature import normalize_nist_name, cactus_preferred_name, resolve_name
from .similarity   import score, pairwise

__version__ = "0.1.0"

__all__ = [
    "MassSpectrum",
    "SpectraLibrary",
    "FitResult",
    "generate_report",
    "plot_contributions",
    "plot_stacked_spectrum",
    "normalize_nist_name",
    "cactus_preferred_name",
    "resolve_name",
    "score",
    "pairwise",
]
