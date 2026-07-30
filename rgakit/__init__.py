"""
rgakit — RGA mass spectrum decomposition and library fitting.
"""

import logging

from .spectrum     import MassSpectrum
from .library      import SpectraLibrary
from .result       import FitResult, TimeFitResult
from .stack        import SpectrumStack
from .report       import generate_report, plot_contributions, plot_stacked_spectrum
from .nomenclature import normalize_nist_name, get_compound_info, resolve_name
from .similarity   import score, pairwise
from .background   import background_correct
from .solvers         import make_solver, SUPPORTED_METHODS
from .decomposition   import decompose, DecompositionResult
from .molecule     import Compound, generate_fragment_wheel, generate_fragment_wheel_svg
from .databases    import (InSilicoDatabase, MassBankDatabase,
                           MonaDatabase, MonaLocalDatabase, NistDatabase)

# Standard library practice: add NullHandler so logs are silently discarded
# unless the calling application configures logging.
logging.getLogger("rgakit").addHandler(logging.NullHandler())

__version__ = "0.1.1"


def setup_logging(level: str = "INFO") -> None:
    """
    Configure a simple console handler for all rgakit loggers.

    Call this once at the top of a script or notebook to see progress
    messages, warnings, and errors from rgakit.

    Parameters
    ----------
    level : ``"DEBUG"`` | ``"INFO"`` | ``"WARNING"`` | ``"ERROR"``
    """
    logger = logging.getLogger("rgakit")
    # Remove any existing handlers to avoid duplicate output
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False


__all__ = [
    "MassSpectrum",
    "SpectraLibrary",
    "FitResult",
    "TimeFitResult",
    "SpectrumStack",
    "generate_report",
    "plot_contributions",
    "plot_stacked_spectrum",
    "normalize_nist_name",
    "get_compound_info",
    "resolve_name",
    "score",
    "pairwise",
    "background_correct",
    "make_solver",
    "SUPPORTED_METHODS",
    "decompose",
    "DecompositionResult",
    "setup_logging",
    "Compound",
    "generate_fragment_wheel",
    "generate_fragment_wheel_svg",
    "InSilicoDatabase",
    "MassBankDatabase",
    "MonaDatabase",
    "MonaLocalDatabase",
    "NistDatabase",
]
