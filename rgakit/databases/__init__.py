"""
rgakit.databases
----------------
Spectral reference database interfaces.

Available databases
-------------------
InSilicoDatabase       — FastEI predicted EI-MS SQLite library (~2.25 M spectra)
MassBankDatabase       — MassBank REST API client (massbank.eu or self-hosted)
MonaDatabase           — MassBank of North America REST API client (mona.fiehnlab.ucdavis.edu)
MonaLocalDatabase      — Local MoNA SQLite library
NistDatabase           — Local NIST EI-MS SQLite library
LocalSpectralDatabase  — Base class for all local SQLite spectral libraries
"""

from .local     import LocalSpectralDatabase
from .insilico  import InSilicoDatabase
from .massbank  import MassBankDatabase
from .mona      import MonaDatabase, MonaLocalDatabase
from .nist      import NistDatabase

__all__ = [
    "LocalSpectralDatabase",
    "InSilicoDatabase",
    "MassBankDatabase",
    "MonaDatabase",
    "MonaLocalDatabase",
    "NistDatabase",
]
