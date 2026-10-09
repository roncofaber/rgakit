# rgakit

Python toolkit for decomposing RGA (Residual Gas Analyzer) electron-ionisation
mass spectra into known compound contributions.

Developed for synchrotron beamline ALS BL12.0.1.2 to analyse gas desorption
from X-ray irradiated perovskite thin films.

## Features

- **NNLS and sparse fitting** - non-negative least-squares, LASSO, Elastic Net,
  OMP and ROMP decomposition against a reference library; stochastic OMP/ROMP
  trials are seedable for reproducible fits
- **Similarity metrics** - cosine, Jaccard, Pearson, spectral entropy; pairwise
  matrix for clustering
- **Library search** - rank candidates by any similarity metric before fitting
- **Iterative library building** - grow a library compound by compound from a
  spectral database until the residual is explained
- **Reference databases** - local SQLite libraries (NIST, MoNA, FastEI
  in-silico, ~2.25M spectra) plus REST clients for MassBank and MoNA
- **Blind source separation** - NMF decomposition of time-resolved RGA data
  into pure component spectra, matched against a database afterwards
- **JDX & MSP I/O** - JCAMP-DX (round-trip with metadata) and NIST MSP
  (multi-entry files)
- **Background correction** - linear subtraction using beam-off shutter windows
- **In-silico library generation** - SMILES-based fragment enumeration,
  optional MLIP geometry relaxation, export to a fitting-ready library
- **Interactive HTML report** - observed vs fitted, residual, contributions,
  stacked breakdown
- **Interactive curation review** - local web app to review candidate compounds
  one by one (structure, spectrum, stats) and record keep/remove verdicts
- **Smart naming** - common trivial names (Water, Methane, ...) with IUPAC
  fallback via NCI Cactus

## Install

```bash
pip install rgakit
# extra solver methods (LASSO, Elastic Net):
pip install "rgakit[solve]"
```

## Quick start

```python
from rgakit import MassSpectrum, SpectraLibrary, pairwise

# Build library from a folder of .jdx or .msp files
lib = SpectraLibrary.from_dir("data/jdx/")

# Load an experimental spectrum
ms = MassSpectrum.from_jdx_file("sample.jdx")

# Candidate search (cosine, jaccard, pearson, or entropy)
print(lib.search(ms, top_n=5, method="entropy"))

# NNLS fit (method="lasso" | "elastic_net" | "omp" | "romp" also supported)
result = lib.fit(ms)
result.summary()

# HTML report
from rgakit import generate_report
generate_report(result, library=lib, spectrum=ms, output_path="report.html")

# Pairwise similarity matrix (for clustering / classification)
mat, names = pairwise(lib, method="cosine")
# mat is a symmetric (N, N) ndarray; use with seaborn.heatmap or UMAP
```

### Building libraries from databases

```python
from rgakit import SpectraLibrary, NistDatabase, InSilicoDatabase, MassBankDatabase

# Fetch spectra from NIST WebBook (name, CAS, or SMILES, cached locally)
lib = SpectraLibrary.from_nist(names=["water", "methane", "methylammonium"])

# Query a local SQLite spectral database
nist = NistDatabase()                # or InSilicoDatabase() for FastEI (~2.25M spectra)
spectra = nist.get("methylamine")
matches = nist.search_by_spectrum(ms, k=10)

# MassBank / MoNA via REST
mb = MassBankDatabase()
spectra = mb.get("propane")

# Grow a library iteratively until the residual is explained
lib, result = SpectraLibrary.from_fit(ms, nist, max_compounds=30)
```

### MSP format

```python
# Load all spectra from a single NIST MSP file (multi-entry)
lib = SpectraLibrary.from_msp("nist_library.msp")

# Load one entry by index
ms = MassSpectrum.from_msp_file("nist_library.msp", index=3)

# Save a spectrum as MSP
ms.save_msp("output/water.msp")
```

## Time-resolved data and NMF

```python
from rgakit import SpectrumStack, decompose, NistDatabase

# Direct construction or clabs adapter
stack = SpectrumStack(time, pressure, shutter=(open_time, close_time))
# stack = SpectrumStack.from_rga(rga)

stack = stack.background_correct(window=5, gap_before=2, gap_after=2)
ms = stack.averaged()          # average over the shutter-open window

# Blind source separation: P ~ W x H, no reference library needed
decomp = decompose(stack, n_components=8, exclude_mz=[2, 18, 28], random_state=0)
decomp.summary()
decomp.plot_spectra()

# Match the resolved pure spectra against a database
nist = NistDatabase()
for name, spectrum in decomp.to_library():
    print(nist.search_by_spectrum(spectrum, k=5))
```

## Interactive compound curation

Review the union of your libraries one compound at a time in the browser,
recording keep/remove verdicts:

```python
from rgakit import launch_review, apply_decisions

# Review all .pkl libraries in a folder; weights add occurrence statistics
launch_review("data/libraries/",
              weights={"Ethane": [0.0, 0.1, 0.3]},
              decisions_path="decisions.json")

# Then build the curated library from the recorded verdicts
lib = apply_decisions("data/libraries/", "decisions.json")
```

Or from the command line:

```bash
rgakit-review data/libraries/ -d decisions.json --stats stats.csv
```

Keys in the browser: `y` keep, `n` remove, `s`/space skip, arrows navigate,
`u` undo. Each decision is saved immediately and the review resumes where you
left off.

## Workflow with clabs

```python
# Background-correct the raw RGA measurement in clabs, then extract spectrum
rga.background_correct()
ms = MassSpectrum.from_rga(rga)
ms.save_jdx("sample.jdx")
```

## In-silico libraries from SMILES

```python
from rgakit.molecule import Compound

# Fast workflow: SMILES-only fragmentation and recombination
mol = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
mol.do_fragmentation(max_heavy=6)   # enumerate + H-cap + radical bonding
mol.do_recombination()              # pairwise SMILES-level recombination
lib = mol.to_library()              # NIST lookup for each fragment

# Optional physics-based refinement with an ASE-compatible MLIP calculator
mol.relax(calc)
mol.relax_fragments(calc, log_dir="logs/frags")
mol.relax_recombination(calc, log_dir="logs/rec")

# Visualise the fragmentation tree
from rgakit.molecule import generate_fragment_wheel_svg
generate_fragment_wheel_svg(mol, "eai_wheel.svg")
```

## File structure

```
rgakit/
  core modules
    spectrum.py       MassSpectrum (JDX, MSP, txt I/O; similarity; from_rga)
    library.py        SpectraLibrary (fit, search, from_fit, fit_time_series, fit_bootstrap)
    result.py         FitResult / TimeFitResult (summary tables, plots)
    stack.py          SpectrumStack (time-resolved data, shutter windows, background correction)
    decomposition.py  NMF blind source separation (decompose, DecompositionResult)
    similarity.py     cosine / jaccard / pearson / entropy; pairwise matrix
    background.py     standalone linear background correction
    nomenclature.py   name resolution (trivial names, Cactus IUPAC, NIST)
    report.py         interactive HTML report generation
    review.py         interactive compound curation review (local web app)
    chemutils.py      shared chemical utilities
  solvers/            nnls, lasso, elastic_net, omp, romp via make_solver()
  io/                 JCAMP-DX (jdx) and NIST MSP (msp) parsers and writers
  molecule/           Compound, fragment enumeration, MLIP relaxation, recombination,
                      fragment wheels (matplotlib / SVG)
  databases/          NistDatabase, InSilicoDatabase, MonaDatabase, MonaLocalDatabase,
                      MassBankDatabase, LocalSpectralDatabase (SQLite base class)
```

## Logging

rgakit logs are silent by default. Call `rgakit.setup_logging()` at the top of
a script or notebook to see progress, warnings, and errors on the console.

## License

MIT
