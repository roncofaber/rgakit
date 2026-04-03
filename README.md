# rgakit

Python toolkit for decomposing RGA (Residual Gas Analyzer) electron-ionisation
mass spectra into known compound contributions.

Developed for synchrotron beamline ALS BL12.0.1.2 to analyse gas desorption
from X-ray irradiated perovskite thin films.

## Features

- **NNLS fitting** — non-negative least-squares decomposition against a reference library
- **Similarity metrics** — cosine, Jaccard, Pearson, spectral entropy; pairwise matrix for clustering
- **Library search** — rank candidates by any similarity metric before NNLS fitting
- **NIST library access** — fetch and cache spectra from NIST WebBook by name or CAS
- **JDX & MSP I/O** — JCAMP-DX (round-trip with metadata) and NIST MSP (multi-entry files)
- **Background correction** — linear subtraction using beam-off windows (via `clabs`)
- **Interactive HTML report** — observed vs fitted, residual, contributions, stacked breakdown
- **Smart naming** — common trivial names (Water, Methane, …) with IUPAC fallback via NCI Cactus

## Install

```bash
pip install -e .
# Optional: plotly for HTML reports
pip install plotly
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

# NNLS fit
result = lib.fit(ms)
result.summary()

# HTML report
from rgakit import generate_report
generate_report(result, library=lib, spectrum=ms, output_path="report.html")

# Pairwise similarity matrix (for clustering / classification)
mat, names = pairwise(lib, method="cosine")
# mat is a symmetric (N, N) ndarray; use with seaborn.heatmap or UMAP
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

## Workflow with clabs

```python
# Background-correct the raw RGA measurement in clabs, then extract spectrum
rga.background_correct()
ms = MassSpectrum.from_rga(rga)
ms.save_jdx("sample.jdx")
```

## File structure

```
rgakit/
  spectrum.py     — MassSpectrum (JDX, MSP, txt I/O; compare; from_rga)
  library.py      — SpectraLibrary (NNLS fit, search, from_dir/from_msp)
  result.py       — FitResult (summary table, plot)
  similarity.py   — cosine / jaccard / pearson / entropy; pairwise matrix
  background.py   — standalone linear background correction
  nomenclature.py — name resolution (common names → Cactus IUPAC → NIST)
  report.py       — interactive HTML report generation
```

## matchms comparison

rgakit uses **NNLS** for quantitative spectral decomposition — it returns how
much of each compound is present, not just which compound matches best.
[matchms](https://github.com/matchms/matchms) uses cosine scoring optimised
for LC-MS/MS (neutral losses, precursor matching) and is better suited for
metabolite identification from tandem spectra. For low-resolution EI-RGA data,
NNLS decomposition gives richer quantitative information.

rgakit borrows the cosine similarity concept from matchms as a pre-screening
step (`SpectraLibrary.search()`).
