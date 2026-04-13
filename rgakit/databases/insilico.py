"""
insilico.py
Interface to the in-silico EI-MS SQLite spectral library.

The database uses the shared spectral library schema (schema.sql) with
``compounds`` and ``spectra`` tables.  Peaks are stored as numpy float64
binary blobs.

Two usage modes:

**Lookup by structure** — retrieve the predicted spectrum for a known compound:

    from rgakit import InSilicoDatabase

    db = InSilicoDatabase("/path/to/spectral_libraries/in_silico_library.db")
    spec = db.get(smiles="CCO")       # ethanol by SMILES
    spec = db.get(cas="64-17-5")      # by CAS number
    spec = db.get(name="ethanol")     # by name
    spec = db.get(inchi="InChI=1S/…") # by InChI string

**Reverse lookup by spectrum** — find candidate compounds from an experimental
spectrum (requires the HNSW index and Word2Vec model files):

    db = InSilicoDatabase(
        "/path/to/spectral_libraries/in_silico_library.db",
        index_path="/path/to/spectral_libraries/references_index.bin",
        model_path="/path/to/references_word2vec.model",
    )
    hits = db.search_by_spectrum(experimental_spectrum, k=10)
    for spec, distance in hits:
        print(spec.name, distance)

The HNSW index maps to the ``hnsw_idx`` column in the ``spectra`` table.
Run ``build_hnsw.py`` once after populating the database to build the index
and populate ``hnsw_idx``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .local import LocalSpectralDatabase

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_NAME = "references_index.bin"
_DEFAULT_MODEL_NAME = "references_word2vec.model"
_HNSW_DIM           = 500

_HNSW_SQL = """
    SELECT s.hnsw_idx, s.id, c.smiles, c.exact_mass, s.mzs, s.intensities
    FROM   spectra s
    JOIN   compounds c ON s.inchikey = c.inchikey
    WHERE  s.source   = 'in_silico'
    AND    s.hnsw_idx IN ({placeholders})
"""

from rgakit.molecule._utils import smiles_to_inchikey, smiles_from_inchi  # noqa: E402


def _smiles_from_pubchem(identifier: str, namespace: str = "name") -> str | None:
    """Resolve a name or CAS number to SMILES via PubChem."""
    try:
        import pubchempy as pcp
        compounds = pcp.get_compounds(identifier, namespace)
        if compounds:
            smi = compounds[0].isomeric_smiles
            logger.debug("PubChem resolved %r → %s", identifier, smi)
            return smi
        logger.debug("PubChem: no match for %r (%s)", identifier, namespace)
    except Exception as exc:
        logger.debug("PubChem lookup failed for %r: %s", identifier, exc)
    return None


class InSilicoDatabase(LocalSpectralDatabase):
    """
    Read-only interface to the in-silico EI-MS SQLite spectral library.

    Parameters
    ----------
    db_path    : path to ``in_silico_library.db`` (new shared schema)
    index_path : path to the HNSW index file (built by ``build_hnsw.py``).
                 Defaults to ``references_index.bin`` in the same directory.
    model_path : path to the Word2Vec model file.
                 Defaults to ``references_word2vec.model`` in the same directory.
    """

    _SOURCE = "in_silico"

    def __init__(
        self,
        db_path:    str | Path,
        index_path: str | Path | None = None,
        model_path: str | Path | None = None,
    ):
        super().__init__(db_path)
        db_dir = self._db_path.parent
        self._index_path = Path(index_path) if index_path else db_dir / _DEFAULT_INDEX_NAME
        self._model_path = Path(model_path) if model_path else db_dir.parent / _DEFAULT_MODEL_NAME
        self._hnsw      = None
        self._w2v_model = None

    # ------------------------------------------------------------------
    # Override _to_spectrum for in-silico specifics
    # ------------------------------------------------------------------

    def _to_spectrum(self, row: tuple, label: str | None = None):
        """
        In-silico variant: convert mz floats to integers and rescale
        intensity if stored in 0-1 normalized form.
        """
        from rgakit.spectrum import MassSpectrum

        (spec_id, inchikey, source,
         smiles, exact_mass, cas,
         mzs_blob, int_blob, *_rest) = row

        mz        = np.frombuffer(mzs_blob, dtype=np.float64).astype(int)
        intensity = np.frombuffer(int_blob,  dtype=np.float64)

        if intensity.size == 0 or intensity.max() == 0:
            return None
        if intensity.max() <= 1.0:
            intensity = intensity * 999.0

        name = label or self._name_for(inchikey, fallback=smiles or spec_id)

        return MassSpectrum(
            mz        = mz,
            intensity = intensity,
            name      = name,
            metadata  = {
                "source":   source,
                "smiles":   smiles,
                "mw":       exact_mass,
                "inchikey": inchikey,
                "compid":   spec_id,
            },
        )

    # ------------------------------------------------------------------
    # Decode HNSW result rows (different column set)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_hnsw_row(row: tuple, label: str | None = None):
        """Convert an HNSW result row (id, smiles, exact_mass, mzs, ints)."""
        from rgakit.spectrum import MassSpectrum

        spec_id, smiles, exact_mass, mzs_blob, int_blob = row
        mz        = np.frombuffer(mzs_blob, dtype=np.float64).astype(int)
        intensity = np.frombuffer(int_blob,  dtype=np.float64)

        if intensity.max() <= 1.0:
            intensity = intensity * 999.0

        inchikey = smiles_to_inchikey(smiles) if smiles else None
        return MassSpectrum(
            mz        = mz,
            intensity = intensity,
            name      = label or smiles,
            metadata  = {
                "smiles":   smiles,
                "mw":       exact_mass,
                "compid":   spec_id,
                "source":   "in_silico",
                "inchikey": inchikey,
            },
        )

    # ------------------------------------------------------------------
    # Extended get() — adds inchi support and PubChem fallback
    # ------------------------------------------------------------------

    def get(
        self,
        smiles:   str | None = None,
        inchikey: str | None = None,
        cas:      str | None = None,
        name:     str | None = None,
        inchi:    str | None = None,
    ):
        """
        Retrieve a MassSpectrum by structure identifier.

        Provide exactly one of *smiles*, *inchikey*, *cas*, *name*, or *inchi*.
        Returns a :class:`~rgakit.spectrum.MassSpectrum` or ``None``.

        For *name* and *cas*, the database is searched first; if nothing is
        found, PubChem is queried to resolve the identifier to a SMILES.
        """
        # inchi: convert to smiles, then delegate
        if inchi is not None:
            smi = smiles_from_inchi(inchi)
            if smi is None:
                logger.warning("Could not convert InChI to SMILES: %r", inchi)
                return None
            return super().get(smiles=smi)

        # name / cas: try SQL first, then PubChem fallback
        if name is not None or cas is not None:
            result = super().get(name=name, cas=cas)
            if result is not None:
                return result
            key   = "name" if name is not None else "cas"
            value = name   if name is not None else cas
            logger.info("Falling back to PubChem for %s=%r …", key, value)
            smi = _smiles_from_pubchem(value, namespace=key)
            if smi is None:
                logger.warning("PubChem could not resolve %s %r", key, value)
                return None
            return super().get(smiles=smi)

        return super().get(smiles=smiles, inchikey=inchikey)

    # ------------------------------------------------------------------
    # HNSW vector search
    # ------------------------------------------------------------------

    def _ensure_search_index(self) -> None:
        if self._hnsw is not None:
            return

        try:
            import gensim.models
        except ImportError:
            raise ImportError("gensim is required for spectrum search: pip install gensim")
        try:
            import hnswlib
        except ImportError:
            raise ImportError("hnswlib is required for spectrum search: pip install hnswlib")

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Word2Vec model not found: {self._model_path}\n"
                "Pass model_path= explicitly or run build_hnsw.py."
            )
        if not self._index_path.exists():
            raise FileNotFoundError(
                f"HNSW index not found: {self._index_path}\n"
                "Run build_hnsw.py to build the index."
            )

        logger.info("Loading Word2Vec model from %s …", self._model_path)
        self._w2v_model = gensim.models.Word2Vec.load(str(self._model_path))

        n_elements = self.conn.execute(
            "SELECT COUNT(*) FROM spectra WHERE source='in_silico' AND hnsw_idx IS NOT NULL"
        ).fetchone()[0]

        logger.info("Loading HNSW index from %s …", self._index_path)
        self._hnsw = hnswlib.Index(space="l2", dim=_HNSW_DIM)
        self._hnsw.load_index(str(self._index_path), max_elements=n_elements)
        logger.info("Search index ready (%d vectors).", self._hnsw.element_count)

    def _embed(self, spectrum) -> np.ndarray | None:
        model = self._w2v_model
        mz    = spectrum.mz.astype(float)
        inten = spectrum.intensity.astype(float)
        inten = inten / inten.max()
        keep  = inten > 0.001
        mz, inten = mz[keep], inten[keep]

        vec = np.zeros(model.wv.vector_size, dtype=np.float64)
        for m, w in zip(mz, inten):
            word = f"peak@{m:.0f}"
            if word in model.wv.key_to_index:
                vec += (w ** 0.5) * model.wv[word]

        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return (vec / norm).astype(np.float32)

    def search_by_spectrum(self, spectrum, k: int = 10, ef: int = 300) -> list[tuple]:
        """
        Find the *k* most similar in-silico spectra using vector search.

        Parameters
        ----------
        spectrum : MassSpectrum
        k        : number of nearest neighbours (default 10)
        ef       : HNSW search accuracy parameter, must be > k (default 300)

        Returns
        -------
        list of ``(MassSpectrum, float)`` tuples, sorted by ascending distance.
        """
        self._ensure_search_index()

        vec = self._embed(spectrum)
        if vec is None:
            logger.warning(
                "Could not embed spectrum — no peaks matched the Word2Vec vocabulary."
            )
            return []

        self._hnsw.set_ef(max(ef, k + 1))
        indices, distances = self._hnsw.knn_query(vec.reshape(1, -1), k=k)

        hnsw_ids     = [int(i) for i in indices[0]]
        placeholders = ",".join("?" * len(hnsw_ids))
        rows         = self.conn.execute(
            _HNSW_SQL.format(placeholders=placeholders), hnsw_ids
        ).fetchall()

        # rows: (hnsw_idx, id, smiles, exact_mass, mzs_blob, int_blob)
        row_by_idx = {row[0]: row[1:] for row in rows}

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            row = row_by_idx.get(int(idx))
            if row:
                results.append((self._decode_hnsw_row(row), float(dist)))

        logger.info("search_by_spectrum() → %d hits (best distance %.4f)", len(results),
                    results[0][1] if results else float("nan"))
        return results

    def __repr__(self) -> str:
        return f"InSilicoDatabase({str(self._db_path)!r})"
