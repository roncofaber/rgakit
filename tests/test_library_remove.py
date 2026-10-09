import numpy as np
import pytest

from rgakit.spectrum import MassSpectrum
from rgakit.library import SpectraLibrary


def _spec(name, inchikey):
    return MassSpectrum(
        mz=np.array([28, 44]),
        intensity=np.array([1.0, 0.5]),
        name=name,
        metadata={"inchikey": inchikey},
    )


def test_remove_by_inchikey_removes_matching_spectrum():
    lib = SpectraLibrary([
        _spec("Nitrogen", "IJGRMHOSHXDMSA-UHFFFAOYSA-N"),
        _spec("Carbon dioxide", "CURLTUGMZLYLDI-UHFFFAOYSA-N"),
    ])

    removed = lib.remove_by_inchikey("CURLTUGMZLYLDI-UHFFFAOYSA-N")

    assert removed is True
    assert "Carbon dioxide" not in lib.names()
    assert "Nitrogen" in lib.names()
    assert len(lib) == 1


def test_remove_by_inchikey_returns_false_when_not_found():
    lib = SpectraLibrary([_spec("Nitrogen", "IJGRMHOSHXDMSA-UHFFFAOYSA-N")])

    removed = lib.remove_by_inchikey("DOES-NOT-EXIST")

    assert removed is False
    assert len(lib) == 1


def test_remove_by_inchikey_ignores_spectra_without_metadata():
    spec = MassSpectrum(mz=np.array([28]), intensity=np.array([1.0]), name="Unknown")
    lib = SpectraLibrary([spec])

    removed = lib.remove_by_inchikey("IJGRMHOSHXDMSA-UHFFFAOYSA-N")

    assert removed is False
    assert len(lib) == 1


def test_omp_random_state_reproducible():
    import numpy as np
    from rgakit.solvers.omp import make_omp

    rng = np.random.default_rng(0)
    A = rng.random((50, 10))
    w_true = np.zeros(10)
    w_true[[1, 4]] = [0.8, 0.5]
    y = A @ w_true + 0.01 * rng.random(50)

    s1 = make_omp(n_trials=20, temperature=0.5, random_state=42)
    s2 = make_omp(n_trials=20, temperature=0.5, random_state=42)
    w1, _ = s1(A, y)
    w2, _ = s2(A, y)
    assert np.allclose(w1, w2)
    assert not np.allclose(w1, w_true)


def test_fit_random_state_reproducible():
    import numpy as np
    from rgakit.spectrum import MassSpectrum
    from rgakit.library import SpectraLibrary

    rng = np.random.default_rng(1)
    mzs = [np.array([16, 28, 32, 44]),
           np.array([15, 29, 43, 58]),
           np.array([27, 41, 55, 70])]
    specs = []
    for i, mz in enumerate(mzs):
        specs.append(MassSpectrum(
            mz=mz,
            intensity=np.array([1.0, 0.7, 0.4, 0.3]),
            name=f"c{i}",
            metadata={"inchikey": f"IK{i}"},
        ))
    lib = SpectraLibrary(specs)

    y_mz = np.array([16, 28, 32, 15, 29, 43, 58, 27, 41])
    y_in = np.array([1.0, 0.9, 0.5, 0.8, 0.6, 0.5, 0.2, 0.3, 0.2])

    kw = dict(method="omp", n_trials=10, temperature=0.5, random_state=3)
    r1 = lib.fit(y_mz, y_in, **kw)
    r2 = lib.fit(y_mz, y_in, **kw)
    assert all(np.allclose(v, r2.weights[k]) for k, v in r1.weights.items())
