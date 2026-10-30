import os
import sqlite3

import numpy as np
import pytest

from rgakit.databases.local import LocalSpectralDatabase

# (inchikey, smiles, exact_mass, cas, formula, name, mzs, intensities)
_FIXTURE_COMPOUNDS = [
    ("VNWKTOKETHGBQD-UHFFFAOYSA-N", "C", 16.0, "74-82-8", "CH4", "Methane", [16, 15], [100.0, 10.0]),
    ("OTMSDBZUPAUEDD-UHFFFAOYSA-N", "CC", 30.0, "74-84-0", "C2H6", "Ethane", [30, 28], [100.0, 50.0]),
    ("NEHMKBQYUWJMIP-UHFFFAOYSA-N", "CCl", 50.0, "74-87-3", "CH3Cl", "Chloromethane", [50, 15], [100.0, 20.0]),
    # rgakit.environment ENVIRONMENT_SETS["chamber_gases"], minus methane
    # (already above): lets tests exercise build_environment_library's
    # success path without the real NIST database.
    ("UFHFLCQGNIYNRP-UHFFFAOYSA-N", "[HH]", 2.0, "1333-74-0", "H2", "Hydrogen", [2, 1], [100.0, 5.0]),
    ("XLYOFNOQVPJJNP-UHFFFAOYSA-N", "O", 18.0, "7732-18-5", "H2O", "Water", [18, 17], [100.0, 23.0]),
    ("UGFAIRIUMAVXCW-UHFFFAOYSA-N", "[C-]#[O+]", 28.0, "630-08-0", "CO", "Carbon monoxide", [28, 12], [100.0, 5.0]),
    ("CURLTUGMZLYLDI-UHFFFAOYSA-N", "O=C=O", 44.0, "124-38-9", "CO2", "Carbon dioxide", [44, 22], [100.0, 1.0]),
    ("IJGRMHOSHXDMSA-UHFFFAOYSA-N", "N#N", 28.0, "7727-37-9", "N2", "Nitrogen", [28, 14], [100.0, 7.0]),
    ("MYMOFIZGZYHOMD-UHFFFAOYSA-N", "O=O", 32.0, "7782-44-7", "O2", "Oxygen", [32, 34], [100.0, 11.0]),
    ("XKRFYHLGVUSROY-UHFFFAOYSA-N", "[Ar]", 40.0, "7440-37-1", "Ar", "Argon", [40, 20], [100.0, 10.0]),
    ("QGZKDVFQNNGYKY-UHFFFAOYSA-N", "N", 17.0, "7664-41-7", "H3N", "Ammonia", [17, 1], [100.0, 80.0]),
]


def _make_fixture_db(path):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE compounds (
            inchikey TEXT PRIMARY KEY, smiles TEXT, exact_mass REAL,
            cas TEXT, molecular_formula TEXT
        );
        CREATE TABLE compound_names (
            inchikey TEXT, name TEXT, computed INTEGER
        );
        CREATE TABLE spectra (
            id TEXT, inchikey TEXT, source TEXT,
            mzs BLOB, intensities BLOB,
            precursor_mz REAL, precursor_type TEXT,
            ionization_mode TEXT, ms_level TEXT,
            instrument_type TEXT, instrument TEXT, collision_energy TEXT,
            splash TEXT, score REAL
        );
    """)

    for ik, smi, mass, cas, formula, name, mzs, intens in _FIXTURE_COMPOUNDS:
        conn.execute(
            "INSERT INTO compounds (inchikey, smiles, exact_mass, cas, molecular_formula) "
            "VALUES (?, ?, ?, ?, ?)",
            (ik, smi, mass, cas, formula),
        )
        conn.execute(
            "INSERT INTO compound_names (inchikey, name, computed) VALUES (?, ?, 0)",
            (ik, name),
        )
        mzs_blob = np.asarray(mzs, dtype=np.float64).tobytes()
        int_blob = np.asarray(intens, dtype=np.float64).tobytes()
        conn.execute(
            "INSERT INTO spectra (id, inchikey, source, mzs, intensities) VALUES (?, ?, ?, ?, ?)",
            (ik, ik, "fixture", mzs_blob, int_blob),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def fixture_db(tmp_path):
    path = tmp_path / "fixture.db"
    _make_fixture_db(str(path))
    return LocalSpectralDatabase(str(path))


@pytest.fixture(scope="session")
def nist_db_path():
    path = os.environ.get("RGAKIT_NIST_DB")
    if not path or not os.path.exists(path):
        pytest.skip("set RGAKIT_NIST_DB to a nist_library.db to run")
    return path
