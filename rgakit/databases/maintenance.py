"""
maintenance.py
--------------
Database health checks and repair utilities.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def diagnose(db_path: str | Path) -> dict:
    """
    Run health checks on a spectral library database.

    Returns a dict with counts of issues found.
    """
    conn = sqlite3.connect(str(db_path))
    results: dict[str, int] = {}

    results["compounds"] = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]

    results["names"] = conn.execute("SELECT COUNT(*) FROM compound_names").fetchone()[0]

    results["no_name"] = conn.execute("""
        SELECT COUNT(DISTINCT c.inchikey) FROM compounds c
        LEFT JOIN compound_names cn ON c.inchikey = cn.inchikey
        WHERE cn.id IS NULL
    """).fetchone()[0]

    results["lookup_failed"] = conn.execute(
        "SELECT COUNT(DISTINCT inchikey) FROM compound_names WHERE computed = 2"
    ).fetchone()[0]

    results["no_smiles"] = conn.execute(
        "SELECT COUNT(*) FROM compounds WHERE smiles IS NULL"
    ).fetchone()[0]

    results["bad_newline"] = conn.execute(
        r"""SELECT COUNT(*) FROM compound_names
            WHERE name LIKE '%' || char(10) || '%'
               OR INSTR(name, '\n') > 0"""
    ).fetchone()[0]

    results["no_formula"] = conn.execute(
        "SELECT COUNT(*) FROM compounds WHERE molecular_formula IS NULL"
    ).fetchone()[0]

    conn.close()

    print(f"Database: {db_path}")
    print(f"  Compounds:        {results['compounds']:>10,}")
    print(f"  Name entries:     {results['names']:>10,}")
    print(f"  No name:          {results['no_name']:>10,}")
    print(f"  Lookup failed:    {results.get('lookup_failed', 0):>10,}")
    print(f"  No SMILES:        {results['no_smiles']:>10,}")
    print(f"  No formula:       {results['no_formula']:>10,}")
    print(f"  Bad newlines:     {results['bad_newline']:>10,}")
    return results


def fix_smiles_from_inchi(db_path: str | Path) -> int:
    """Populate missing SMILES from InChI (RDKit first, Open Babel fallback)."""
    from ..chemutils import smiles_from_inchi

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute(
        "SELECT inchikey, inchi FROM compounds "
        "WHERE smiles IS NULL AND inchi IS NOT NULL"
    ).fetchall()

    updated = 0
    for inchikey, inchi in rows:
        smi = smiles_from_inchi(inchi)
        if smi:
            conn.execute("UPDATE compounds SET smiles = ? WHERE inchikey = ?",
                         (smi, inchikey))
            updated += 1

    conn.commit()
    conn.close()
    logger.info("fix_smiles_from_inchi: updated %d/%d compounds.", updated, len(rows))
    return updated


def fix_split_names(db_path: str | Path) -> int:
    """Split newline-concatenated name entries into individual rows."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    bad_rows = conn.execute(
        r"""SELECT id, inchikey, name, computed FROM compound_names
            WHERE name LIKE '%' || char(10) || '%'
               OR INSTR(name, '\n') > 0"""
    ).fetchall()

    added = 0
    for row_id, inchikey, name_blob, computed in bad_rows:
        names = [n.strip() for n in name_blob.replace("\\n", "\n").split("\n") if n.strip()]
        if len(names) <= 1:
            continue
        # Keep the first name in the original row
        conn.execute("UPDATE compound_names SET name = ? WHERE id = ?",
                     (names[0], row_id))
        # Insert the rest as new rows
        for extra in names[1:]:
            conn.execute(
                "INSERT INTO compound_names (inchikey, name, computed) VALUES (?, ?, ?)",
                (inchikey, extra, computed),
            )
            added += 1

    conn.commit()
    conn.close()
    logger.info("fix_split_names: split %d entries, added %d new rows.", len(bad_rows), added)
    return added


def fix_missing_names(db_path: str | Path, use_pubchem: bool = False,
                      batch_size: int = 500) -> int:
    """
    Fill missing compound names.

    Uses molecular formula as a computed name by default.
    If *use_pubchem* is True, attempts PubChem lookup for real names
    (slow — rate-limited to ~5 req/s).
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    nameless = conn.execute("""
        SELECT c.inchikey, c.smiles, c.molecular_formula
        FROM compounds c
        LEFT JOIN compound_names cn ON c.inchikey = cn.inchikey
        WHERE cn.id IS NULL
    """).fetchall()

    if not nameless:
        logger.info("fix_missing_names: no missing names.")
        conn.close()
        return 0

    added = 0
    for inchikey, smiles, formula in nameless:
        name = None
        computed = 1

        if use_pubchem and smiles:
            try:
                import pubchempy as pcp
                results = pcp.get_compounds(smiles, "smiles")
                if results and results[0].iupac_name:
                    name = results[0].iupac_name
                    computed = 0
            except Exception:
                pass

        if name is None:
            name = formula or smiles or inchikey

        conn.execute(
            "INSERT INTO compound_names (inchikey, name, computed) VALUES (?, ?, ?)",
            (inchikey, name, computed),
        )
        added += 1

        if added % batch_size == 0:
            conn.commit()
            logger.info("fix_missing_names: %d / %d …", added, len(nameless))

    conn.commit()
    conn.close()
    logger.info("fix_missing_names: added %d names.", added)
    return added


def validate_chemistry(db_path: str | Path, fix: bool = False,
                       report_path: str | Path | None = None) -> dict:
    """
    Validate SMILES ↔ InChIKey consistency for every compound.

    Checks
    ------
    1. SMILES parses in RDKit
    2. InChIKey regenerated from SMILES matches the stored InChIKey
    3. Molecular formula regenerated from SMILES matches stored formula
    4. Exact mass regenerated from SMILES matches stored mass

    Parameters
    ----------
    db_path : path to the database
    fix     : if True, update the database with corrected values where
              the SMILES is valid but other fields are wrong/missing.

    Returns
    -------
    dict with counts: total, valid, invalid_smiles, ik_mismatch,
    formula_mismatch, mass_mismatch, fixed.
    """
    from ..chemutils import (inchi_from_smiles, inchikey_from_inchi,
                              formula_from_smiles, mol_weight_from_smiles)
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute(
        "SELECT inchikey, smiles, molecular_formula, exact_mass, inchi "
        "FROM compounds"
    ).fetchall()

    counts = {
        "total": len(rows),
        "valid": 0,
        "no_smiles": 0,
        "invalid_smiles": 0,
        "ik_mismatch": 0,
        "formula_mismatch": 0,
        "mass_mismatch": 0,
        "isotope_labeled": 0,
        "fixed": 0,
    }

    issues = []   # (inchikey, smiles, issue_type, stored_value, expected_value)

    for stored_ik, smiles, stored_formula, stored_mass, stored_inchi in rows:
        if not smiles:
            counts["no_smiles"] += 1
            issues.append((stored_ik, "", "no_smiles", "", ""))
            continue

        regen_inchi  = inchi_from_smiles(smiles)
        regen_ik     = inchikey_from_inchi(regen_inchi) if regen_inchi else None
        regen_formula = formula_from_smiles(smiles)
        regen_mass   = mol_weight_from_smiles(smiles)

        if regen_inchi is None:
            counts["invalid_smiles"] += 1
            issues.append((stored_ik, smiles, "invalid_smiles", smiles, ""))
            continue

        is_isotope = bool(stored_inchi and "/i" in stored_inchi)
        if is_isotope:
            counts["isotope_labeled"] += 1

        is_valid = True
        updates = {}

        # Check InChIKey (connectivity layer only — first 14 chars)
        if regen_ik and stored_ik and regen_ik != stored_ik:
            if regen_ik[:14] != stored_ik[:14]:
                counts["ik_mismatch"] += 1
                issues.append((stored_ik, smiles, "ik_mismatch", stored_ik, regen_ik))
                is_valid = False

        # Check formula (skip isotope-labeled — D vs H is a convention difference)
        if not is_isotope:
            if stored_formula and regen_formula and stored_formula != regen_formula:
                counts["formula_mismatch"] += 1
                issues.append((stored_ik, smiles, "formula_mismatch",
                               stored_formula, regen_formula))
                if fix:
                    updates["molecular_formula"] = regen_formula

            if not stored_formula and regen_formula and fix:
                updates["molecular_formula"] = regen_formula

        # Check mass (skip isotope-labeled — stored mass uses D=2.014,
        # RDKit MolWt uses average atomic mass which differs)
        if not is_isotope and stored_mass and regen_mass:
            if abs(stored_mass - regen_mass) > 0.5:
                counts["mass_mismatch"] += 1
                issues.append((stored_ik, smiles, "mass_mismatch",
                               f"{stored_mass:.4f}", f"{regen_mass:.4f}"))
                if fix:
                    updates["exact_mass"] = regen_mass

        if not stored_mass and regen_mass and fix:
            updates["exact_mass"] = regen_mass

        if is_valid:
            counts["valid"] += 1

        if fix and updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE compounds SET {set_clause} WHERE inchikey = ?",
                list(updates.values()) + [stored_ik],
            )
            counts["fixed"] += 1

    if fix:
        conn.commit()
    conn.close()

    # Print summary
    print(f"Chemistry validation: {db_path}")
    print(f"  Total compounds:     {counts['total']:>8,}")
    print(f"  Valid:               {counts['valid']:>8,}")
    print(f"  No SMILES:           {counts['no_smiles']:>8,}")
    print(f"  Invalid SMILES:      {counts['invalid_smiles']:>8,}")
    print(f"  Isotope-labeled:     {counts['isotope_labeled']:>8,}")
    print(f"  InChIKey mismatch:   {counts['ik_mismatch']:>8,}")
    print(f"  Formula mismatch:    {counts['formula_mismatch']:>8,}")
    print(f"  Mass mismatch:       {counts['mass_mismatch']:>8,}")
    if fix:
        print(f"  Fixed:               {counts['fixed']:>8,}")

    # Write CSV report
    if report_path is not None:
        import csv
        report_path = Path(report_path)

        # Fetch names for the report
        conn = sqlite3.connect(str(db_path))
        name_lookup = {}
        for ik, *_ in issues:
            if ik and ik not in name_lookup:
                row = conn.execute(
                    "SELECT name FROM compound_names WHERE inchikey = ? LIMIT 1",
                    (ik,),
                ).fetchone()
                name_lookup[ik] = row[0] if row else ""
        conn.close()

        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["inchikey", "name", "smiles", "issue", "stored", "expected"])
            for ik, smi, issue, stored, expected in issues:
                writer.writerow([ik, name_lookup.get(ik, ""), smi, issue, stored, expected])

        print(f"  Report: {report_path} ({len(issues)} issues)")

    return counts


def validate_against_nist(
    db_path:     str | Path,
    report_path: str | Path | None = None,
    limit:       int | None = None,
    only_flagged: bool = True,
) -> list[dict]:
    """
    Cross-reference local compounds against the NIST WebBook server.

    Parameters
    ----------
    db_path      : path to the local database
    report_path  : CSV output path (optional)
    limit        : max compounds to check (None = all)
    only_flagged : if True (default), only check compounds that have
                   formula or mass mismatches from validate_chemistry.
                   If False, check all compounds with a CAS number.

    Returns
    -------
    list of dicts with mismatches found.
    """
    import time
    try:
        import nistchempy as nist
    except ImportError:
        raise ImportError("nistchempy is required: pip install nistchempy")

    conn = sqlite3.connect(str(db_path))

    if only_flagged:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Descriptors, rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")

        all_rows = conn.execute(
            "SELECT inchikey, smiles, molecular_formula, exact_mass, cas, inchi "
            "FROM compounds WHERE smiles IS NOT NULL AND cas IS NOT NULL"
        ).fetchall()

        flagged = []
        for ik, smi, formula, mass, cas, inchi_str in all_rows:
            if inchi_str and "/i" in inchi_str:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                flagged.append((ik, smi, formula, mass, cas))
                continue
            regen_f = rdMolDescriptors.CalcMolFormula(mol)
            regen_m = Descriptors.MolWt(mol)
            if (formula and regen_f and formula != regen_f) or \
               (mass and abs(mass - regen_m) > 0.5):
                flagged.append((ik, smi, formula, mass, cas))
        rows = flagged
        logger.info("validate_against_nist: %d flagged compounds to check.", len(rows))
    else:
        rows = conn.execute(
            "SELECT inchikey, smiles, molecular_formula, exact_mass, cas "
            "FROM compounds WHERE cas IS NOT NULL"
        ).fetchall()

    if limit:
        rows = rows[:limit]

    print(f"Checking {len(rows)} compounds against NIST WebBook …")

    mismatches = []
    checked = 0
    for row in rows:
        ik, smi, stored_formula, stored_mass, cas = row[:5]

        try:
            nc = nist.get_compound(cas)
        except Exception as e:
            mismatches.append({
                "inchikey": ik, "cas": cas, "issue": "nist_lookup_failed",
                "detail": str(e),
            })
            time.sleep(0.5)
            continue

        if nc is None:
            mismatches.append({
                "inchikey": ik, "cas": cas, "issue": "not_on_nist",
            })
            time.sleep(0.3)
            continue

        issues = []

        # Compare InChIKey
        if nc.inchi_key and ik and nc.inchi_key[:14] != ik[:14]:
            issues.append(("ik_mismatch", ik, nc.inchi_key))

        # Compare formula
        if nc.formula and stored_formula and nc.formula != stored_formula:
            issues.append(("formula", stored_formula, nc.formula))

        # Compare mass
        if nc.mol_weight and stored_mass:
            if abs(nc.mol_weight - stored_mass) > 0.5:
                issues.append(("mass", f"{stored_mass:.4f}", f"{nc.mol_weight:.4f}"))

        # Compare SMILES → formula against NIST formula
        if nc.formula and smi:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rdkit_f = rdMolDescriptors.CalcMolFormula(mol)
                if rdkit_f != nc.formula:
                    issues.append(("smiles_vs_nist_formula", rdkit_f, nc.formula))

        for issue_type, local_val, nist_val in issues:
            mismatches.append({
                "inchikey": ik, "cas": cas, "smiles": smi,
                "nist_name": nc.name,
                "issue": issue_type,
                "local": local_val, "nist": nist_val,
            })

        checked += 1
        if checked % 20 == 0:
            print(f"  {checked}/{len(rows)} checked, {len(mismatches)} issues …")
        time.sleep(0.3)

    print(f"\nDone: {checked} checked, {len(mismatches)} issues found.")

    if report_path is not None:
        import csv
        with open(report_path, "w", newline="") as f:
            fields = ["inchikey", "cas", "smiles", "nist_name",
                       "issue", "local", "nist", "detail"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(mismatches)
        print(f"Report: {report_path}")

    return mismatches


def fix_all(db_path: str | Path, use_pubchem: bool = False) -> None:
    """Run all fixes in order."""
    print(f"\n{'='*60}")
    print(f"Fixing: {db_path}")
    print(f"{'='*60}")

    diagnose(db_path)

    n1 = fix_smiles_from_inchi(db_path)
    if n1: print(f"  Fixed {n1} missing SMILES from InChI")

    n2 = fix_split_names(db_path)
    if n2: print(f"  Split {n2} newline-concatenated names")

    n3 = fix_missing_names(db_path, use_pubchem=use_pubchem)
    if n3: print(f"  Added {n3} missing names")

    print("\nAfter fixes:")
    diagnose(db_path)
