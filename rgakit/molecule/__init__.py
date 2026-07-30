"""
rgakit.molecule
Molecular structure and fragmentation for MS library generation.

Given a SMILES string (or any RDKit-parseable source), build a Compound,
enumerate its chemically plausible fragments, optionally relax them with an
MLIP calculator, and export the result as a SpectraLibrary ready for fitting.

Two workflows are supported:

**Fast (SMILES-only, no MLIP)**
    Fragmentation and recombination are done purely at the SMILES level.
    Fragments are embedded with MMFF for visualisation only.

    from rgakit.molecule import Compound

    mol = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
    mol.do_fragmentation(max_heavy=6)   # enumerate + H-cap + radical bonding
    mol.do_recombination()              # pairwise SMILES-level recombination
    lib = mol.to_library()              # NIST lookup for each fragment

**MLIP (optional, physics-based)**
    After the SMILES step, optionally relax fragment and recombination
    geometries with an ASE-compatible calculator (e.g. FAIRChemCalculator).

    mol = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
    mol.relax(calc)                                    # relax parent geometry
    mol.do_fragmentation(max_heavy=6)                  # SMILES-level enumeration
    mol.relax_fragments(calc, log_dir="logs/frags")    # MLIP geometry optimisation
    mol.do_recombination()                             # SMILES-level bonding
    mol.relax_recombination(calc, log_dir="logs/rec")  # MLIP-verified bond formation
    lib = mol.to_library()                             # NIST lookup for each fragment
"""

from .compound      import Compound
from .draw          import generate_fragment_wheel, generate_fragment_wheel_svg
from .utils         import (
    atoms_to_graph,
    ase_to_rdkit,
    rdkit_to_ase,
    smiles_to_inchikey,
    smiles_from_inchi,
)
from .recombination import (
    recombination_candidate,
    relax_candidate,
    relax_all_recombinations,
)

__all__ = [
    "Compound",
    "generate_fragment_wheel",
    "generate_fragment_wheel_svg",
    "atoms_to_graph",
    "ase_to_rdkit",
    "rdkit_to_ase",
    "smiles_to_inchikey",
    "smiles_from_inchi",
    "recombination_candidate",
    "relax_candidate",
    "relax_all_recombinations",
]
