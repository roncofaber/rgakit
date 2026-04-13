"""
rgakit.molecule
Molecular structure and fragmentation for MS library generation.

Given a SMILES string (or any RDKit-parseable source), build a Compound,
enumerate its chemically plausible fragments, optionally relax them with an
MLIP calculator, and export the result as a SpectraLibrary ready for fitting.

Two workflows are supported:

**Fast (SMILES-only, no MLIP)**
    All fragmentation and recombination is done purely at the SMILES level —
    no 3-D geometry optimisation.  Fragments are embedded with MMFF for
    visualisation only.  ``do_recombination()`` is redundant here because
    ``do_fragmentation()`` already includes pairwise radical bonding.

    from rgakit.molecule import Compound

    mol = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
    mol.do_fragmentation(max_heavy=6)   # H-cap + radical bonding + rearrangements (SMILES)
    lib = mol.to_library()   # NIST lookup for each fragment

**MLIP (slow, physics-based)**
    Parent and each fragment are relaxed with an ASE-compatible MLIP
    calculator.  Recombination uses a two-phase constrained + unconstrained
    FIRE2 run to form new bonds.  Requires ``fairchem`` or a similar framework.

    mol = Compound("CC(C)(C)OC(=O)CC[NH3+].[I-]", name="EAI")
    mol.relax(calc)                                          # relax parent geometry
    mol.do_fragmentation(calc, hcap=True, max_heavy=6)      # relax raw radicals + H-capped versions
    mol.do_recombination(calc)             # MLIP recombination products
    lib = mol.to_library()                 # NIST lookup for each fragment
"""

from .compound      import Compound
from ._utils        import (
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
    "atoms_to_graph",
    "ase_to_rdkit",
    "rdkit_to_ase",
    "smiles_to_inchikey",
    "smiles_from_inchi",
    "recombination_candidate",
    "relax_candidate",
    "relax_all_recombinations",
]
