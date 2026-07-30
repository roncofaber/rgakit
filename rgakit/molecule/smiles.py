"""
smiles.py
SMILES-level radical and ion chemistry helpers.

Provides pure-SMILES functions used during EI fragmentation:
radical capping, anion-to-radical conversion, radical combination,
beta-scission, and H-migration enumeration.

Property extraction (mass, charge) and format conversions live in
:mod:`.utils`.
"""

from __future__ import annotations

def hcap_smiles(smi: str | None) -> str | None:
    """
    Convert a radical SMILES to a H-saturated closed-shell SMILES.

    Zeros out all radical-electron counts so RDKit fills the open valences
    with implicit H during sanitization. Returns None on failure.
    """
    if smi is None:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


def anion_to_radical(smi: str | None) -> str | None:
    """
    Convert a closed-shell anion SMILES to its neutral radical form.

    Removes the formal negative charge on each negatively-charged atom and
    adds the same number of radical electrons in its place.  This models the
    EI step of stripping an electron from an anion (e.g. [I-] becomes [I•])
    so that the resulting radical can participate in covalent recombination
    with other radical fragments.

    Only processes anions (net charge < 0). Returns None for cations,
    neutral species, or on RDKit failure.

    Examples
    --------
    "[I-]"  gives "[I]"   (one radical electron)
    "[Cl-]" gives "[Cl]"
    "[O-2]" gives "[O]"   (two radical electrons)
    """
    if smi is None:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if Chem.GetFormalCharge(mol) >= 0:
        return None  # not an anion

    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        q = atom.GetFormalCharge()
        if q < 0:
            atom.SetFormalCharge(0)
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + abs(q))
    try:
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(rw)
    except Exception:
        return None


def combine_radicals(smi_a: str | None, smi_b: str | None) -> list[str]:
    """
    Try to form bonds between every pair of radical centres in smi_a and smi_b.

    For each (centre_a, centre_b) pair the bond order is determined by how many
    radical electrons both centres share:
      - 1 shared electron on each side: SINGLE bond
      - 2 shared electrons on each side: also try DOUBLE bond
      - 3 shared electrons on each side: also try TRIPLE bond

    Each attempt reduces the radical count on both centres by the bond order and
    adds the corresponding bond.  Products that fail RDKit sanitization are
    silently skipped.  Returns a list of unique canonical product SMILES.
    """
    if smi_a is None or smi_b is None:
        return []
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return []

    centers_a = [(a.GetIdx(), a.GetNumRadicalElectrons())
                 for a in mol_a.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    centers_b = [(a.GetIdx(), a.GetNumRadicalElectrons())
                 for a in mol_b.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if not centers_a or not centers_b:
        return []

    _BOND = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }

    n_a      = mol_a.GetNumAtoms()
    combined = AllChem.CombineMols(mol_a, mol_b)
    products: set[str] = set()

    for ca, rad_a in centers_a:
        for cb, rad_b in centers_b:
            max_order = min(rad_a, rad_b, 3)
            for order in range(1, max_order + 1):
                bond_type = _BOND[order]
                rw = Chem.RWMol(combined)
                rw.AddBond(ca, cb + n_a, bond_type)
                rw.GetAtomWithIdx(ca).SetNumRadicalElectrons(rad_a - order)
                rw.GetAtomWithIdx(cb + n_a).SetNumRadicalElectrons(rad_b - order)
                try:
                    Chem.SanitizeMol(rw)
                    products.add(Chem.MolToSmiles(rw))
                except Exception:
                    pass

    return list(products)


def deprotonate_smiles(smi: str | None) -> str | None:
    """
    Remove one proton from the first protonated heteroatom (N, O, or S with
    formal charge > 0 and at least one H neighbour).

    Models proton loss in EI-MS: e.g. [NH3+] fragments give their neutral amine
    counterpart [NH2], which is what EI-MS databases index.  Only acts on
    cationic molecules (net charge > 0).  Returns None when no suitable atom is
    found or RDKit sanitisation fails.

    Examples
    --------
    "CC[NH3+]"            gives "CCN"
    "CC(C)(C)OC(=O)CC[NH3+]"  gives "CC(C)(C)OC(=O)CCNH2"
    "[NH4+]"              gives "N"
    """
    if smi is None:
        return None
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None or Chem.GetFormalCharge(mol) <= 0:
        return None

    mol_h = Chem.AddHs(mol)
    rw    = Chem.RWMol(mol_h)

    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() not in (7, 8, 16):
            continue
        if atom.GetFormalCharge() <= 0:
            continue
        h_nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if not h_nbrs:
            continue

        heavy = atom.GetIdx()
        h     = h_nbrs[0]

        # Tag the heteroatom before removal so we can re-find it after
        # indices shift (RemoveAtom invalidates all indices above the removed one).
        rw.GetAtomWithIdx(heavy).SetAtomMapNum(99)
        rw.RemoveAtom(h)
        for a in rw.GetAtoms():
            if a.GetAtomMapNum() == 99:
                a.SetFormalCharge(a.GetFormalCharge() - 1)
                a.SetNoImplicit(False)
                a.SetAtomMapNum(0)
                break

        try:
            Chem.SanitizeMol(rw)
            return Chem.MolToSmiles(Chem.RemoveHs(rw))
        except Exception:
            return None

    return None


def beta_scission_products(smi: str | None) -> list[str]:
    """
    Enumerate β-scission products of a radical SMILES.

    For each radical centre α and each single bond α–β–γ:
      - Break the β–γ bond
      - Upgrade α–β to a double bond (radical uses its electron to form π)
      - γ gains a radical electron

    The fragments produced may themselves be radicals; those are H-capped
    before being returned.  Products identical to the original H-capped SMILES
    are not returned (no new information).

    Returns a list of unique canonical product SMILES (closed-shell).
    """
    if smi is None:
        return []
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    # Make H explicit so that C–H bonds appear as β–γ scission targets.
    # AddHs appends H atoms at the end, so existing heavy-atom indices are stable.
    mol = Chem.AddHs(mol)

    radicals = [(a.GetIdx(), a.GetNumRadicalElectrons())
                for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if not radicals:
        return []

    original_hcap = hcap_smiles(smi)
    products: set[str] = set()

    for alpha, alpha_rad in radicals:
        for beta_atom in mol.GetAtomWithIdx(alpha).GetNeighbors():
            beta = beta_atom.GetIdx()
            ab_bond = mol.GetBondBetweenAtoms(alpha, beta)
            # Only β-scission across single bonds
            if ab_bond.GetBondTypeAsDouble() != 1.0:
                continue

            for gamma_atom in mol.GetAtomWithIdx(beta).GetNeighbors():
                gamma = gamma_atom.GetIdx()
                if gamma == alpha:
                    continue

                rw = Chem.RWMol(mol)
                # Radical at α forms the new π bond (loses one radical e⁻)
                rw.GetAtomWithIdx(alpha).SetNumRadicalElectrons(alpha_rad - 1)
                # alpha-beta bond: single becomes double
                rw.RemoveBond(alpha, beta)
                rw.AddBond(alpha, beta, Chem.BondType.DOUBLE)
                # Break β–γ; γ gets the radical
                rw.RemoveBond(beta, gamma)
                g_rad = rw.GetAtomWithIdx(gamma).GetNumRadicalElectrons()
                rw.GetAtomWithIdx(gamma).SetNumRadicalElectrons(g_rad + 1)

                try:
                    Chem.SanitizeMol(rw)
                except Exception:
                    continue

                for frag in Chem.GetMolFrags(rw, asMols=True, sanitizeFrags=False):
                    # Skip pure-hydrogen fragments (ejected H•) — they carry no
                    # structural information and trigger RDKit RemoveHs warnings.
                    if frag.GetNumHeavyAtoms() == 0:
                        continue
                    try:
                        Chem.SanitizeMol(frag)
                        # Remove explicit H before generating SMILES so products
                        # are in the same canonical form as the rest of the library.
                        fsmi = Chem.MolToSmiles(Chem.RemoveHs(frag))
                    except Exception:
                        continue
                    # H-cap any radical fragment
                    if any(a.GetNumRadicalElectrons() > 0 for a in frag.GetAtoms()):
                        fsmi = hcap_smiles(fsmi) or fsmi
                    if fsmi and fsmi != original_hcap:
                        products.add(fsmi)

    return list(products)


def h_migration_products(smi: str | None, max_distance: int = 5) -> list[str]:
    """
    Enumerate H-shift products of a radical SMILES.

    For each radical centre α and each H atom within *max_distance* bonds:
      - Transfer the H from its donor atom to α (radical migrates to donor)
      - H-cap the resulting radical to give a closed-shell isomer

    The default *max_distance=5* covers 1,2- through 1,5-H shifts.
    1,5-H shifts are especially important in EI fragmentation (McLafferty
    rearrangement); the older default of 3 missed these.

    Returns unique canonical SMILES that differ from the plain H-capped
    SMILES (i.e. only structurally distinct rearrangement products).
    """
    if smi is None:
        return []
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    mol_h = Chem.AddHs(mol)

    radicals = [(a.GetIdx(), a.GetNumRadicalElectrons())
                for a in mol_h.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if not radicals:
        return []

    original_hcap = hcap_smiles(smi)
    products: set[str] = set()

    for alpha, alpha_rad in radicals:
        for h_idx in range(mol_h.GetNumAtoms()):
            if mol_h.GetAtomWithIdx(h_idx).GetAtomicNum() != 1:
                continue
            path = Chem.GetShortestPath(mol_h, alpha, h_idx)
            dist = len(path) - 1
            if dist < 2 or dist > max_distance:
                continue

            # Donor: the non-H atom bonded to this H
            donors = [n.GetIdx() for n in mol_h.GetAtomWithIdx(h_idx).GetNeighbors()
                      if n.GetAtomicNum() != 1]
            if not donors:
                continue
            donor = donors[0]
            if donor == alpha:
                continue

            rw = Chem.RWMol(mol_h)
            # Transfer H: break donor–H, form α–H
            rw.RemoveBond(h_idx, donor)
            rw.AddBond(alpha, h_idx, Chem.BondType.SINGLE)
            # Radical moves from α to donor
            rw.GetAtomWithIdx(alpha).SetNumRadicalElectrons(alpha_rad - 1)
            d_rad = rw.GetAtomWithIdx(donor).GetNumRadicalElectrons()
            rw.GetAtomWithIdx(donor).SetNumRadicalElectrons(d_rad + 1)

            try:
                Chem.SanitizeMol(rw)
                canon = Chem.MolToSmiles(Chem.RemoveHs(rw))
                hcapped = hcap_smiles(canon)
                if hcapped and hcapped != original_hcap:
                    products.add(hcapped)
            except Exception:
                pass

    return list(products)


# Heteroatoms that weaken α-C-H bonds under EI conditions
_ALPHA_HETEROATOMS = {7, 8, 9, 16, 17, 35, 53}   # N, O, F, S, Cl, Br, I


def alpha_dehydrogenation_products(smi: str | None) -> list[str]:
    """
    Enumerate products of α-C-H loss next to heteroatoms.

    For each carbon bonded to a heteroatom (N, O, S, halogen), remove one
    hydrogen to create a radical, then apply β-scission to the radical.
    This captures pathways like:

        CH₃CH₂I  →  CH₃-ĊH-I + H•  →  CH₂=CHI + H•

    Returns unique canonical SMILES of the β-scission products (closed-shell).
    The intermediate radical and the lost H• are not returned.
    """
    if smi is None:
        return []
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []

    # Only apply to closed-shell molecules (no existing radicals)
    if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
        return []

    mol_h = Chem.AddHs(mol)
    products: set[str] = set()

    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        # Check if this C is bonded to a heteroatom
        has_hetero = any(
            n.GetAtomicNum() in _ALPHA_HETEROATOMS
            for n in atom.GetNeighbors()
        )
        if not has_hetero:
            continue

        # Find H atoms bonded to this carbon
        h_neighbors = [n.GetIdx() for n in atom.GetNeighbors()
                       if n.GetAtomicNum() == 1]
        if not h_neighbors:
            continue

        # Remove one H to create a radical on this carbon.
        # Tag the α-carbon so we can find it after RemoveAtom shifts indices.
        c_idx = atom.GetIdx()
        h_idx = h_neighbors[0]
        rw = Chem.RWMol(mol_h)
        rw.GetAtomWithIdx(c_idx).SetAtomMapNum(99)
        rw.RemoveBond(c_idx, h_idx)
        rw.RemoveAtom(h_idx)

        # Set radical on the tagged carbon (implicit H must be blocked)
        for a in rw.GetAtoms():
            if a.GetAtomMapNum() == 99:
                a.SetNoImplicit(True)
                a.SetNumRadicalElectrons(1)
                a.SetAtomMapNum(0)
                break

        try:
            Chem.SanitizeMol(rw)
            radical_smi = Chem.MolToSmiles(Chem.RemoveHs(rw))
        except Exception:
            continue

        # Apply β-scission to the radical intermediate
        for bsmi in beta_scission_products(radical_smi):
            if bsmi:
                products.add(bsmi)

    return list(products)
