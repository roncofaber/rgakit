"""
recombination.py
----------------
Utilities for preparing, relaxing, and screening radical/ion recombination
candidates using an MLIP calculator (e.g. FAIRChemCalculator from fairchem).
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from ase import Atoms

from .compound import Compound, atoms_to_graph

logger = logging.getLogger(__name__)


def recombination_candidate(
    frag_a: Compound,
    frag_b: Compound,
    gap:  float = 1.5,
    seed: int | None = None,
) -> Atoms:
    """
    Prepare an initial-guess Atoms object for a recombination event.

    Orients the two fragments so that their reactive centres face each other
    while their bulk geometry points away — analogous to polymer docking in
    mdinterface.  The "outward vector" of each fragment is the unit vector from
    the body COM (non-boundary atoms) toward the reactive centre.

    Placement steps:
      1. Compute outward vector *dir_a* for fragment A.
      2. Rotate B so its outward vector *dir_b* anti-aligns with *dir_a*
         (B's body points away from A, reactive centre points toward A).
      3. Apply a random dihedral rotation of B around *dir_a* to sample
         different bonding geometries.
      4. Translate B so its reactive centre sits *gap* Å from A's along *dir_a*.

    Parameters
    ----------
    frag_a, frag_b : Fragment objects to combine
    gap            : distance between reactive centres in Angstroms (default 1.5)
    seed           : random seed for reproducibility (default: random)
    """
    rng = np.random.default_rng(seed)

    atoms_a = frag_a.atoms.copy()
    atoms_b = frag_b.atoms.copy()

    def _reactive_centre(frag, atoms):
        """Position of the primary reactive centre (first boundary atom or COM)."""
        if frag.boundary_idx:
            return atoms.get_positions()[frag.boundary_idx[0]].copy()
        return atoms.get_center_of_mass()

    def _body_com(frag, atoms):
        """COM of non-boundary (body) atoms; falls back to full COM."""
        if frag.boundary_idx:
            body = [i for i in range(len(atoms)) if i not in set(frag.boundary_idx)]
            if body:
                return atoms.get_positions()[body].mean(axis=0)
        return atoms.get_center_of_mass()

    def _outward_vec(frag, atoms):
        """
        Unit vector from body COM toward reactive centre.
        Returns a random unit vector for degenerate (single-atom) cases.
        """
        rc  = _reactive_centre(frag, atoms)
        com = _body_com(frag, atoms)
        v   = rc - com
        n   = np.linalg.norm(v)
        if n < 1e-6:
            v = rng.standard_normal(3)
            n = np.linalg.norm(v)
        return v / n

    def _rot_from_to(v_from, v_to):
        """
        Rotation matrix that maps unit vector *v_from* onto *v_to*.
        Uses Rodrigues' formula; handles anti-parallel case with a 180° flip.
        """
        ax = np.cross(v_from, v_to)
        c  = float(np.dot(v_from, v_to))
        s  = np.linalg.norm(ax)
        if s < 1e-9:
            if c > 0:
                return np.eye(3)
            # Anti-parallel: 180° around any perpendicular axis
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(perp, v_from)) > 0.9:
                perp = np.array([0.0, 1.0, 0.0])
            perp -= np.dot(perp, v_from) * v_from
            perp /= np.linalg.norm(perp)
            K = np.array([[0, -perp[2], perp[1]],
                          [perp[2], 0, -perp[0]],
                          [-perp[1], perp[0], 0]])
            return np.eye(3) + 2 * (K @ K)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + K + K @ K * ((1 - c) / (s * s))

    dir_a = _outward_vec(frag_a, atoms_a)
    dir_b = _outward_vec(frag_b, atoms_b)
    rc_a  = _reactive_centre(frag_a, atoms_a)
    rc_b  = _reactive_centre(frag_b, atoms_b)

    # Rotate B so dir_b aligns to -dir_a (bodies point away from each other)
    R_align = _rot_from_to(dir_b, -dir_a)
    atoms_b.set_positions((R_align @ (atoms_b.get_positions() - rc_b).T).T + rc_b)

    # Random dihedral rotation around dir_a to sample bonding geometry
    angle = rng.uniform(0.0, 2 * np.pi)
    K     = np.array([[       0, -dir_a[2],  dir_a[1]],
                      [ dir_a[2],        0, -dir_a[0]],
                      [-dir_a[1],  dir_a[0],        0]])
    R_dih = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    atoms_b.set_positions((R_dih @ (atoms_b.get_positions() - rc_b).T).T + rc_b)

    # Translate B so rc_b lands *gap* Å from rc_a along dir_a
    atoms_b.translate(rc_a + gap * dir_a - rc_b)

    n_a      = len(atoms_a)
    combined = atoms_a + atoms_b
    combined.info["charge"] = frag_a.charge + frag_b.charge
    # Use minimum multiplicity (singlet preference): two radicals pairing gives spin=0
    combined.info["spin"]   = abs(frag_a.spin - frag_b.spin)

    # Store reactive centre indices in the combined system for Hookean spring use.
    # Falls back to the atom nearest the COM for isolated ions (no boundary_idx).
    def _reactive_idx(frag, atoms):
        if frag.boundary_idx:
            return frag.boundary_idx[0]
        pos = atoms.get_positions()
        com = atoms.get_center_of_mass()
        return int(np.argmin(np.linalg.norm(pos - com, axis=1)))

    combined.info["reactive_pair"] = (
        _reactive_idx(frag_a, atoms_a),
        n_a + _reactive_idx(frag_b, atoms_b),
    )

    return combined


def relax_candidate(
    candidate:    Atoms,
    calc,
    fmax:         float      = 5e-2,
    steps:        int        = 100,
    log_dir                  = None,
    name:         str        = "recombination",
    spring_k:     float      = 5.0,
    spring_steps: int        = 50,
) -> Atoms:
    """
    Relax a recombination candidate Atoms object using an ASE calculator.

    Two-phase relaxation:

    **Phase 1 — Hookean spring pre-relaxation** (if reactive_pair is set):
    A gentle Hookean spring between the two reactive centres (bond length =
    sum of covalent radii, k = *spring_k* eV/Å²) is applied for
    *spring_steps* steps.  This prevents atoms from flying apart due to
    the large MLIP forces that arise from the close / overlapping initial
    geometry produced by recombination_candidate().

    **Phase 2 — unconstrained FIRE2 relaxation**:
    The spring is removed and a full FIRE2 run is performed to *fmax*
    convergence within *steps* steps.

    Parameters
    ----------
    candidate    : Atoms object returned by recombination_candidate()
    calc         : ASE-compatible calculator
    fmax         : FIRE2 force convergence threshold (eV/Ang)
    steps        : maximum FIRE2 steps for unconstrained phase
    log_dir      : directory for output files; created if absent
    name         : base filename (without extension) for .traj and .log
    spring_k     : Hookean spring constant (eV/Å²); default 5.0
    spring_steps : number of constrained pre-relaxation steps; default 50
    """
    if len(candidate) < 2:
        logger.warning("Skipping relaxation of '%s': fewer than 2 atoms.", name)
        return candidate

    from ase.optimize import FIRE2
    from ase.constraints import Hookean
    from ase.data import covalent_radii, atomic_numbers as _anum
    from pathlib import Path

    trajectory = logfile = None
    if log_dir is not None:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        trajectory = str(d / f"{name}.traj")
        logfile    = str(d / f"{name}.log")

    candidate.calc = calc

    # ------------------------------------------------------------------
    # Phase 1: Hookean spring pre-relaxation
    # ------------------------------------------------------------------
    pair = candidate.info.get("reactive_pair")
    if pair is not None:
        idx_a, idx_b = pair
        syms = candidate.get_chemical_symbols()
        rt   = (covalent_radii[_anum[syms[idx_a]]] +
                covalent_radii[_anum[syms[idx_b]]])
        spring = Hookean(idx_a, idx_b, spring_k, rt=rt)
        candidate.set_constraint([spring])
        logger.debug(
            "  %s: Hookean spring %s-%s, rt=%.2f Å, k=%.1f eV/Å²",
            name, syms[idx_a], syms[idx_b], rt, spring_k,
        )
        opt = FIRE2(candidate, trajectory=trajectory, logfile=logfile)
        try:
            opt.run(fmax=fmax * 10, steps=spring_steps)
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "Spring pre-relaxation failed for '%s' (%s: %s) — "
                "proceeding to unconstrained phase.",
                name, type(exc).__name__, exc,
            )
        candidate.set_constraint()   # remove spring

    # ------------------------------------------------------------------
    # Phase 2: unconstrained FIRE2 relaxation (appended to same trajectory)
    # ------------------------------------------------------------------
    opt = FIRE2(candidate, trajectory=trajectory,
                append_trajectory=True, logfile=logfile)
    try:
        candidate.info["converged"] = opt.run(fmax=fmax, steps=steps)
    except (ValueError, RuntimeError) as exc:
        logger.warning(
            "Skipping '%s': MLIP could not evaluate forces (%s: %s). "
            "Atoms may be too far apart or outside the model's cutoff.",
            name, type(exc).__name__, exc,
        )
        candidate.info["converged"] = False
    return candidate


def _is_single_molecule(atoms: Atoms, mult: float = 1.2) -> bool:
    """
    Return True if all atoms form exactly one connected component.

    Builds a NetworkX bond graph via `atoms_to_graph` and checks that
    `nx.connected_components` yields a single component. A recombination
    product passes only when the two input fragments have actually bonded
    into one molecule, not when they remain as two dangling pieces.
    """
    if len(atoms) < 2:
        return True
    G = atoms_to_graph(atoms, mult=mult)
    return nx.is_connected(G)


def relax_all_recombinations(
    fragments:  list,
    calc,
    gap:        float       = 1.5,
    fmax:       float       = 5e-2,
    steps:      int         = 100,
    log_dir                 = None,
    bond_mult:  float       = 1.2,
) -> list[Atoms]:
    """
    Enumerate all pairwise recombination candidates, relax each one, and
    return only those that converged to a stably bonded structure.

    A result is kept when the relaxed structure forms a single connected
    molecule (all atoms reachable via bonds using ASE natural covalent
    cutoffs * *bond_mult*).  Pairs that remain as two dangling fragments
    after relaxation are discarded.

    Parameters
    ----------
    fragments : list of Compound objects (e.g. mol.raw_fragments)
    calc      : ASE-compatible calculator
    gap       : reactive-centre separation for initial placement (Ang)
    fmax      : FIRE force convergence threshold (eV/Ang)
    steps     : maximum FIRE steps per candidate
    log_dir   : directory for trajectory/log files; created if absent
    bond_mult : covalent-radius multiplier for bond detection (default 1.2)

    Returns
    -------
    list of relaxed Atoms objects, centered at origin, with
    atoms.info["converged"] and atoms.info["fragments"] = (i, j) set.
    """
    from itertools import combinations

    stable = []
    pairs  = list(combinations(range(len(fragments)), 2))
    logger.info("Testing %d recombination pairs.", len(pairs))

    for i, j in pairs:
        fa, fb = fragments[i], fragments[j]
        name   = f"{i:03d}_{fa.formula}_x_{j:03d}_{fb.formula}"

        candidate = recombination_candidate(fa, fb, gap=gap, seed=i * 10000 + j)
        relax_candidate(candidate, calc, fmax=fmax, steps=steps,
                        log_dir=log_dir, name=name)

        if not _is_single_molecule(candidate, mult=bond_mult):
            logger.debug("  %s: two fragments remain after relaxation, skipped.", name)
            continue

        logger.info("  %s: bonded into a single molecule.", name)
        candidate.info["fragments"] = (i, j)
        candidate.center()
        stable.append(candidate)

    logger.info("Found %d stable recombination products.", len(stable))
    return stable
