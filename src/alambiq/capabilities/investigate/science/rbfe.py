"""Checks on a relative free-energy request, before any sampling.

The ligand checks are exact: element, isotope, aromaticity and charge must be kept
by every mapped atom, stereocenters and double bonds must keep their spatial
orientation, and the mapped core must be connected and complete in every ring it
touches. A mapping that violates any of them would make the transformation a
different one from the one the result is labeled with.
"""

from __future__ import annotations

import math


def check_settings(
    *,
    replicas: int,
    lambda_windows: int,
    threads: int,
    timeout_seconds: int,
    temperature_kelvin: float,
    pressure_bar: float,
    ionic_strength_molar: float,
    solvent_padding_nm: float,
    equilibration_ps: float,
    production_ps: float,
    time_per_iteration_ps: float,
    checkpoint_interval_ps: float,
    positions_interval_ps: float,
    atom_mapping: list[list[int]],
) -> None:
    """Refuse settings OpenFE cannot run or that would make the timing inexact."""
    for name, value, low, high in (
        ("replicas", replicas, 1, 10),
        ("lambda_windows", lambda_windows, 5, 41),
        ("threads", threads, 1, 128),
        ("timeout_seconds", timeout_seconds, 60, 10**9),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; {name} requires an integer in [{low}, {high}]"
            )
    for name, value, low, high in (
        ("temperature_kelvin", temperature_kelvin, 250, 400),
        ("pressure_bar", pressure_bar, 0.1, 100),
        ("ionic_strength_molar", ionic_strength_molar, 0, 1),
        ("solvent_padding_nm", solvent_padding_nm, 1.5, 3),
        ("equilibration_ps", equilibration_ps, 0.02, 100000),
        ("production_ps", production_ps, 0.04, 200000),
        ("time_per_iteration_ps", time_per_iteration_ps, 0.02, 10),
        ("checkpoint_interval_ps", checkpoint_interval_ps, 0.02, 1000),
        ("positions_interval_ps", positions_interval_ps, 0.02, 1000),
    ):
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; {name} requires a finite value in [{low}, {high}]"
            )
    iteration = time_per_iteration_ps
    for value in [
        iteration / 0.002,
        equilibration_ps / iteration,
        production_ps / iteration,
        checkpoint_interval_ps / iteration,
        positions_interval_ps / iteration,
    ]:
        if not math.isclose(value, round(value), rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                "Timing requires whole 2 fs steps and whole replica-exchange iterations"
            )
    if checkpoint_interval_ps > production_ps or positions_interval_ps > production_ps:
        raise ValueError(
            f"checkpoint_interval_ps is {checkpoint_interval_ps:g} ps and "
            f"positions_interval_ps is {positions_interval_ps:g} ps against "
            f"production_ps of {production_ps:g} ps; output intervals must fit within "
            "production sampling"
        )
    if (
        not 1 <= len(atom_mapping) <= 500
        or any(len(pair) != 2 or min(pair) < 0 for pair in atom_mapping)
        or any(
            len({pair[i] for pair in atom_mapping}) != len(atom_mapping) for i in (0, 1)
        )
    ):
        raise ValueError(
            f"atom_mapping has {len(atom_mapping)} pairs; atom mapping requires 1–500 "
            "distinct zero-based integer pairs"
        )


def validate_ligands(a, b, pairs):
    import numpy as np
    from rdkit import Chem

    identities = []
    for mol in (a, b):
        if mol is None or not 1 <= mol.GetNumAtoms() <= 500:
            raise ValueError(
                "Each ligand requires one molecule with 1–500 explicit atoms"
            )
        if Chem.GetFormalCharge(mol) != 0 or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(
                f"A ligand has formal charge {Chem.GetFormalCharge(mol)} and "
                f"{len(Chem.GetMolFrags(mol))} fragment(s); this profile uses connected "
                "neutral ligand species"
            )
        if (
            mol.GetNumConformers() != 1
            or not mol.GetConformer().Is3D()
            or not np.isfinite(mol.GetConformer().GetPositions()).all()
        ):
            raise ValueError("Each ligand requires one finite 3D conformer")
        if any(
            atom.GetNumImplicitHs() or atom.GetNumExplicitHs()
            for atom in mol.GetAtoms()
        ):
            raise ValueError("Ligand hydrogens must be explicit coordinate atoms")
        if any(
            str(info.specified) == "Unspecified"
            for info in Chem.FindPotentialStereo(mol)
        ):
            raise ValueError("Ligand stereochemistry requires explicit specification")
        if not mol.HasProp("atom.dprop.PartialCharge"):
            raise ValueError(
                "A ligand has no atom.dprop.PartialCharge property; supply precomputed "
                "partial charges in atom.dprop.PartialCharge"
            )
        try:
            charges = np.array(
                mol.GetProp("atom.dprop.PartialCharge").split(), dtype=float
            )
        except ValueError as exc:
            raise ValueError(
                "atom.dprop.PartialCharge holds a non-numerical entry; partial charges "
                "must be numerical"
            ) from exc
        if (
            charges.shape != (mol.GetNumAtoms(),)
            or not np.isfinite(charges).all()
            or abs(charges.sum()) > 1e-6
            or not np.any(abs(charges) > 1e-8)
        ):
            raise ValueError(
                f"atom.dprop.PartialCharge holds {charges.size} charges for "
                f"{mol.GetNumAtoms()} atoms with sum {charges.sum():g}; finite per-atom "
                "partial charges must sum to the neutral formal charge and not all be zero"
            )
        identities.append(Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True))
    if any(i >= a.GetNumAtoms() or j >= b.GetNumAtoms() for i, j in pairs):
        raise ValueError(
            "A mapped atom index lies outside its exact ligand; each index must be below "
            f"the atom count ({a.GetNumAtoms()} and {b.GetNumAtoms()})"
        )
    mapping = dict(pairs)
    if len(mapping) != len(pairs) or len(set(mapping.values())) != len(pairs):
        raise ValueError(
            "Atom mapping repeats an atom index; atom mapping must be one-to-one"
        )
    for i, j in pairs:
        aa, bb = a.GetAtomWithIdx(i), b.GetAtomWithIdx(j)
        if (
            aa.GetAtomicNum(),
            aa.GetIsotope(),
            aa.GetIsAromatic(),
            aa.GetFormalCharge(),
        ) != (
            bb.GetAtomicNum(),
            bb.GetIsotope(),
            bb.GetIsAromatic(),
            bb.GetFormalCharge(),
        ):
            raise ValueError(
                "Mapped core atoms must preserve element, isotope, aromaticity and formal charge"
            )
        tetrahedral = {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
        if aa.GetChiralTag() in tetrahedral or bb.GetChiralTag() in tetrahedral:
            neighbors_a = [n.GetIdx() for n in aa.GetNeighbors()]
            neighbors_b = [n.GetIdx() for n in bb.GetNeighbors()]
            if (
                aa.GetChiralTag() not in tetrahedral
                or bb.GetChiralTag() not in tetrahedral
                or len(neighbors_a) != 4
                or len(neighbors_b) != 4
                or any(n not in mapping for n in neighbors_a)
                or {mapping[n] for n in neighbors_a} != set(neighbors_b)
            ):
                raise ValueError(
                    "Mapped stereocenters require the complete mapped tetrahedral neighborhood"
                )
            order = [neighbors_b.index(mapping[n]) for n in neighbors_a]
            odd = (
                sum(order[x] > order[y] for x in range(4) for y in range(x + 1, 4)) % 2
            )
            if (aa.GetChiralTag() != bb.GetChiralTag()) != bool(odd):
                raise ValueError(
                    "Mapped stereocenters must preserve spatial orientation"
                )
    for i, j in pairs:
        for k, mapped_k in pairs:
            ba, bb = a.GetBondBetweenAtoms(i, k), b.GetBondBetweenAtoms(j, mapped_k)
            if (ba is None) != (bb is None) or (
                ba is not None and ba.GetBondType() != bb.GetBondType()
            ):
                raise ValueError(
                    "Mapped core connectivity and bond orders must be preserved"
                )
            orientation = {
                Chem.BondStereo.STEREOE: -1,
                Chem.BondStereo.STEREOTRANS: -1,
                Chem.BondStereo.STEREOZ: 1,
                Chem.BondStereo.STEREOCIS: 1,
            }
            if ba is not None and (
                ba.GetStereo() in orientation or bb.GetStereo() in orientation
            ):
                if (
                    ba.GetStereo() not in orientation
                    or bb.GetStereo() not in orientation
                ):
                    raise ValueError(
                        "Mapped alkene stereochemistry must be specified at both endpoints"
                    )
                controls_a, controls_b = (
                    list(ba.GetStereoAtoms()),
                    list(bb.GetStereoAtoms()),
                )
                if (
                    len(controls_a) != 2
                    or len(controls_b) != 2
                    or any(n not in mapping for n in controls_a)
                ):
                    raise ValueError(
                        "Mapped alkene stereochemistry requires mapped controlling neighbors"
                    )
                if mapping[ba.GetBeginAtomIdx()] != bb.GetBeginAtomIdx():
                    controls_b.reverse()
                flips = sum(
                    mapping[n] != m for n, m in zip(controls_a, controls_b, strict=True)
                )
                if (
                    orientation[ba.GetStereo()] * (-1) ** flips
                    != orientation[bb.GetStereo()]
                ):
                    raise ValueError(
                        "Mapped alkene stereochemistry must preserve spatial orientation"
                    )
    for mol, mapped in [(a, set(mapping)), (b, set(mapping.values()))]:
        for ring in mol.GetRingInfo().AtomRings():
            if mapped.intersection(ring) and not set(ring) <= mapped:
                raise ValueError("Map each participating ring completely")
        reached, pending = set(), [next(iter(mapped))]
        while pending:
            atom = pending.pop()
            if atom in reached:
                continue
            reached.add(atom)
            pending.extend(
                n.GetIdx()
                for n in mol.GetAtomWithIdx(atom).GetNeighbors()
                if n.GetIdx() in mapped - reached
            )
        if reached != mapped:
            raise ValueError("Mapped core must form one connected fragment")
    return {"smiles": identities, "formal_charges": [0, 0], "mapped_atoms": len(pairs)}
