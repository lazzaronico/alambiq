"""Explicit terminal templates and source-positioned molecular coordinates.

Coordinate modeling retains deposited versus modeled atoms. Force-field
parameterization, bonded relaxation and physical sampling are separate operations.
"""

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from alambiq.capabilities.investigate.science.predicted_complex import (
    EXPECTED_HEAVY_ATOMS,
)


def _keys(value, names, where="the recipe"):
    if not isinstance(value, dict) or set(value) != set(names):
        found = sorted(value) if isinstance(value, dict) else "no mapping"
        raise ValueError(
            f"{where} has the fields {found}; exactly the fields "
            f"{sorted(names)} are required"
        )


def _count(value):
    return len(value) if isinstance(value, list) else 0


def _token(value, where):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
        raise ValueError(
            f"{where} is {value!r}; 1 to 40 letters, digits, underscores or hyphens "
            "are required"
        )


def _indices(values, maximum, where):
    if (
        not isinstance(values, list)
        or any(type(i) is not int or not 1 <= i <= maximum for i in values)
        or values != sorted(set(values))
    ):
        raise ValueError(
            f"{where} is {values!r}; sorted, distinct one-based indices from 1 to "
            f"{maximum} are required"
        )


def coordinate_recipe(supplied):
    data = copy.deepcopy(supplied)
    _keys(data, {"schema_version", "target", "donor", "terminal_model", "substrate"})
    if data["schema_version"] != "0.1":
        raise ValueError(
            f"The recipe has schema_version {data['schema_version']!r}; 0.1 is required"
        )
    target, donor, model, ligand = (
        data[k] for k in ("target", "donor", "terminal_model", "substrate")
    )
    _keys(target, {"assembly_id", "model_number", "chains"}, "target")
    _keys(donor, {"model_number", "chains"}, "donor")
    _token(target["assembly_id"], "target.assembly_id")
    for where, obj in (("target", target), ("donor", donor), ("substrate", ligand)):
        if type(obj.get("model_number")) is not int or obj["model_number"] < 1:
            raise ValueError(
                f"{where}.model_number is {obj.get('model_number')!r}; a positive "
                "integer deposited model number is required"
            )
    if not isinstance(target["chains"], list) or not 1 <= len(target["chains"]) <= 16:
        raise ValueError(
            f"target.chains has {_count(target['chains'])} entries; 1 to 16 are required"
        )
    for row in target["chains"]:
        _keys(
            row,
            {"source_chain", "id", "sequence", "missing_residues", "histidines"},
            "A target chain",
        )
        _token(row["source_chain"], "target.chains source_chain")
        if not isinstance(row["id"], str) or not re.fullmatch(
            r"[A-Za-z0-9]", row["id"]
        ):
            raise ValueError(
                f"The id {row['id']!r} of target chain {row['source_chain']} is not a "
                "single letter or digit; one alphanumeric character is required"
            )
        if not isinstance(row["sequence"], str) or not re.fullmatch(
            r"[ACDEFGHIKLMNPQRSTVWY]{4,2000}", row["sequence"]
        ):
            raise ValueError(
                f"The sequence of target chain {row['source_chain']} is not 4 to 2,000 "
                "letters from the 20 canonical amino acids; the complete canonical "
                "sequence is required"
            )
        _indices(
            row["missing_residues"],
            len(row["sequence"]),
            f"missing_residues of target chain {row['source_chain']}",
        )
        if not isinstance(row["histidines"], dict) or any(
            not isinstance(i, str)
            or not re.fullmatch(r"[1-9][0-9]*", i)
            or int(i) > len(row["sequence"])
            or row["sequence"][int(i) - 1] != "H"
            or variant not in {"HID", "HIE", "HIP"}
            for i, variant in row["histidines"].items()
        ):
            raise ValueError(
                f"The histidines of target chain {row['source_chain']} name a position "
                "that is not a histidine of the sequence, or a state other than HID, "
                "HIE or HIP; only histidine positions with these states are accepted"
            )
    if (
        sum(len(r["sequence"]) for r in target["chains"]) > 4000
        or sum(len(r["missing_residues"]) for r in target["chains"]) > 64
    ):
        raise ValueError(
            f"The target chains have {sum(len(r['sequence']) for r in target['chains'])} "
            f"residues and {sum(len(r['missing_residues']) for r in target['chains'])} "
            "missing residues; at most 4,000 residues and 64 modeled terminal residues "
            "are accepted"
        )
    for key in ("source_chain", "id"):
        if len({r[key] for r in target["chains"]}) != len(target["chains"]):
            raise ValueError(
                f"target.chains repeats a value of {key}; the values of source_chain "
                "and of id must each be distinct"
            )
    if len({r["id"].lower() for r in target["chains"]}) != len(target["chains"]):
        raise ValueError(
            "Two target chain ids differ only in case; the ids must be distinct when "
            "compared in lowercase, because they name files"
        )
    if not isinstance(donor["chains"], list) or not 1 <= len(donor["chains"]) <= 16:
        raise ValueError(
            f"donor.chains has {_count(donor['chains'])} entries; 1 to 16 are required"
        )
    for name in donor["chains"]:
        _token(name, "A name in donor.chains")
    if len(set(donor["chains"])) != len(donor["chains"]):
        raise ValueError(
            "donor.chains repeats a chain name; distinct names are required"
        )
    _keys(
        model,
        {
            "anchor_residues",
            "max_fit_rmsd_angstrom",
            "junction_min_angstrom",
            "junction_max_angstrom",
            "clash_distance_angstrom",
            "water_exclusion_angstrom",
        },
        "terminal_model",
    )
    if (
        type(model["anchor_residues"]) is not int
        or not 1 <= model["anchor_residues"] <= 10
    ):
        raise ValueError(
            f"terminal_model.anchor_residues is {model['anchor_residues']!r}; an "
            "integer from 1 to 10 adjacent backbone anchor residues is required"
        )
    for key, value in model.items():
        if key != "anchor_residues" and (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 < value <= 20
        ):
            raise ValueError(
                f"terminal_model.{key} is {value!r}; positive finite angstrom values "
                "up to 20 are required"
            )
    if model["junction_min_angstrom"] > model["junction_max_angstrom"]:
        raise ValueError(
            f"terminal_model.junction_min_angstrom ({model['junction_min_angstrom']}) "
            f"exceeds junction_max_angstrom ({model['junction_max_angstrom']}); the "
            "bounds must be ordered"
        )
    _keys(
        ligand,
        {
            "model_number",
            "chain",
            "residue_number",
            "insertion_code",
            "component",
            "altloc",
            "source_form",
            "formal_charge",
            "source_atom_names",
            "anchor_component",
            "anchor_atom_names",
            "hydrogen_max_iterations",
        },
        "substrate",
    )
    for key in ("chain", "component", "anchor_component"):
        _token(ligand[key], f"substrate.{key}")
    if (
        type(ligand["residue_number"]) is not int
        or type(ligand["formal_charge"]) is not int
    ):
        raise ValueError(
            "substrate.residue_number or substrate.formal_charge is not an integer; "
            "integers are required"
        )
    for key in ("altloc", "insertion_code"):
        if not isinstance(ligand[key], str) or not re.fullmatch(
            r"[A-Za-z0-9]?", ligand[key]
        ):
            raise ValueError(
                f"substrate.{key} is {ligand[key]!r}; a blank or a single letter or "
                "digit is required"
            )
    if (
        not isinstance(ligand["source_form"], str)
        or not 1 <= len(ligand["source_form"]) <= 500
        or any(ord(c) < 32 for c in ligand["source_form"])
    ):
        raise ValueError(
            "substrate.source_form is empty, longer than 500 characters or contains a "
            "control character; a single line of 1 to 500 characters is required"
        )
    for key in ("source_atom_names", "anchor_atom_names"):
        values = ligand[key]
        if (
            not isinstance(values, list)
            or not 3 <= len(values) <= 150
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"substrate.{key} has {_count(values)} "
                "entries or repeats a name; 3 to 150 distinct heavy-atom names in "
                "molecular input order are required"
            )
        for value in values:
            _token(value, f"A name in substrate.{key}")
    if len(ligand["source_atom_names"]) != len(ligand["anchor_atom_names"]):
        raise ValueError(
            f"substrate.source_atom_names has {len(ligand['source_atom_names'])} entries "
            f"and anchor_atom_names has {len(ligand['anchor_atom_names'])}; the lists "
            "must have equal length because both map the same molecule"
        )
    if (
        type(ligand["hydrogen_max_iterations"]) is not int
        or not 1 <= ligand["hydrogen_max_iterations"] <= 10000
    ):
        raise ValueError(
            f"substrate.hydrogen_max_iterations is {ligand['hydrogen_max_iterations']!r}; "
            "an integer from 1 to 10,000 is required"
        )
    return data


def dependencies():
    """The libraries this preparation uses, with their versions recorded in the result."""
    import gemmi
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign

    return gemmi, np, Chem, AllChem, rdMolAlign


def versions():
    import gemmi
    import numpy as np
    from rdkit import rdBase

    return {
        "gemmi": gemmi.__version__,
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
    }


def _model(structure, number):
    matches = [m for m in structure if m.num == number]
    if len(matches) != 1:
        raise ValueError(
            f"model_number {number} matches {len(matches)} deposited models; "
            "exactly 1 is required"
        )
    return matches[0]


def _structure(block):
    gemmi, _, _, _, _ = dependencies()
    structure = gemmi.make_structure_from_block(block)
    # Deposited atom tables can separate polymer, ligand and water parts of the
    # same author chain. Match Gemmi's ordinary read_structure convention.
    structure.merge_chain_parts()
    return structure


def _heavy(residue):
    """Keep one coherent highest-occupancy alternative; ties use label order."""
    clone = residue.clone()
    clone.remove_hydrogens()
    labels = sorted({a.altloc for a in clone if a.altloc != "\0"})
    chosen = (
        max(labels, key=lambda s: sum(a.occ for a in clone if a.altloc == s))
        if labels
        else "\0"
    )
    for i in reversed(range(len(clone))):
        if clone[i].altloc not in {"\0", chosen}:
            del clone[i]
    for atom in clone:
        atom.altloc = "\0"
        if not all(math.isfinite(v) for v in atom.pos.tolist()):
            raise ValueError(
                f"Residue {residue.name} {residue.seqid.num} has non-finite coordinates; "
                "finite coordinates are required"
            )
    if not clone or len({a.name for a in clone}) != len(clone):
        raise ValueError(
            f"Residue {residue.name} {residue.seqid.num} has no heavy atoms or repeats a "
            "heavy-atom name; distinct heavy-atom names are required"
        )
    return clone, chosen.replace("\0", "")


def _atom(residue, name):
    matches = [a for a in residue if a.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"Residue {residue.name} {residue.seqid.num} has {len(matches)} atoms named "
            f"{name}; exactly 1 is required"
        )
    return matches[0]


def _complete_heavy(residue, terminal):
    expected = EXPECTED_HEAVY_ATOMS.get(residue.name)
    names = {a.name for a in residue}
    if expected is None or names - ({"OXT"} if terminal else set()) != expected:
        raise ValueError(
            f"Residue {residue.name} {residue.seqid.num} is not a canonical amino acid "
            "with exactly its heavy atoms (OXT is allowed only at the C terminus); "
            "complete canonical residues are required"
        )


def prepare_protein(target, donor, recipe):
    gemmi, np, _, _, _ = dependencies()
    assemblies = [
        a for a in target.assemblies if a.name == recipe["target"]["assembly_id"]
    ]
    if len(assemblies) != 1:
        raise ValueError(
            f"assembly_id {recipe['target']['assembly_id']!r} matches {len(assemblies)} "
            "biological assemblies of the target structure; exactly 1 is required"
        )
    assembly = gemmi.make_assembly(
        assemblies[0],
        _model(target, recipe["target"]["model_number"]),
        gemmi.HowToNameCopiedChain.AddNumber,
    )
    selected = []
    for row in recipe["target"]["chains"]:
        matches = [c for c in assembly if c.name == row["source_chain"]]
        if len(matches) != 1:
            raise ValueError(
                f"The assembly has {len(matches)} chains named {row['source_chain']}; "
                "exactly 1 is required"
            )
        selected.append(matches[0])
    donor_model = _model(donor, recipe["donor"]["model_number"])
    donor_chains = []
    for name in recipe["donor"]["chains"]:
        matches = [c for c in donor_model if c.name == name]
        if len(matches) != 1:
            raise ValueError(
                f"The donor model has {len(matches)} chains named {name}; exactly 1 is required"
            )
        donor_chains.append(matches[0])
    output = gemmi.Model(1)
    records, modeled_positions, observed_atoms, alternate_choices = [], [], [], []
    settings = recipe["terminal_model"]
    for source_chain, row in zip(selected, recipe["target"]["chains"], strict=True):
        chain = gemmi.Chain(row["id"])
        observed = {}
        for residue in source_chain.get_polymer():
            if residue.seqid.icode.strip() or residue.seqid.num in observed:
                raise ValueError(
                    f"Chain {row['source_chain']} has residue {residue.seqid.num} with an "
                    "insertion code or a repeated number; distinct residue numbers "
                    "without insertion codes are required"
                )
            observed[residue.seqid.num], label = _heavy(residue)
            alternate_choices.append(
                dict(
                    chain=row["source_chain"],
                    residue=residue.seqid.num,
                    selected_altloc=label,
                )
            )
        length = len(row["sequence"])
        missing = [i for i in range(1, length + 1) if i not in observed]
        if (
            not observed
            or min(observed) < 1
            or sorted(observed) != list(range(min(observed), max(observed) + 1))
            or missing != row["missing_residues"]
            or max(observed) > length
        ):
            raise ValueError(
                f"The unobserved residues of chain {row['source_chain']} differ from the "
                "declared missing_residues or are not terminal groups; the declared "
                "missing residues must be terminal groups around one observed range"
            )
        for i, residue in observed.items():
            _complete_heavy(residue, i == length)
            if (
                gemmi.find_tabulated_residue(residue.name).one_letter_code
                != row["sequence"][i - 1]
            ):
                raise ValueError(
                    f"Residue {i} of chain {row['source_chain']} is {residue.name}, but "
                    f"the declared sequence has {row['sequence'][i - 1]}; the observed and "
                    "declared sequences must be equal"
                )
            observed_atoms.extend(
                dict(
                    chain=row["id"],
                    residue=i,
                    atom=a.name,
                    element=a.element.name,
                    position_angstrom=a.pos.tolist(),
                )
                for a in residue
            )
        templates = []
        for dc in donor_chains:
            residues = {}
            for residue in dc.get_polymer():
                if residue.seqid.icode.strip() or residue.seqid.num in residues:
                    raise ValueError(
                        f"Donor chain {dc.name} has a residue with an insertion code or a "
                        "repeated number; distinct residue numbers without insertion "
                        "codes are required"
                    )
                residues[residue.seqid.num], _ = _heavy(residue)
            if (
                sorted(residues) != list(range(1, length + 1))
                or "".join(
                    gemmi.find_tabulated_residue(residues[i].name).one_letter_code
                    for i in range(1, length + 1)
                )
                != row["sequence"]
            ):
                raise ValueError(
                    f"Donor chain {dc.name} has a sequence or numbering that differs from "
                    f"the target sequence of chain {row['source_chain']}; the exact "
                    f"complete target sequence, numbered 1 to {length}, is required"
                )
            for i, residue in residues.items():
                _complete_heavy(residue, i == length)
            templates.append((dc.name, residues))
        n = settings["anchor_residues"]
        if len(observed) < n:
            raise ValueError(
                f"Chain {row['source_chain']} has {len(observed)} observed residues; at "
                f"least terminal_model.anchor_residues ({n}) are required"
            )
        new = {}
        for terminus, anchors, absent in [
            ("N", sorted(observed)[:n], list(range(1, min(observed)))),
            ("C", sorted(observed)[-n:], list(range(max(observed) + 1, length + 1))),
        ]:
            if not absent:
                continue
            environment = np.array(
                [
                    a.pos.tolist()
                    for c in selected
                    for r in c.get_polymer()
                    if c.name != source_chain.name
                    or (terminus == "N" and r.seqid.num > min(observed))
                    or (terminus == "C" and r.seqid.num < max(observed))
                    for a in _heavy(r)[0]
                ]
            )
            candidates = []
            for donor_name, residues in templates:
                fixed = [
                    _atom(observed[i], name).pos
                    for i in anchors
                    for name in ("N", "CA", "C", "O")
                ]
                mobile = [
                    _atom(residues[i], name).pos
                    for i in anchors
                    for name in ("N", "CA", "C", "O")
                ]
                fit = gemmi.superpose_positions(fixed, mobile)
                generated = {i: residues[i].clone() for i in absent}
                for residue in generated.values():
                    for atom in residue:
                        atom.pos = fit.transform.apply(atom.pos).tolist()
                left = (
                    _atom(generated[max(absent)], "C").pos
                    if terminus == "N"
                    else _atom(observed[max(observed)], "C").pos
                )
                right = (
                    _atom(observed[min(observed)], "N").pos
                    if terminus == "N"
                    else _atom(generated[min(absent)], "N").pos
                )
                distance = left.dist(right)
                xyz = np.array([a.pos.tolist() for r in generated.values() for a in r])
                distances = np.linalg.norm(xyz[:, None] - environment[None, :], axis=-1)
                clashes = int(np.sum(distances < settings["clash_distance_angstrom"]))
                summary = dict(
                    chain=donor_name,
                    rmsd_angstrom=fit.rmsd,
                    junction_angstrom=distance,
                    nonadjacent_contacts=clashes,
                    nearest_nonadjacent_angstrom=float(distances.min()),
                )
                candidates.append((summary, generated))
            eligible = [
                p
                for p in candidates
                if p[0]["rmsd_angstrom"] < settings["max_fit_rmsd_angstrom"]
                and settings["junction_min_angstrom"]
                <= p[0]["junction_angstrom"]
                <= settings["junction_max_angstrom"]
                and p[0]["nonadjacent_contacts"] == 0
            ]
            if not eligible:
                raise ValueError(
                    f"No donor chain meets the terminal-placement criteria for the "
                    f"{terminus} terminus of chain {row['source_chain']}: fit RMSD below "
                    f"{settings['max_fit_rmsd_angstrom']} Å, junction length from "
                    f"{settings['junction_min_angstrom']} to "
                    f"{settings['junction_max_angstrom']} Å, and no modeled atom within "
                    f"{settings['clash_distance_angstrom']} Å of another protein atom"
                )
            summary, generated = min(
                eligible, key=lambda p: (p[0]["rmsd_angstrom"], p[0]["chain"])
            )
            new.update(generated)
            modeled_positions.extend(
                a.pos.tolist() for r in generated.values() for a in r
            )
            records.append(
                dict(
                    target_chain=row["id"],
                    terminus=terminus,
                    anchor_residues=anchors,
                    modeled_residues=absent,
                    donor_chain=summary["chain"],
                    backbone_fit_rmsd_angstrom=summary["rmsd_angstrom"],
                    junction_bond_angstrom=summary["junction_angstrom"],
                    candidates=[p[0] for p in candidates],
                )
            )
        for i in range(1, length + 1):
            residue = observed[i] if i in observed else new[i]
            if residue.name == "HIS":
                if str(i) not in row["histidines"]:
                    raise ValueError(
                        f"Histidine {i} of chain {row['source_chain']} has no entry in "
                        "histidines; a state (HID, HIE or HIP) is required for every "
                        "histidine"
                    )
                residue.name = row["histidines"][str(i)]
            residue.subchain = row["id"]
            residue.entity_type = gemmi.EntityType.Polymer
            chain.add_residue(residue)
        for residue in source_chain:
            if residue.name == "HOH":
                chain.add_residue(_heavy(residue)[0])
        output.add_chain(chain)
    removed = []
    for chain in output:
        for i in reversed(range(len(chain))):
            residue = chain[i]
            if (
                residue.name == "HOH"
                and modeled_positions
                and min(
                    np.linalg.norm(
                        np.array(a.pos.tolist()) - modeled_positions, axis=1
                    ).min()
                    for a in residue
                )
                < settings["water_exclusion_angstrom"]
            ):
                removed.append(dict(chain=chain.name, residue=residue.seqid.num))
                del chain[i]
    result = gemmi.Structure()
    result.name = "explicit_terminal_model"
    result.add_model(output)
    return (
        result,
        selected,
        dict(
            terminal_templates=records,
            observed_atoms=observed_atoms,
            alternate_choices=alternate_choices,
            removed_waters=removed,
        ),
    )


def prepare_ligands(source_block, molecule_path, selected, recipe, output):
    gemmi, np, Chem, AllChem, rdMolAlign = dependencies()
    spec = recipe["substrate"]
    molecule = Chem.MolFromMolFile(
        str(molecule_path), removeHs=False, strictParsing=True
    )
    if (
        molecule is None
        or len(Chem.GetMolFrags(molecule)) != 1
        or molecule.GetNumAtoms() != len(spec["source_atom_names"])
        or any(
            a.GetAtomicNum() <= 1 or a.GetNumRadicalElectrons() or a.GetIsotope()
            for a in molecule.GetAtoms()
        )
    ):
        raise ValueError(
            "The molecule file cannot be parsed, has more than one fragment, has an atom "
            "count different from substrate.source_atom_names, or contains hydrogens, "
            "radicals or isotopes; one explicit connected heavy-atom molecule matching "
            "the source map is required"
        )
    if Chem.GetFormalCharge(molecule) != spec["formal_charge"] or any(
        str(s.specified) == "Unspecified" for s in Chem.FindPotentialStereo(molecule)
    ):
        raise ValueError(
            f"The molecule has formal charge {Chem.GetFormalCharge(molecule)} against "
            f"substrate.formal_charge {spec['formal_charge']}, or has unspecified "
            "stereochemistry; the charge must equal substrate.formal_charge and the "
            "stereochemistry must be complete"
        )
    source_model = _model(_structure(source_block), spec["model_number"])
    residues = [
        r
        for c in source_model
        if c.name == spec["chain"]
        for r in c
        if r.seqid.num == spec["residue_number"]
        and r.seqid.icode.strip() == spec["insertion_code"]
        and r.name == spec["component"]
    ]
    if len(residues) != 1:
        raise ValueError(
            f"{len(residues)} residues match substrate chain {spec['chain']}, number "
            f"{spec['residue_number']}, insertion code {spec['insertion_code']!r} and "
            f"component {spec['component']}; exactly 1 is required"
        )
    source_atoms = [
        a
        for a in residues[0]
        if not a.is_hydrogen() and a.altloc in {"\0", spec["altloc"] or "\0"}
    ]
    atoms = {a.name: a for a in source_atoms}
    if len(atoms) != len(source_atoms) or set(atoms) != set(spec["source_atom_names"]):
        raise ValueError(
            f"The substrate residue {spec['component']} {spec['residue_number']} with "
            f"altloc {spec['altloc']!r} has heavy atoms that differ from "
            "substrate.source_atom_names; it must contain exactly the mapped heavy atoms"
        )
    names = spec["source_atom_names"]
    if any(
        atoms[name].element.atomic_number != molecule.GetAtomWithIdx(i).GetAtomicNum()
        for i, name in enumerate(names)
    ):
        raise ValueError(
            "The elements of the deposited atoms named in substrate.source_atom_names "
            "differ from the elements of the molecule at the same positions; they must "
            "be equal"
        )
    bonds = source_block.get_mmcif_category("_chem_comp_bond.")
    orders = {"sing": 1.0, "doub": 2.0, "trip": 3.0, "arom": 1.5}
    observed = {}
    for i, comp in enumerate(bonds.get("comp_id", [])):
        a, b = bonds["atom_id_1"][i], bonds["atom_id_2"][i]
        if comp == spec["component"] and a in atoms and b in atoms:
            pair = tuple(sorted((names.index(a), names.index(b))))
            if pair in observed or bonds["value_order"][i].lower() not in orders:
                raise ValueError(
                    f"The bond table of component {comp} repeats the bond {a}-{b} or gives "
                    "an order other than sing, doub, trip or arom; one supported order "
                    "per bond is required"
                )
            observed[pair] = orders[bonds["value_order"][i].lower()]
    expected = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondTypeAsDouble()
        for b in molecule.GetBonds()
    }
    if observed != expected:
        raise ValueError(
            "The heavy-atom bonds of the deposited component differ from the bonds of "
            "the molecule; they must be equal"
        )
    positioned = Chem.Mol(molecule)
    positioned.RemoveAllConformers()
    conf = Chem.Conformer(len(names))
    conf.Set3D(True)
    for i, name in enumerate(names):
        if not all(math.isfinite(v) for v in atoms[name].pos.tolist()):
            raise ValueError(
                f"The deposited atom {name} has non-finite coordinates; finite "
                "coordinates are required"
            )
        conf.SetAtomPosition(i, atoms[name].pos.tolist())
    positioned.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(positioned, replaceExistingTags=True)
    if Chem.MolToSmiles(positioned) != Chem.MolToSmiles(molecule):
        raise ValueError(
            "The stereochemistry derived from the deposited coordinates differs from that "
            "of the molecule; they must be equal"
        )
    protein = np.array(
        [
            a.pos.tolist()
            for c in selected
            for r in c.get_polymer()
            for a in _heavy(r)[0]
        ]
    )
    ligands = {}
    for chain, row in zip(selected, recipe["target"]["chains"], strict=True):
        anchors = [r for r in chain if r.name == spec["anchor_component"]]
        if len(anchors) != 1:
            raise ValueError(
                f"Target chain {chain.name} has {len(anchors)} residues named "
                f"{spec['anchor_component']}; exactly 1 analogue anchor is required"
            )
        anchor_atoms = _heavy(anchors[0])[0]
        if {a.name for a in anchor_atoms} != set(spec["anchor_atom_names"]) or any(
            _atom(anchor_atoms, name).element.atomic_number
            != molecule.GetAtomWithIdx(i).GetAtomicNum()
            for i, name in enumerate(spec["anchor_atom_names"])
        ):
            raise ValueError(
                f"The atoms of analogue {spec['anchor_component']} in chain {chain.name} "
                "differ from substrate.anchor_atom_names, or their elements differ from "
                "those of the molecule; the names and elements must match"
            )
        anchor = Chem.Mol(molecule)
        anchor.RemoveAllConformers()
        anchor_conf = Chem.Conformer(len(names))
        for i, name in enumerate(spec["anchor_atom_names"]):
            anchor_conf.SetAtomPosition(i, _atom(anchor_atoms, name).pos.tolist())
        anchor.AddConformer(anchor_conf)
        candidate = Chem.Mol(positioned)
        rmsd = rdMolAlign.AlignMol(
            candidate, anchor, atomMap=[(i, i) for i in range(len(names))]
        )
        xyz = candidate.GetConformer().GetPositions().copy()
        distances = np.linalg.norm(xyz[:, None, :] - protein[None, :, :], axis=2)
        overlap = np.maximum(0, 2.2 - distances).ravel()
        candidate = Chem.AddHs(candidate, addCoords=True)
        properties = AllChem.MMFFGetMoleculeProperties(candidate)
        if properties is None:
            raise ValueError(
                "MMFF94 has no parameters for the molecule; the molecule must be covered "
                "by MMFF94 to place hydrogens"
            )
        force = AllChem.MMFFGetMoleculeForceField(candidate, properties)
        for i in range(len(names)):
            force.AddFixedPoint(i)
        force.Initialize()
        if (
            force.Minimize(maxIts=spec["hydrogen_max_iterations"]) != 0
            or not np.array_equal(
                xyz, candidate.GetConformer().GetPositions()[: len(names)]
            )
            or not np.isfinite(candidate.GetConformer().GetPositions()).all()
        ):
            raise ValueError(
                f"The hydrogen optimization for chain {row['id']} did not converge within "
                f"substrate.hydrogen_max_iterations ({spec['hydrogen_max_iterations']}) "
                "or changed a heavy-atom position; convergence with every heavy atom "
                "unchanged is required"
            )
        summary = dict(
            target_chain=row["id"],
            source_chain=spec["chain"],
            source_residue=spec["residue_number"],
            source_altloc=spec["altloc"],
            source_form=spec["source_form"],
            source_occupancy=sorted({a.occ for a in atoms.values()}),
            source_atom_mapping=names,
            anchor_atom_mapping=spec["anchor_atom_names"],
            analogue_rmsd_angstrom=rmsd,
            protein_geometric_overlap_score=float(overlap @ overlap),
            formal_charge=Chem.GetFormalCharge(candidate),
            canonical_isomeric_smiles=Chem.MolToSmiles(Chem.RemoveHs(candidate)),
            heavy_atom_count=len(names),
            atom_count=candidate.GetNumAtoms(),
            heavy_coordinates_preserved_exactly=True,
        )
        candidate.SetProp(
            "ALAMBIQ_COORDINATE_MODEL", json.dumps(summary, sort_keys=True)
        )
        path = output / ("ligand-" + row["id"] + ".sdf")
        with Chem.SDWriter(str(path)) as writer:
            writer.write(candidate)
        summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        ligands[row["id"]] = summary
    return ligands


def display_structure(protein, recipe, output):
    """Canonical polymer labels and positioned components for the shared viewer."""
    gemmi, _, Chem, _, _ = dependencies()
    display = protein.clone()
    ligand_chains = []
    for chain, row in zip(display[0], recipe["target"]["chains"], strict=True):
        for residue in chain:
            if residue.name == "HOH":
                residue.subchain = "W" + row["id"]
                residue.entity_type = gemmi.EntityType.Water
            elif residue.name in {"HID", "HIE", "HIP"}:
                residue.name = "HIS"
        molecule = next(
            iter(
                Chem.SDMolSupplier(
                    str(output / ("ligand-" + row["id"] + ".sdf")), removeHs=False
                )
            )
        )
        ligand_chain = gemmi.Chain("L" + row["id"])
        residue = gemmi.Residue()
        residue.name = "SUB"
        residue.seqid = gemmi.SeqId(1, " ")
        residue.subchain = ligand_chain.name
        residue.entity_type = gemmi.EntityType.NonPolymer
        for i, original in enumerate(molecule.GetAtoms()):
            atom = gemmi.Atom()
            atom.name = original.GetSymbol() + str(i + 1)
            atom.element = gemmi.Element(original.GetSymbol())
            atom.pos = list(molecule.GetConformer().GetAtomPosition(i))
            atom.occ = 1.0
            residue.add_atom(atom)
        ligand_chain.add_residue(residue)
        ligand_chains.append(ligand_chain)
    # Adding chains can reallocate Gemmi's native container. Finish all work
    # on existing chain references before appending positioned components.
    for chain in ligand_chains:
        display[0].add_chain(chain)
    display.setup_entities()
    sequences = {}
    for row in recipe["target"]["chains"]:
        chain = next(c for c in display[0] if c.name == row["id"])
        entity = display.get_entity_of(chain.get_polymer())
        entity.full_sequence = [r.name for r in chain.get_polymer()]
        sequences[entity.name] = row["sequence"]
    display.assign_label_seq_id()
    document = display.make_mmcif_document()
    document.sole_block().set_mmcif_category(
        "_entity_poly.",
        {
            "entity_id": list(sequences),
            "type": ["polypeptide(L)"] * len(sequences),
            "pdbx_seq_one_letter_code_can": list(sequences.values()),
            "pdbx_seq_one_letter_code": list(sequences.values()),
        },
    )
    document.write_file(str(output / "prepared.cif"))


def prepare(paths, supplied, output):
    """Model the termini, position the substrate, and write every file of the model."""
    gemmi, np, _, _, _ = dependencies()
    recipe = coordinate_recipe(supplied)
    output = Path(output)
    output.mkdir(exist_ok=False)
    target = gemmi.cif.read_file(str(paths["structure"])).sole_block()
    donor = gemmi.cif.read_file(str(paths["donor"])).sole_block()
    substrate = gemmi.cif.read_file(str(paths["substrate_source"])).sole_block()
    protein, selected, records = prepare_protein(
        _structure(target), _structure(donor), recipe
    )
    protein.write_pdb(str(output / "protein.pdb"))
    ligands = prepare_ligands(substrate, paths["molecule"], selected, recipe, output)
    display_structure(protein, recipe, output)
    for row in recipe["target"]["chains"]:
        (output / ("sequence-" + row["id"] + ".fasta")).write_text(
            ">" + row["id"] + "\n" + row["sequence"] + "\n"
        )
    (output / "protein.fasta").write_text(
        "".join(
            ">" + row["id"] + "\n" + row["sequence"] + "\n"
            for row in recipe["target"]["chains"]
        )
    )
    record = dict(
        schema_version="0.1",
        preparation_kind="reactive-coordinates",
        parameters=recipe,
        protein_residues=sum(
            len(row["sequence"]) for row in recipe["target"]["chains"]
        ),
        protein_heavy_atom_count=sum(
            1 for c in protein[0] for r in c.get_polymer() for a in r
        ),
        atom_count=sum(1 for c in protein[0] for r in c for a in r)
        + sum(r["atom_count"] for r in ligands.values()),
        water_count=sum(r.name == "HOH" for c in protein[0] for r in c),
        protein_sha256=hashlib.sha256(
            (output / "protein.pdb").read_bytes()
        ).hexdigest(),
        display_structure_sha256=hashlib.sha256(
            (output / "prepared.cif").read_bytes()
        ).hexdigest(),
        ligands=ligands,
        environment_versions=versions(),
        scope="Heavy-coordinate protein terminal model and hydrogen-complete positioned substrates. Explicit terminal geometry precedes bonded relaxation, force-field parameterization, solvation and physical sampling.",
        **records,
    )
    (output / "preparation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return record
