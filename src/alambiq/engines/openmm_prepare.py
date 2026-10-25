"""Prepare one explicitly selected standard protein chain with OpenMM.

Runs inside the openmm environment. The standard library is imported at the top;
numpy, OpenMM and PDBFixer are loaded when a request runs. The caller supplies
the request and an output directory; the engine writes the prepared system,
state, topology and manifest there and returns the manifest.
"""

import hashlib
import io
import json
import math
import random
from collections import Counter
from pathlib import Path

import openmm_common as diagnostics

np = openmm = app = unit = None


def _load():
    global np, openmm, app, unit
    import numpy
    import openmm as _openmm
    from openmm import app as _app
    from openmm import unit as _unit

    np, openmm, app, unit = numpy, _openmm, _app, _unit


LETTERS = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)


def write_prepared_cif(topology, positions, stream):
    """Add topology-owned identities to OpenMM's otherwise atom-only mmCIF."""
    from openmm.app.internal.pdbx.reader.PdbxContainers import DataCategory
    from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader
    from openmm.app.internal.pdbx.writer.PdbxWriter import PdbxWriter

    raw = io.StringIO()
    app.PDBxFile.writeFile(topology, positions, raw, keepIds=True)
    blocks = []
    PdbxReader(io.StringIO(raw.getvalue())).read(blocks)
    block = blocks[0]
    atoms = block.getObj("atom_site")
    protein = [r for r in topology.residues() if r.chain.id == "A"]
    if not protein or any(r.name not in LETTERS for r in protein):
        raise ValueError(
            "Prepared protein metadata requires the normalized standard chain A"
        )
    sequence = "".join(LETTERS[r.name] for r in protein)
    entities = [["1", "polymer"]]
    asymmetric = {"A": "1"}
    components = {}
    topology_atoms = list(topology.atoms())
    if atoms.getRowCount() != len(topology_atoms):
        raise ValueError("OpenMM mmCIF atom count differs from prepared topology")
    for index, atom in enumerate(topology_atoms):
        residue = atom.residue
        if atoms.getValue("label_atom_id", index) != atom.name:
            raise ValueError("OpenMM mmCIF atom order differs from prepared topology")
        if residue.chain.id == "A":
            asym, entity, seq = "A", "1", residue.id
        else:
            if residue.name not in {"HOH", "NA", "CL"}:
                raise ValueError(
                    f"Unexpected component {residue.name}; this profile allows only protein, water, Na and Cl"
                )
            if residue.name not in components:
                components[residue.name] = str(len(entities) + 1)
                entities.append(
                    [
                        components[residue.name],
                        "water" if residue.name == "HOH" else "non-polymer",
                    ]
                )
            entity = components[residue.name]
            # Water shares one asym; every ion has its own instance. Author IDs
            # remain those of the exact serialized topology, not these labels.
            asym = "W" if residue.name == "HOH" else f"I{residue.index + 1}"
            asymmetric[asym] = entity
            seq = "."
        for key, value in [
            ("label_asym_id", asym),
            ("label_entity_id", entity),
            ("label_seq_id", seq),
        ]:
            atoms.setValue(value, key, index)
    for name, columns, rows in [
        ("entity", ["id", "type"], entities),
        (
            "struct_asym",
            ["id", "entity_id"],
            [list(pair) for pair in asymmetric.items()],
        ),
        (
            "entity_poly",
            ["entity_id", "type", "pdbx_seq_one_letter_code_can"],
            [["1", "polypeptide(L)", sequence]],
        ),
        (
            "entity_poly_seq",
            ["entity_id", "num", "mon_id", "hetero"],
            [["1", str(i + 1), r.name, "n"] for i, r in enumerate(protein)],
        ),
        (
            "struct",
            ["title"],
            [["Alambiq OpenMM prepared protein in water; minimized coordinates"]],
        ),
    ]:
        block.append(DataCategory(name, columns, rows))
    PdbxWriter(stream).write(blocks)


def reject_alternates(path, format):
    if format == "pdb":
        if any(
            line.startswith(("ATOM  ", "HETATM")) and line[16:17] not in {" ", ""}
            for line in path.read_text().splitlines()
        ):
            raise ValueError(
                "Alternate atom locations require explicit upstream selection"
            )
    else:
        # Reuse the exact parser shipped with pinned OpenMM.
        from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

        blocks = []
        with path.open() as stream:
            PdbxReader(stream).read(blocks)
        if len(blocks) != 1:
            raise ValueError("Supply one mmCIF data block")
        atoms = blocks[0].getObj("atom_site")
        if atoms is None:
            raise ValueError("Missing atom_site coordinates")
        column = atoms.getAttributeIndex("label_alt_id")
        if column != -1 and any(
            row[column] not in {".", "?", ""} for row in atoms.getRowList()
        ):
            raise ValueError(
                "Alternate atom locations require explicit upstream selection"
            )


def reject_substitution_crosslinks(topology, residues, substitutions):
    targets = {residues[r["prepared_residue_number"] - 1] for r in substitutions}
    peptides = {
        frozenset(((left, "C"), (right, "N")))
        for left, right in zip(residues, residues[1:], strict=False)
    }
    for a, b in topology.bonds():
        if (
            a.residue != b.residue
            and (a.residue in targets or b.residue in targets)
            and frozenset(((a.residue, a.name), (b.residue, b.name))) not in peptides
        ):
            raise ValueError(
                "Substitution at a crosslinked residue requires a separate chemical preparation"
            )


def resolve_substitutions(residues, requested):
    """Resolve original source identities before normalization or any editing."""
    names = {letter: name for name, letter in LETTERS.items()}
    result, seen = [], set()
    for row in requested:
        matches = [
            r
            for r in residues
            if (r.id, r.insertionCode.strip())
            == (row["residue_id"], row["insertion_code"].strip())
        ]
        if len(matches) != 1:
            raise ValueError(
                "Substitution must identify exactly one selected source residue"
            )
        residue = matches[0]
        index = residues.index(residue)
        if (
            index in seen
            or row["wild_type"] not in names
            or row["mutant"] not in names
            or row["wild_type"] == row["mutant"]
        ):
            raise ValueError(
                "Substitutions require distinct standard identities and unique source sites"
            )
        if residue.name != names[row["wild_type"]]:
            raise ValueError(
                "Substitution wild_type differs from the selected source residue"
            )
        seen.add(index)
        result.append(
            {
                **row,
                "insertion_code": residue.insertionCode.strip(),
                "source_chain_id": residue.chain.id,
                "source_residue_name": residue.name,
                "prepared_chain_id": "A",
                "prepared_residue_number": index + 1,
                "prepared_residue_name": names[row["mutant"]],
            }
        )
    if result:
        # Check the original topology before selected-chain exclusion can remove
        # the other end of an inter-chain disulfide or other covalent link.
        reject_substitution_crosslinks(residues[0].chain.topology, residues, result)
    return result


def complete_heavy_atoms(modeller, platform, seed, substitutions=()):
    """Explicit standard substitutions/completion; preserve all retained coordinates."""
    from importlib.metadata import version

    import pdbfixer
    from pdbfixer import PDBFixer

    residues = list(modeller.topology.residues())
    identities = [(r.id, r.name, r.chain.id) for r in residues]
    for residue in residues:
        if not {"N", "CA", "C", "O"} <= {a.name for a in residue.atoms()}:
            raise ValueError(
                "Atom completion requires intact N/CA/C/O backbone anchors"
            )
    before = {
        (a.residue.index, a.name): np.array(
            modeller.positions[a.index].value_in_unit(unit.nanometer)
        )
        for a in modeller.topology.atoms()
    }

    def inter_residue_bonds(topology):
        return {
            tuple(sorted(((a.residue.index, a.name), (b.residue.index, b.name))))
            for a, b in topology.bonds()
            if a.residue != b.residue
        }

    crosslinks = inter_residue_bonds(modeller.topology)
    targets = {row["prepared_residue_number"] - 1 for row in substitutions}
    reject_substitution_crosslinks(modeller.topology, residues, substitutions)
    stream = io.StringIO()
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, stream, keepIds=True)
    stream.seek(0)
    fixer = PDBFixer(pdbxfile=stream, platform=platform)
    # Keep full precision and existing bonds instead of the text round-trip.
    fixer.topology, fixer.positions = modeller.topology, modeller.positions
    fixer.missingResidues = {}
    for row in substitutions:
        index = row["prepared_residue_number"] - 1
        if index not in range(len(residues)) or identities[index] != (
            str(index + 1),
            row["source_residue_name"],
            "A",
        ):
            raise ValueError("Resolved substitution differs from normalized topology")
        if (
            row["prepared_residue_name"] not in LETTERS
            or row["prepared_residue_name"] not in fixer.templates
        ):
            raise ValueError(
                "Substitutions require bundled standard PDBFixer templates"
            )
        source_template = fixer.templates[row["source_residue_name"]]
        allowed = {a.name for a in source_template.topology.atoms()} | {"OXT"}
        if any(a.name not in allowed for a in residues[index].atoms()):
            raise ValueError("Substitution source residue has unexpected heavy atoms")
        identities[index] = (str(index + 1), row["prepared_residue_name"], "A")
    if substitutions:
        fixer.applyMutations(
            [
                f"{row['source_residue_name']}-{row['prepared_residue_number']}-{row['prepared_residue_name']}"
                for row in substitutions
            ],
            "A",
        )
    retained = {(a.residue.index, a.name) for a in fixer.topology.atoms()}
    removed = set(before) - retained
    if any(
        index not in targets or name in {"N", "CA", "C", "O"} for index, name in removed
    ):
        raise ValueError(
            "Substitution removed an unchanged residue atom or backbone anchor"
        )
    fixer.findMissingAtoms()
    expected = {
        (r.index, a.name) for r, atoms in fixer.missingAtoms.items() for a in atoms
    }
    expected |= {
        (r.index, name) for r, names in fixer.missingTerminals.items() for name in names
    }
    fixer.addMissingAtoms(seed=seed)
    after = {
        (a.residue.index, a.name): np.array(
            fixer.positions[a.index].value_in_unit(unit.nanometer)
        )
        for a in fixer.topology.atoms()
    }
    if [(r.id, r.name, r.chain.id) for r in fixer.topology.residues()] != identities:
        raise ValueError("Completion changed residue identities")
    if set(after) != retained | expected or any(
        not np.allclose(after[key], value, atol=1e-12, rtol=0)
        for key, value in before.items()
        if key in retained
    ):
        raise ValueError(
            "Completion changed existing heavy atoms or added unexpected atoms"
        )
    if substitutions and inter_residue_bonds(fixer.topology) != crosslinks:
        raise ValueError(
            "Substitution/completion changed an inter-residue bond; inspect the chemical state"
        )
    if not all(np.isfinite(value).all() for value in after.values()):
        raise ValueError("Completion produced non-finite coordinates")
    directory = Path(pdbfixer.__file__).parent
    templates = {
        name: hashlib.sha256(
            (directory / "templates" / (name + ".pdb")).read_bytes()
        ).hexdigest()
        for name in sorted(
            {r.name for r in fixer.topology.residues()}
            | {r["source_residue_name"] for r in substitutions}
        )
    }
    return app.Modeller(fixer.topology, fixer.positions), {
        "provider": "pdbfixer",
        "version": version("pdbfixer"),
        "seed": seed,
        "templates_sha256": templates,
        "existing_heavy_atoms_preserved_before_minimization": not removed,
        "retained_heavy_atoms_preserved_before_minimization": True,
        "substitutions": list(substitutions),
        "removed_atoms": [
            {
                "prepared_residue_number": i + 1,
                "atom_name": name,
                "source_position_nm": before[(i, name)].tolist(),
            }
            for i, name in sorted(removed)
        ],
        "added_atoms": [
            {
                "prepared_residue_number": i + 1,
                "residue_name": residues[i].name,
                "atom_name": name,
                "initial_position_nm": after[(i, name)].tolist(),
            }
            for i, name in sorted(expected)
        ],
    }


def prepare(config, output):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    source = Path(config["structure"])
    reject_alternates(source, config["structure_format"])
    parser = app.PDBxFile if config["structure_format"] == "mmcif" else app.PDBFile
    supplied = parser(str(source))
    # OpenMM's PDB reader uses a space for a blank insertion code; mmCIF uses
    # the empty string. Both represent the same source residue identity.
    for residue in supplied.topology.residues():
        residue.insertionCode = residue.insertionCode.strip()
    if config["model_index"] >= supplied.getNumFrames():
        raise ValueError(f"model_index {config['model_index']} (0-based) is unavailable in the input structure")
    positions = supplied.getPositions(frame=config["model_index"])
    xyz = np.asarray(positions.value_in_unit(unit.nanometer))
    if not np.isfinite(xyz).all():
        raise ValueError("The input coordinates contain non-finite values")
    selected = [c for c in supplied.topology.chains() if c.id == config["chain_id"]]
    if len(selected) != 1:
        raise ValueError("chain_id must select exactly one OpenMM-parsed chain")
    residues = [r for r in selected[0].residues() if r.name not in {"HOH", "WAT"}]
    if not 3 <= len(residues) <= 1000 or any(r.name not in LETTERS for r in residues):
        raise ValueError(
            "Selected chain must contain 3–1000 standard protein residues; chains with cofactors or nonstandard residues require another preparation profile"
        )
    for left, right in zip(residues, residues[1:], strict=False):
        carbon = [a for a in left.atoms() if a.name == "C"]
        nitrogen = [a for a in right.atoms() if a.name == "N"]
        if (
            len(carbon) != 1
            or len(nitrogen) != 1
            or not 0.10
            <= np.linalg.norm(xyz[carbon[0].index] - xyz[nitrogen[0].index])
            <= 0.18
        ):
            raise ValueError(
                "Backbone is incomplete or discontinuous"
            )
    residue_set = set(residues)
    retained = [
        a
        for a in supplied.topology.atoms()
        if a.residue in residue_set and a.element != app.element.hydrogen
    ]
    mapping = [
        {
            "source_residue_index": r.index,
            "source_chain_id": r.chain.id,
            "source_residue_id": r.id,
            "source_insertion_code": r.insertionCode,
            "name": r.name,
            "prepared_name": r.name,
            "prepared_chain_id": "A",
            "prepared_residue_number": i + 1,
        }
        for i, r in enumerate(residues)
    ]
    substitutions = resolve_substitutions(residues, config.get("substitutions", []))
    if substitutions and config.get("heavy_atom_completion") != "pdbfixer-standard":
        raise ValueError("Explicit substitutions require pdbfixer-standard completion")
    for row in substitutions:
        mapping[row["prepared_residue_number"] - 1]["prepared_name"] = row[
            "prepared_residue_name"
        ]
    heavy = {(residues.index(a.residue), a.name): a.index for a in retained}
    if len(heavy) != len(retained):
        raise ValueError("A residue contains duplicate heavy-atom names")
    retained_set = set(retained)
    excluded = [
        {
            "chain_id": r.chain.id,
            "residue_id": r.id,
            "insertion_code": r.insertionCode,
            "residue_name": r.name,
            "atom_count": len(list(r.atoms())),
        }
        for r in supplied.topology.residues()
        if r not in residue_set
    ]
    selection = {
        "policy": config.get("source_component_policy", "require-protein-only"),
        "selected_chain_id": config["chain_id"],
        "excluded_residues": excluded,
        "scope": "One supplied standard-protein chain; biological assembly selection remains researcher-specified",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    if excluded and selection["policy"] == "require-protein-only":
        raise ValueError(
            "Protein-only preparation would discard source components (see source-selection.json). "
            "Use assembly preparation to retain them, or set source_component_policy to selected-chain-only"
        )
    requested_variants = [None] * len(residues)
    for row in config.get("residue_variants", []):
        matches = [
            i
            for i, r in enumerate(residues)
            if (r.id, r.insertionCode)
            == (row["residue_id"], row["insertion_code"].strip())
        ]
        if len(matches) != 1:
            raise ValueError(
                "Residue variant must identify exactly one selected source residue"
            )
        requested_variants[matches[0]] = row["variant"]
    removed = Counter(
        "hydrogen"
        if a.element == app.element.hydrogen and a.residue in residue_set
        else f"{a.residue.chain.id}:{a.residue.name}"
        for a in supplied.topology.atoms()
        if a not in retained_set
    )
    modeller = app.Modeller(supplied.topology, positions)
    modeller.delete([a for a in supplied.topology.atoms() if a not in retained_set])
    for i, residue in enumerate(modeller.topology.residues()):
        residue.id, residue.insertionCode, residue.chain.id = str(i + 1), "", "A"
    implicit = config.get("solvent_model") == "gbn2"
    forcefields = [
        "amber14/protein.ff14SB.xml",
        "implicit/gbn2.xml" if implicit else "amber14/tip3p.xml",
    ]
    forcefield = app.ForceField(*forcefields)
    platform = openmm.Platform.getPlatformByName(config["platform"])
    for key, value in config["platform_properties"].items():
        platform.setPropertyDefaultValue(key, value)
    completion = None
    if config.get("heavy_atom_completion", "none") == "pdbfixer-standard":
        modeller, completion = complete_heavy_atoms(
            modeller, platform, config["seed"], substitutions
        )
        # A deleted source atom that is later rebuilt (for example terminal OXT)
        # is a new atom even when its residue/name matches the source.
        for row in completion["removed_atoms"]:
            heavy.pop((row["prepared_residue_number"] - 1, row["atom_name"]))
        if completion["removed_atoms"]:
            removed["substitution-heavy-atom"] += len(completion["removed_atoms"])
    variants = modeller.addHydrogens(
        forcefield, pH=config["ph"], variants=requested_variants, platform=platform
    )
    protein_atoms = modeller.topology.getNumAtoms()
    if implicit:
        modeller.topology.setPeriodicBoxVectors(None)
        kappa = 367.434915 * math.sqrt(
            config["ionic_strength_molar"]
            / (config["solvent_dielectric"] * config["temperature_kelvin"])
        )
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
            soluteDielectric=config["solute_dielectric"],
            solventDielectric=config["solvent_dielectric"],
            implicitSolventKappa=kappa / unit.nanometer,
            sasaMethod="ACE",
        )
    else:
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=config["padding_nm"] * unit.nanometer,
            boxShape=config.get("box_shape", "dodecahedron"),
            ionicStrength=config["ionic_strength_molar"] * unit.molar,
            neutralize=True,
        )
        if modeller.topology.getNumAtoms() > 250000:
            raise ValueError("The prepared system exceeds the limit of 250,000 atoms of this profile")
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=0.9 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(
        modeller.topology, system, integrator, platform, config["platform_properties"]
    )
    simulation.context.setPositions(modeller.positions)
    before_state = simulation.context.getState(energy=True, forces=True)
    before = before_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    before_forces = diagnostics.force_diagnostics(before_state, unit)
    simulation.minimizeEnergy(
        tolerance=config.get("minimization_tolerance_kj_mol_nm", 10.0)
        * unit.kilojoule_per_mole
        / unit.nanometer,
        maxIterations=config["minimization_max_iterations"],
    )
    state = simulation.context.getState(
        getPositions=True, getVelocities=True, getEnergy=True, getForces=True
    )
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if (
        not math.isfinite(before)
        or not math.isfinite(energy)
        or not np.isfinite(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        ).all()
    ):
        raise ValueError("The prepared positions or the potential energy are not finite")
    (output / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    (output / "state.xml").write_text(openmm.XmlSerializer.serialize(state))
    with (output / "topology.pdb").open("w") as stream:
        app.PDBFile.writeFile(
            modeller.topology, state.getPositions(), stream, keepIds=True
        )
    with (output / "prepared.cif").open("w") as stream:
        write_prepared_cif(modeller.topology, state.getPositions(), stream)
    sequence = "".join(
        LETTERS[r.name] for r in modeller.topology.residues() if r.chain.id == "A"
    )
    (output / "protein.fasta").write_text(">prepared-protein\n" + sequence + "\n")
    data_dir = Path(app.__file__).parent / "data"
    manifest = {
        "schema_version": "0.1",
        "openmm_version": openmm.version.short_version,
        "parameters": {k: v for k, v in config.items() if k != "structure"},
        "source_model_count": supplied.getNumFrames(),
        "source_model_index": config["model_index"],
        "protein_force_field": "amber-ff14SB",
        "water_model": "implicit-GBn2/ACE" if implicit else "tip3p",
        "structure_metadata": "Prepared topology supplies polymer sequence, entity and asym identities; coordinates and author IDs are OpenMM's export",
        "forcefield_files_sha256": {
            name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in forcefields
        },
        "protonation_policy": "Remove supplied protein hydrogens; OpenMM Modeller pH-dependent variants and hydrogen placement",
        "residue_variants": variants,
        "source_selection": selection,
        "environment": {
            "preparation_ph": config["ph"],
            "protonation": "Fixed OpenMM residue variants with explicit source-indexed overrides; the defaults are the variants OpenMM Modeller selects at the preparation pH, with local rules for cysteines in disulfides and for neutral histidines",
            "added_salt_molar": config["ionic_strength_molar"],
            "salt": "NaCl",
            "salt_convention": "Added bulk salt excludes neutralizing counterions; integer ion counts approximate the requested concentration",
            "residue_counts": dict(
                Counter(
                    r.name
                    for r in modeller.topology.residues()
                    if r.name in {"NA", "CL"}
                )
            ),
        },
        "disulfide_policy": "OpenMM input parser topology bonds and Modeller cysteine variants",
        "nonbonded_method": "NoCutoff" if implicit else "PME",
        "constraints": "HBonds; rigid water",
        "minimization_max_iterations": config["minimization_max_iterations"],
        "potential_energy_before_kj_mol": before,
        "potential_energy_after_kj_mol": energy,
        "forces_before": before_forces,
        "forces_after": diagnostics.force_diagnostics(state, unit),
        "periodic_cell": None
        if implicit
        else diagnostics.cell_diagnostics(state, unit, 0.9),
        "box_shape": None if implicit else config.get("box_shape", "dodecahedron"),
        "protein_geometry": diagnostics.protein_geometry_diagnostics(
            modeller.topology,
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            system,
        ),
        "minimization_assessment": "OpenMM minimization that stops at the force tolerance or at the iteration limit (`minimization_max_iterations`), whichever is reached first; the recorded energy is the potential energy at the stopping point",
        "protein_residues": len(residues),
        "protein_atoms": protein_atoms,
        "atom_count": system.getNumParticles(),
        "residue_counts": dict(Counter(r.name for r in modeller.topology.residues())),
        "removed_source_atoms": dict(removed),
        "residue_mapping": mapping,
        "substitutions": substitutions,
        "substitution_scope": "Explicit standard-residue modeling with bundled templates; substituted side chains are placed by PDBFixer with the supplied seed, retained atoms are preserved, and the system is then minimized",
        "heavy_atom_completion": completion,
        "added_heavy_atoms": [
            {
                "prepared_atom_index": a.index,
                "prepared_residue_number": a.residue.index + 1,
                "atom_name": a.name,
            }
            for a in modeller.topology.atoms()
            if a.residue.index < len(residues)
            and a.element != app.element.hydrogen
            and (a.residue.index, a.name) not in heavy
        ],
        "heavy_atom_mapping": [
            {
                "source_atom_index": heavy[(a.residue.index, a.name)],
                "prepared_atom_index": a.index,
            }
            for a in modeller.topology.atoms()
            if (a.residue.index, a.name) in heavy
        ],
        "platform": platform.getName(),
        "platform_properties": {
            key: platform.getPropertyValue(simulation.context, key)
            for key in platform.getPropertyNames()
        },
    }
    if implicit:
        manifest["environment"].update(
            solvent_model="gbn2",
            temperature_kelvin=config["temperature_kelvin"],
            solute_dielectric=config["solute_dielectric"],
            solvent_dielectric=config["solvent_dielectric"],
            screening_kappa_per_nm=kappa,
            salt_convention="Debye-Huckel screening at the recorded ionic strength, temperature and fixed dielectric; ions enter only through the screening",
            scope="Continuum equilibrium solvation with ACE surface-area term",
        )
    (output / "preparation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def run(request):
    """Prepare the chain and return the manifest that was written beside the outputs."""
    _load()
    output = Path(request["output"])
    prepare(request["config"], output)
    manifest = json.loads((output / "preparation.json").read_text())
    manifest["versions"] = {
        "openmm": openmm.version.short_version,
        "numpy": np.__version__,
    }
    (output / "preparation.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
