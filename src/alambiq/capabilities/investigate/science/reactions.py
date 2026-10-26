"""Inspect explicit molecular species and researcher-supplied reaction equations."""

import copy
import re
from collections import Counter, defaultdict
from fractions import Fraction

from alambiq.capabilities.shared.records import coefficient, digest, label

PROFILE = "rdkit-explicit-species-v1"
RDKIT_VERSION = "2026.03.1"
ROLES = {
    "substrate",
    "product",
    "cofactor",
    "cosubstrate",
    "byproduct",
    "proton",
    "water",
    "counterion",
}


def chemistry():
    from rdkit import Chem

    return Chem


def rdkit_version():
    """The RDKit that is running. Species identity depends on it, so it is recorded."""
    from rdkit import rdBase

    return rdBase.rdkitVersion


def molecule(smiles):
    Chem = chemistry()
    if not isinstance(smiles, str) or not 1 <= len(smiles) <= 20000:
        raise ValueError("Supply a SMILES string of 1 to 20000 characters")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.parseName = False
    params.allowCXSMILES = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or not 1 <= mol.GetNumAtoms() <= 2000:
        raise ValueError(
            "The SMILES must describe a valid molecule of 1 to 2000 explicit atoms"
        )
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("Every atom of a species must be an element")
    # Preserve atom order and map numbers; canonical identity gets a separate copy.
    return mol


def inspect_compound(smiles):
    Chem = chemistry()
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = molecule(smiles)
    identity = Chem.Mol(mol)
    for atom in identity.GetAtoms():
        atom.SetAtomMapNum(0)
    identity = Chem.RemoveHs(identity)
    canonical = Chem.MolToSmiles(identity, isomericSmiles=True)
    elements, isotopes = Counter(), Counter()
    for atom in Chem.AddHs(mol).GetAtoms():
        elements[atom.GetSymbol()] += 1
        if atom.GetIsotope():
            isotopes[f"{atom.GetIsotope()}{atom.GetSymbol()}"] += 1
    drawer = rdMolDraw2D.MolDraw2DSVG(340, 230)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    atoms = [
        {
            "index": a.GetIdx(),
            "element": a.GetSymbol(),
            "isotope": a.GetIsotope(),
            "map": a.GetAtomMapNum(),
            "formal_charge": a.GetFormalCharge(),
            "hydrogens": a.GetTotalNumHs(includeNeighbors=True),
            "cip_label": a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None,
        }
        for a in mol.GetAtoms()
    ]
    return {
        "chemical_id": "compound-"
        + digest({"profile": PROFILE, "rdkit": rdkit_version(), "smiles": canonical}),
        "input_smiles": smiles,
        "canonical_smiles": canonical,
        "mapped_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(identity),
        "formal_charge": Chem.GetFormalCharge(mol),
        "elements": dict(sorted(elements.items())),
        "isotopes": dict(sorted(isotopes.items())),
        "atoms": atoms,
        "bonds": [
            {
                "atoms": [b.GetBeginAtomIdx(), b.GetEndAtomIdx()],
                "order": b.GetBondTypeAsDouble(),
            }
            for b in mol.GetBonds()
        ],
        "unspecified_stereo": [
            {"type": str(s.type), "index": s.centeredOn}
            for s in Chem.FindPotentialStereo(mol)
            if str(s.specified) == "Unspecified"
        ],
        "radical_electrons": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
        "drawing_svg": drawer.GetDrawingText(),
    }


def validate_request(request):
    fields = {"id", "title", "convention", "source", "conditions", "participants"}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError(
            "Reaction requires id, title, convention, source, conditions and participants"
        )
    for key in ("id", "title"):
        label(request[key], key, 500)
    if request["convention"] != "explicit-species":
        raise ValueError(
            "This profile accounts for explicit-species equations, with all protons and cosubstrates supplied"
        )
    source = request["source"]
    if (
        not isinstance(source, dict)
        or not {"kind", "citation"} <= source.keys()
        or source.keys() - {"kind", "citation", "url", "version", "sha256"}
    ):
        raise ValueError(
            "Reaction source requires kind and citation, with optional url, version and sha256"
        )
    if source["kind"] not in {
        "researcher-specified",
        "curated-reference",
        "publication",
        "patent",
        "rule-generated",
        "model-predicted",
    }:
        raise ValueError("source.kind must be a supported kind, such as researcher-specified or publication")
    for k, v in source.items():
        label(v, f"source.{k}", 3000)
    if "sha256" in source and not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("source.sha256 must be 64 lowercase hexadecimal characters")
    conditions = request["conditions"]
    if (
        not isinstance(conditions, dict)
        or "description" not in conditions
        or conditions.keys()
        - {"description", "pH", "temperature_K", "ionic_strength_M"}
    ):
        raise ValueError(
            "Conditions require description; optional pH, temperature_K and ionic_strength_M preserve reported context"
        )
    label(conditions["description"], "conditions.description", 3000)
    for k, bounds in {
        "pH": (-2, 16),
        "temperature_K": (0, 2000),
        "ionic_strength_M": (0, 100),
    }.items():
        if k in conditions:
            v = conditions[k]
            if (
                type(v) not in (int, float)
                or not bounds[0] <= v <= bounds[1]
                or (k == "temperature_K" and v == 0)
            ):
                raise ValueError(
                    f"{k} requires a finite value in its supported physical range"
                )
    participants = request["participants"]
    if not isinstance(participants, list) or not 2 <= len(participants) <= 100:
        raise ValueError("Supply 2–100 participants including both equation sides")
    identifiers = set()
    for p in participants:
        required = {"id", "side", "role", "smiles", "coefficient"}
        if (
            not isinstance(p, dict)
            or not required <= p.keys()
            or p.keys() - required - {"sources", "reference_smiles"}
        ):
            raise ValueError(
                "Each participant requires id, side, role, smiles and coefficient; sources and reference_smiles are optional"
            )
        label(p["id"], "participant.id")
        if p["id"] in identifiers:
            raise ValueError("Participant IDs must be unique within the equation")
        identifiers.add(p["id"])
        if p["side"] not in {"reactant", "product"} or p["role"] not in ROLES:
            raise ValueError(
                "Participants require reactant/product side and an explicit chemical role"
            )
        if (p["role"] == "substrate" and p["side"] != "reactant") or (
            p["role"] in {"product", "byproduct"} and p["side"] != "product"
        ):
            raise ValueError(
                "Substrate/product roles must agree with their equation side"
            )
        coefficient(p["coefficient"])
        sources = p.get("sources", [])
        if not isinstance(sources, list) or len(sources) > 20:
            raise ValueError(
                "Participant sources require a list of at most 20 identities"
            )
        for s in sources:
            if (
                not isinstance(s, dict)
                or not {"database", "accession", "version"} <= s.keys()
                or s.keys() - {"database", "accession", "version", "url"}
            ):
                raise ValueError(
                    "Participant source requires database, accession and version, plus optional url"
                )
            for k, v in s.items():
                label(v, f"participant.source.{k}", 3000)
    if {p["side"] for p in participants} != {"reactant", "product"}:
        raise ValueError("Supply participants on both equation sides")


def atom_correspondence(participants, diagnostics):
    sides = {side: {} for side in ("reactant", "product")}
    bonds = {side: {} for side in sides}
    heavy_unmapped, inconsistent, total = 0, False, 0
    for p in participants:
        side = p["side"]
        for a in p["atoms"]:
            n = a["map"]
            if not n:
                heavy_unmapped += a["element"] != "H"
                continue
            total += 1
            if n in sides[side]:
                diagnostics.append(
                    {
                        "code": "duplicate-atom-map",
                        "participants": [p["id"], sides[side][n]["participant"]],
                        "message": f"Map {n} appears more than once on the {side} side.",
                    }
                )
                inconsistent = True
            sides[side][n] = {**a, "participant": p["id"]}
        for b in p["bonds"]:
            mapped = [p["atoms"][i]["map"] for i in b["atoms"]]
            if all(mapped):
                bonds[side][tuple(sorted(mapped))] = b["order"]
    left, right = sides["reactant"], sides["product"]
    shared = sorted(left.keys() & right.keys())
    for n in shared:
        if (left[n]["element"], left[n]["isotope"]) != (
            right[n]["element"],
            right[n]["isotope"],
        ):
            diagnostics.append(
                {
                    "code": "mapped-atom-identity",
                    "participants": [left[n]["participant"], right[n]["participant"]],
                    "message": f"Map {n} has a different element or isotope on the two sides.",
                }
            )
            inconsistent = True
    multiplicity = total and any(
        coefficient(p["coefficient"]) != 1
        for p in participants
        if any(a["map"] for a in p["atoms"])
    )
    status = (
        "inconsistent"
        if inconsistent
        else "stoichiometric-ambiguity"
        if multiplicity
        else "absent"
        if not total
        else "partial"
        if heavy_unmapped or left.keys() != right.keys()
        else "complete-heavy-atoms"
    )
    changes = []
    atom_changes = []
    if status == "complete-heavy-atoms":
        for key in sorted(bonds["reactant"].keys() | bonds["product"].keys()):
            before, after = bonds["reactant"].get(key), bonds["product"].get(key)
            if before != after:
                changes.append(
                    {
                        "atoms": list(key),
                        "reactant_order": before,
                        "product_order": after,
                    }
                )
        for n in shared:
            fields = {
                k: {"reactant": left[n][k], "product": right[n][k]}
                for k in ("formal_charge", "hydrogens", "cip_label")
                if left[n][k] != right[n][k]
            }
            if fields:
                atom_changes.append({"map": n, "fields": fields})
    return {
        "status": status,
        "pairs": [{"map": n, "reactant": left[n], "product": right[n]} for n in shared]
        if not inconsistent
        else [],
        "reactant_only": sorted(left.keys() - right.keys()),
        "product_only": sorted(right.keys() - left.keys()),
        "unmapped_heavy_atoms": heavy_unmapped,
        "bond_changes": changes,
        "atom_changes": atom_changes,
    }


def inspect_reaction(request):
    validate_request(request)
    diagnostics, participants = [], []
    source_species = defaultdict(list)
    delta = {"elements": defaultdict(Fraction), "isotopes": defaultdict(Fraction)}
    charge = Fraction(0)
    for supplied in request["participants"]:
        p = {**copy.deepcopy(supplied), **inspect_compound(supplied["smiles"])}
        p["coefficient"] = str(coefficient(supplied["coefficient"]))
        participants.append(p)
        sign = -1 if p["side"] == "reactant" else 1
        amount = sign * coefficient(p["coefficient"])
        for kind in delta:
            for key, count in p[kind].items():
                delta[kind][key] += amount * count
        charge += amount * p["formal_charge"]
        if p["unspecified_stereo"]:
            diagnostics.append(
                {
                    "code": "unspecified-stereochemistry",
                    "participants": [p["id"]],
                    "message": "The species has unspecified stereogenic atoms or bonds.",
                }
            )
        if p["radical_electrons"]:
            diagnostics.append(
                {
                    "code": "radical-species",
                    "participants": [p["id"]],
                    "message": f"The species has {p['radical_electrons']} radical electrons.",
                }
            )
        if "reference_smiles" in p:
            reference = inspect_compound(p["reference_smiles"])
            p["reference_chemical_id"] = reference["chemical_id"]
            if p["chemical_id"] != reference["chemical_id"]:
                diagnostics.append(
                    {
                        "code": "reference-structure-mismatch",
                        "participants": [p["id"]],
                        "message": "The structure and its reference are different species. Compare stereochemistry, charge, isotopes and composition.",
                    }
                )
        for source in p.get("sources", []):
            source_species[
                (source["database"], source["accession"], source["version"])
            ].append(p)
    for key, rows in sorted(source_species.items()):
        if len({p["chemical_id"] for p in rows}) > 1:
            diagnostics.append(
                {
                    "code": "source-identity-ambiguous",
                    "participants": sorted({p["id"] for p in rows}),
                    "message": f"Source {key[0]}:{key[1]} ({key[2]}) is assigned to different species.",
                }
            )
    residuals = {
        kind: {k: str(v) for k, v in sorted(counts.items()) if v}
        for kind, counts in delta.items()
    }
    balanced = not charge and not any(residuals.values())
    if not balanced:
        diagnostics.append(
            {
                "code": "equation-imbalance",
                "participants": [p["id"] for p in participants],
                "message": "Products minus reactants leave a nonzero element, isotope or charge residual. Check the species and coefficients.",
            }
        )
    mapping = atom_correspondence(participants, diagnostics)
    if mapping["status"] in {"partial", "stoichiometric-ambiguity"}:
        diagnostics.append(
            {
                "code": "atom-correspondence-coverage",
                "participants": [p["id"] for p in participants],
                "message": f"The atom maps are {mapping['status']}. Bond changes are reconstructed when the maps cover every heavy atom one to one.",
            }
        )
    equation = defaultdict(Fraction)
    for p in participants:
        equation[p["chemical_id"]] += coefficient(p["coefficient"]) * (
            -1 if p["side"] == "reactant" else 1
        )
    if not any(equation.values()):
        diagnostics.append(
            {
                "code": "zero-net-transformation",
                "participants": [p["id"] for p in participants],
                "message": "Every species cancels across the equation, so the net transformation is zero.",
            }
        )
    return {
        "schema_version": "0.1",
        "profile": PROFILE,
        "rdkit_version": rdkit_version(),
        "request": copy.deepcopy(request),
        "request_sha256": digest(request),
        "reaction_id": "reaction-"
        + digest(
            {
                "profile": PROFILE,
                "convention": request["convention"],
                "stoichiometry": {k: str(v) for k, v in sorted(equation.items()) if v},
            }
        ),
        "participants": participants,
        "balance": {
            **residuals,
            "charge": str(charge),
            "balanced": bool(balanced),
            "basis": "products-minus-reactants",
        },
        "mapping": mapping,
        "diagnostics": diagnostics,
        "method": "A species is identified by its RDKit canonical isomeric SMILES. The source SMILES and the atom maps are retained separately. Explicit hydrogens and atom-map labels are removed from the identity copy only. Charge, stereochemistry, isotope labels, disconnected fragments and tautomeric form are part of the identity. All participants enter the balance, in exact rational arithmetic. Conditions are recorded as context, and the species determine the balance. CIP labels are the priorities that RDKit assigns to each species.",
    }
