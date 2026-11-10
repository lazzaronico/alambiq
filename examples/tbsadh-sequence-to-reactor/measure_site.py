"""Where does the predicted acetophenone sit relative to the deposited catalytic site?

The deposited structure 1YKF holds NADP and a catalytic zinc in each of the four subunits of
TbSADH. The prediction holds the four chains and the substrate, and neither the cofactor nor
the zinc. This script superposes the predicted tetramer on 1YKF by the alpha carbons (over the
assignment of predicted chains to deposited chains with the lowest RMSD, because Boltz-2 names
the chains of a homotetramer in no particular order of the crystal) and reports:

- the alpha-carbon RMSD of the fit;
- the distance of the carbonyl oxygen of the substrate to each deposited zinc;
- the distance of its carbonyl carbon to C4 of the nicotinamide of each deposited NADP, the
  carbon that receives the hydride.

Usage:
    python measure_site.py --run-dir RUN [--reference 1YKF.cif]
    python measure_site.py --predicted predicted.cif [--reference 1YKF.cif]

`--run-dir` reads the predicted structure of the `complex` step from the run directory. The
script needs `gemmi` and `numpy`. It is a measurement outside the campaign, as the comparison
of the pose with the pockets is in the other examples.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path), format=gemmi.CoorFormat.Mmcif)
    structure.setup_entities()
    return structure


def alpha_carbons(structure: gemmi.Structure) -> dict[tuple[str, int], np.ndarray]:
    found = {}
    for chain in structure[0]:
        for residue in chain:
            atom = residue.find_atom("CA", "*")
            if atom is not None and residue.name != "HOH":
                found[(chain.name, residue.seqid.num)] = np.array(atom.pos.tolist())
    return found


def kabsch(moving: np.ndarray, fixed: np.ndarray):
    """The rotation and the two centroids that best superpose `moving` on `fixed`."""
    centre_moving, centre_fixed = moving.mean(0), fixed.mean(0)
    u, _, vt = np.linalg.svd((moving - centre_moving).T @ (fixed - centre_fixed))
    sign = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, sign]) @ u.T
    moved = (rotation @ (moving - centre_moving).T).T + centre_fixed
    rmsd = float(np.sqrt(((moved - fixed) ** 2).sum(1).mean()))
    return rotation, centre_moving, centre_fixed, rmsd


def fit_chains(predicted: gemmi.Structure, reference: gemmi.Structure) -> dict:
    """The superposition over the best assignment of predicted chains to deposited chains."""
    p, r = alpha_carbons(predicted), alpha_carbons(reference)
    chains = sorted({c for c, _ in p} & {c for c, _ in r})
    best = None
    for assignment in itertools.permutations(chains):
        mapping = dict(zip(chains, assignment, strict=True))
        pairs = [
            (key, (mapping[key[0]], key[1]))
            for key in sorted(p)
            if key[0] in mapping and (mapping[key[0]], key[1]) in r
        ]
        moving = np.array([p[a] for a, _ in pairs])
        fixed = np.array([r[b] for _, b in pairs])
        rotation, cm, cf, rmsd = kabsch(moving, fixed)
        if best is None or rmsd < best["rmsd_angstrom"]:
            best = {
                "rmsd_angstrom": rmsd,
                "residues": len(pairs),
                "assignment": mapping,
                "rotation": rotation,
                "moving_centre": cm,
                "fixed_centre": cf,
            }
    return best


def site(predicted: gemmi.Structure, reference: gemmi.Structure) -> dict:
    fit = fit_chains(predicted, reference)

    def move(xyz):
        return fit["rotation"] @ (np.array(xyz) - fit["moving_centre"]) + fit["fixed_centre"]

    ligands = [
        residue
        for chain in predicted[0]
        for residue in chain
        if residue.name != "HOH" and residue.find_atom("CA", "*") is None
    ]
    if len(ligands) != 1:
        raise ValueError(f"expected one ligand in the prediction, found {len(ligands)}")
    atoms = {a.name: move(a.pos.tolist()) for a in ligands[0]}
    oxygen = next(n for n in atoms if n.startswith("O"))
    carbon = min(
        (n for n in atoms if n.startswith("C")),
        key=lambda n: np.linalg.norm(atoms[n] - atoms[oxygen]),
    )
    zinc, nicotinamide = {}, {}
    for chain in reference[0]:
        for residue in chain:
            if residue.name == "ZN":
                zinc[chain.name] = np.array(residue[0].pos.tolist())
            if residue.name == "NAP":
                nicotinamide[chain.name] = np.array(residue.find_atom("C4N", "*").pos.tolist())
    return {
        "alpha_carbon_rmsd_angstrom": fit["rmsd_angstrom"],
        "residues_fitted": fit["residues"],
        "predicted_chain_to_deposited_chain": fit["assignment"],
        "carbonyl_oxygen": oxygen,
        "carbonyl_carbon": carbon,
        "oxygen_to_zinc_angstrom": {
            c: float(np.linalg.norm(atoms[oxygen] - v)) for c, v in zinc.items()
        },
        "carbon_to_nicotinamide_c4_angstrom": {
            c: float(np.linalg.norm(atoms[carbon] - v)) for c, v in nicotinamide.items()
        },
    }


def predicted_from_run(run_dir: Path) -> Path:
    record = json.loads((run_dir / "run.json").read_text())
    key = next(s["key"] for s in record["steps"] if s["id"] == "complex")
    outputs = json.loads((run_dir / "cache" / key[:2] / f"{key}.json").read_text())["outputs"]
    return Path(outputs["samples"][0]["structure"].replace("{store}", str(run_dir / "artifacts")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--predicted", type=Path)
    parser.add_argument("--reference", type=Path, default=HERE / "1YKF.cif")
    arguments = parser.parse_args()
    if (arguments.run_dir is None) == (arguments.predicted is None):
        parser.error("give one of --run-dir and --predicted")
    path = arguments.predicted or predicted_from_run(arguments.run_dir)
    result = site(read(path), read(arguments.reference))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
