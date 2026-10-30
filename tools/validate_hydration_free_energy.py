"""Compute hydration free energies of small molecules with `molecule.compute_hydration_free_energy`.

Run it on a CUDA host with the OpenFE environment configured:

    ALAMBIQ_ENV_OPENFE=/path/to/python python tools/validate_hydration_free_energy.py \
        --out hydration-validation.json --freesolv /path/to/database.txt

Each molecule of the list below runs through the capability with the settings of the
command line (1 repeat of 0.5 ns of solvent production by default). With `--freesolv`, the
script reads the FreeSolv database (Mobley and Guthrie, J Comput Aided Mol Des 2014), finds
each molecule by its canonical SMILES, and adds the experimental hydration free energy and
its uncertainty. The result file holds the calculated value, the repeats, and the difference
to experiment. The file is written after every molecule, and molecules already in it are
skipped, so an interrupted run continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from alambiq import api

MOLECULES = [
    ("methane", "C"),
    ("ethanol", "CCO"),
    ("benzene", "c1ccccc1"),
    ("phenol", "Oc1ccccc1"),
]


def freesolv(path: Path) -> dict[str, tuple[float, float]]:
    """The experimental value and uncertainty of each molecule, by canonical SMILES."""
    from rdkit import Chem

    table = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(";")]
        molecule = Chem.MolFromSmiles(fields[1])
        if molecule is not None:
            table[Chem.MolToSmiles(molecule)] = (float(fields[3]), float(fields[4]))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--freesolv", type=Path)
    parser.add_argument("--only", help="comma-separated molecule names")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--production-ns", type=float, default=0.5)
    parser.add_argument("--equilibration-ns", type=float, default=0.1)
    parser.add_argument("--platform", default="CUDA")
    arguments = parser.parse_args()

    from rdkit import Chem

    experiment = freesolv(arguments.freesolv) if arguments.freesolv else {}
    chosen = set(arguments.only.split(",")) if arguments.only else None
    result = (
        json.loads(arguments.out.read_text())
        if arguments.out.exists()
        else {
            "repeats": arguments.repeats,
            "production_ns": arguments.production_ns,
            "equilibration_ns": arguments.equilibration_ns,
            "platform": arguments.platform,
            "molecules": [],
        }
    )
    done = {m["molecule"] for m in result["molecules"]}
    for name, smiles in MOLECULES:
        if name in done or (chosen and name not in chosen):
            continue
        with tempfile.TemporaryDirectory() as directory:
            step = {
                "id": "hfe",
                "capability": "molecule.compute_hydration_free_energy",
                "params": {
                    "smiles": smiles,
                    "repeats": arguments.repeats,
                    "production_ns": arguments.production_ns,
                    "equilibration_ns": arguments.equilibration_ns,
                    "platform": arguments.platform,
                },
            }
            outcome = api.run(
                {"id": name, "steps": [step]}, run_dir=Path(directory) / "run"
            )
            if not outcome.ok:
                raise SystemExit([s.error for s in outcome.steps.values() if s.error])
            out = outcome.outputs("hfe")
        entry = {
            "molecule": name,
            "smiles": smiles,
            "hydration_free_energy_kcal_mol": round(
                out["hydration_free_energy_kcal_mol"], 2
            ),
            "uncertainty_kcal_mol": None
            if out["uncertainty_kcal_mol"] is None
            else round(out["uncertainty_kcal_mol"], 2),
            "repeats_kcal_mol": [
                round(r["hydration_free_energy_kcal_mol"], 2) for r in out["repeats"]
            ],
            "minimum_adjacent_overlap": round(
                min(
                    leg["minimum_adjacent_overlap"]
                    for r in out["repeats"]
                    for leg in (r["solvent"], r["vacuum"])
                ),
                3,
            ),
            "seconds": round(out["seconds"]),
        }
        known = experiment.get(Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
        if known:
            entry["experiment_kcal_mol"], entry["experiment_uncertainty_kcal_mol"] = (
                known
            )
            entry["difference_kcal_mol"] = round(
                entry["hydration_free_energy_kcal_mol"] - known[0], 2
            )
        result["molecules"].append(entry)
        arguments.out.write_text(json.dumps(result, indent=2) + "\n")
        print(entry, flush=True)


if __name__ == "__main__":
    main()
