"""Compare the IEF-PCM polarization energy of `molecule.compute_quantum_properties` with
experimental hydration free energies.

Run it with the molecular-properties environment configured:

    ALAMBIQ_ENV_MOLECULAR_PROPERTIES=/path/to/python python tools/validate_solvent_polarization.py \
        --freesolv /path/to/database.txt --out solvent-polarization.json

Each molecule is optimized in water at B3LYP/6-31G*. The polarization energy is the energy in
the continuum minus the gas-phase energy at the same geometry. It contains the electrostatic
part of solvation only: the cavity and dispersion terms of a hydration free energy are not in
it. The experimental values are the hydration free energies of the FreeSolv database
(Mobley and Guthrie, J Comput Aided Mol Des 2014), read from a local copy of its
`database.txt` and found by canonical SMILES. The table shows how far the polarization energy is
from them, molecule by molecule.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from alambiq import api

MOLECULES = [
    ("methanol", "CO"),
    ("ethanol", "CCO"),
    ("acetone", "CC(C)=O"),
    ("phenol", "Oc1ccccc1"),
    ("methane", "C"),
    ("benzene", "c1ccccc1"),
]


def freesolv(path: Path) -> dict[str, float]:
    """The experimental hydration free energy of each molecule, by canonical SMILES."""
    from rdkit import Chem

    table = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = [f.strip() for f in line.split(";")]
        molecule = Chem.MolFromSmiles(fields[1])
        if molecule is not None:
            table[Chem.MolToSmiles(molecule)] = float(fields[3])
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--freesolv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    from rdkit import Chem

    experiments = freesolv(arguments.freesolv)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for name, smiles in MOLECULES:
            experiment = experiments[Chem.MolToSmiles(Chem.MolFromSmiles(smiles))]
            step = {
                "id": "properties",
                "capability": "molecule.compute_quantum_properties",
                "params": {
                    "smiles": smiles,
                    "solvent": "water",
                    "threads": arguments.threads,
                },
            }
            outcome = api.run(
                {"id": name, "steps": [step]}, run_dir=Path(directory) / name
            )
            if not outcome.ok:
                raise SystemExit([s.error for s in outcome.steps.values() if s.error])
            polarization = outcome.outputs("properties")[
                "solvent_polarization_energy_kcal_mol"
            ]
            rows.append(
                {
                    "molecule": name,
                    "smiles": smiles,
                    "polarization_kcal_mol": round(polarization, 2),
                    "experiment_kcal_mol": experiment,
                    "difference_kcal_mol": round(polarization - experiment, 2),
                }
            )
            print(f"{name:10s} {polarization:7.2f} {experiment:7.2f}", flush=True)
    result = {"functional": "b3lyp", "basis": "6-31G*", "molecules": rows}
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
