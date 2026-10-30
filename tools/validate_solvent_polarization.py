"""Compare the IEF-PCM polarization energy of `molecule.compute_quantum_properties` with
experimental hydration free energies.

Run it with the molecular-properties environment configured:

    ALAMBIQ_ENV_MOLECULAR_PROPERTIES=/path/to/python \
        python tools/validate_solvent_polarization.py --out solvent-polarization.json

Each molecule is optimized in water at B3LYP/6-31G*. The polarization energy is the energy in
the continuum minus the gas-phase energy at the same geometry. It contains the electrostatic
part of solvation only: the cavity and dispersion terms of a hydration free energy are not in
it. The experimental values are the hydration free energies of the FreeSolv database
(Mobley and Guthrie, J Comput Aided Mol Des 2014) at 298 K and 1 mol/L in both phases. The
table shows how far the polarization energy is from them, molecule by molecule.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from alambiq import api

# name, SMILES, experimental hydration free energy in kcal/mol
MOLECULES = [
    ("water", "O", -6.32),
    ("methanol", "CO", -5.11),
    ("ethanol", "CCO", -5.01),
    ("acetone", "CC(C)=O", -3.85),
    ("methane", "C", 1.99),
    ("benzene", "c1ccccc1", -0.86),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for name, smiles, experiment in MOLECULES:
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
