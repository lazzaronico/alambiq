"""Protonate a protein at a pH with PROPKA and PDB2PQR.

Runs inside the protonation environment (PDB2PQR 3.7 with PROPKA 3.5). The input is the
first model of a PDB file, reduced to the atoms of one conformation per residue and to the
amino acids. PROPKA predicts a pKa for each titratable group. PDB2PQR protonates each group
at the pH, adds the hydrogens, and optimizes the hydrogen-bond network, which can turn the
amide of asparagine and glutamine and the ring of histidine by 180 degrees. The driver
records which residues the optimizer turned.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

WATER = {"HOH", "WAT", "DOD", "TIP", "SOL"}
STANDARD = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}  # fmt: skip
SUMMARY_ROW = re.compile(
    r"^\s+(?P<group>[A-Z][A-Z0-9+-]*)\s+(?P<number>-?\d+)\s+(?P<chain>\S)\s+"
    r"(?P<pka>-?\d+\.\d+)\s+(?P<model>-?\d+\.\d+)\s*$"
)
TERMINI = {"N+": "N-terminus", "C-": "C-terminus"}
MOVED_ANGSTROM = 0.05


def _is_hydrogen(line: str) -> bool:
    element = line[76:78].strip().upper()
    if element:
        return element in ("H", "D")
    return line[12:16].strip().upper().startswith(("H", "D"))


def read_input(path: Path) -> dict:
    """The lines to hand on, and what was left out.

    The first model is read. Of the alternate locations of a residue, the first one named
    is kept. Water, hydrogen atoms and heteroatoms are dropped; residues of non-standard
    amino acids in HETATM records are listed.
    """
    kept: list[str] = []
    hetero: dict[str, int] = {}
    waters = hydrogens = alternates = 0
    chosen: dict[tuple, str] = {}
    insertion_codes: set[str] = set()
    for line in path.read_text().splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        name = line[17:20].strip()
        if record == "HETATM":
            if name in WATER:
                waters += 1
            else:
                hetero[name] = hetero.get(name, 0) + 1
            continue
        if _is_hydrogen(line):
            hydrogens += 1
            continue
        if line[26].strip():
            insertion_codes.add(f"{line[21]}:{name}{line[22:26].strip()}{line[26]}")
        altloc = line[16]
        if altloc.strip():
            key = (line[21], line[22:27], line[12:16])
            first = chosen.setdefault(key, altloc)
            if first != altloc:
                alternates += 1
                continue
            line = line[:16] + " " + line[17:]
        kept.append(line)
    return {
        "lines": kept,
        "heteroatoms": hetero,
        "waters": waters,
        "hydrogens": hydrogens,
        "alternates": alternates,
        "insertion_codes": sorted(insertion_codes),
    }


def heavy_atoms(lines: list[str]) -> dict[tuple, tuple]:
    """Heavy atoms by (chain, number, atom name) with coordinates and residue name."""
    atoms = {}
    for line in lines:
        if line[:6].strip() not in ("ATOM", "HETATM") or _is_hydrogen(line):
            continue
        key = (line[21], int(line[22:26]), line[12:16].strip())
        coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        atoms[key] = (coords, line[17:20].strip())
    return atoms


FLIP_DRIVER = """
import json, os, sys
from pdb2pqr import main
from pdb2pqr.hydrogens import structures

flips = {}
original = structures.Flip.fix_flip


def record(self, bondatom):
    original(self, bondatom)
    residue = bondatom.residue
    flips[f"{residue.chain_id}|{residue.res_seq}|{residue.name}"] = bool(residue.wasFlipped)


structures.Flip.fix_flip = record
sys.argv = ["pdb2pqr"] + sys.argv[1:]
try:
    main.main()
finally:
    with open(os.environ["ALAMBIQ_FLIPS"], "w") as handle:
        json.dump(flips, handle)
"""


def turned_residues(flips: dict[str, bool]) -> list[dict]:
    """The residues PDB2PQR turned by 180 degrees, from the flags its optimizer recorded.

    The optimizer judges each Asn, Gln and His that can turn; the driver above records
    the final state of each residue it judged.
    """
    turned = []
    for key, flipped in flips.items():
        if flipped:
            chain, number, name = key.split("|")
            turned.append({"chain": chain, "number": int(number), "name": name})
    return sorted(turned, key=lambda t: (t["chain"], t["number"]))


def added_heavy_atoms(before: dict, after: dict) -> list[dict]:
    """Heavy atoms the output has and the input lacked, by residue."""
    added: dict[tuple, list[str]] = {}
    for (chain, number, atom), (_, name) in after.items():
        if (chain, number, atom) not in before:
            added.setdefault((chain, number, name), []).append(atom)
    return [
        {"chain": chain, "number": number, "name": name, "atoms": sorted(atoms)}
        for (chain, number, name), atoms in sorted(added.items())
    ]


def parse_titration(log: str) -> list[dict]:
    """The rows of the PROPKA summary: group, residue, pKa and the model pKa."""
    rows = []
    inside = False
    for line in log.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            inside = True
            continue
        if not inside:
            continue
        found = SUMMARY_ROW.match(line)
        if found:
            rows.append(
                {
                    "group": found["group"],
                    "chain": found["chain"],
                    "number": int(found["number"]),
                    "pka": float(found["pka"]),
                    "model_pka": float(found["model"]),
                }
            )
        elif rows and not line.strip():
            break
    return rows


def residue_names(lines: list[str]) -> dict[tuple, str]:
    names = {}
    for line in lines:
        if line[:6].strip() == "ATOM":
            names[(line[21], int(line[22:26]))] = line[17:20].strip()
    return names


def titration_states(rows: list[dict], names_in: dict, names_out: dict, ph: float) -> list[dict]:
    """Each titratable group with its pKa and the residue name it was given."""
    states = []
    for row in rows:
        key = (row["chain"], row["number"])
        states.append(
            {
                **row,
                "name_in": names_in.get(key, ""),
                "name_out": names_out.get(key, ""),
                "protonated_at_ph": row["pka"] > ph,
            }
        )
    return states


def run(cfg: dict) -> dict:
    """Protonate the structure and report the titration states and the turned residues."""
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    source = read_input(Path(cfg["structure"]))
    if source["insertion_codes"]:
        raise ValueError(
            "Residues with insertion codes are not protonated here: "
            + ", ".join(source["insertion_codes"][:5])
            + ". Renumber the chain first."
        )
    if not source["lines"]:
        raise ValueError("The first model of the structure has no amino acid atoms.")
    cleaned = output / "input.pdb"
    cleaned.write_text("\n".join(source["lines"]) + "\nEND\n")
    command = [sys.executable, "-c", FLIP_DRIVER]
    command += [
        "--ff=AMBER",
        "--ffout=AMBER",
        f"--with-ph={cfg['ph']}",
        "--titration-state-method=propka",
        "--keep-chain",
        "--drop-water",
        "--pdb-output",
        str(output / "protonated.pdb"),
    ]
    if not cfg["optimize_hydrogen_network"]:
        command.append("--noopt")
    command += [str(cleaned), str(output / "protonated.pqr")]
    flips_path = output / "flips.json"
    environment = {**os.environ, "ALAMBIQ_FLIPS": str(flips_path)}
    done = subprocess.run(
        command, capture_output=True, text=True, timeout=cfg["timeout"], env=environment
    )
    log = done.stdout + "\n" + done.stderr
    (output / "pdb2pqr.log").write_text(log)
    if done.returncode != 0 or not (output / "protonated.pdb").is_file():
        tail = "\n".join(log.strip().splitlines()[-8:])
        raise RuntimeError(f"PDB2PQR failed with exit code {done.returncode}:\n{tail}")
    result_lines = (output / "protonated.pdb").read_text().splitlines()
    atom_lines = [line for line in result_lines if line[:6].strip() == "ATOM"]
    before = heavy_atoms(source["lines"])
    after = heavy_atoms(atom_lines)
    rows = parse_titration(log)
    hydrogens_out = sum(1 for line in atom_lines if _is_hydrogen(line))
    return {
        "atoms": len(atom_lines),
        "hydrogens": hydrogens_out,
        "titration": titration_states(
            rows, residue_names(source["lines"]), residue_names(atom_lines), cfg["ph"]
        ),
        "turned": turned_residues(json.loads(flips_path.read_text())),
        "added_heavy_atoms": added_heavy_atoms(before, after),
        "dropped": {
            "heteroatoms": source["heteroatoms"],
            "waters": source["waters"],
            "input_hydrogens": source["hydrogens"],
            "alternate_location_atoms": source["alternates"],
        },
        "files": {"protonated": "protonated.pdb", "pqr": "protonated.pqr", "log": "pdb2pqr.log"},
        "versions": versions(),
    }


def versions() -> dict[str, str]:
    from importlib.metadata import version

    return {name: version(name) for name in ("pdb2pqr", "propka")}
