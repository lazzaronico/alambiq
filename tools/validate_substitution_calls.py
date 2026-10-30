"""Check that `protein.call_substitutions` recovers edits planted in Swiss-Prot proteins.

Run it with the mmseqs environment configured and the Swiss-Prot resource installed:

    ALAMBIQ_ENV_MMSEQS=/path/to/python python tools/validate_substitution_calls.py \
        --fasta uniprot_sprot.fasta --out substitution-calls.json

`uniprot_sprot.fasta` is the file the resource was built from (see the `build` field of
`alambiq resource show swissprot-2026_03`).

A random sample of Swiss-Prot proteins (50 to 1,000 residues of the 20 amino acids) receives
planted edits, in three groups of equal size: 1 to 5 random substitutions; an insertion of 2
residues; a deletion of 3 residues. Each edited protein is called against the database. The
call is scored on three counts per group:

* the closest reference has the sequence of the original entry (an identical entry counts),
  or is another entry with as few differences from the variant as the original has;
* the aligned region of the variant is reproduced exactly by applying the called
  differences to the aligned region of the reference;
* residues are left outside the alignment at an end of the variant or of the reference.

For the substitution group the exact set of planted substitutions must also be called when no
residues are left outside the alignment. The database holds the original entry, so the test
measures the alignment and the reading of its differences, not the choice of a reference among
relatives.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from alambiq import api

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def read_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            parts = line[1:].split()[0].split("|")
            name = parts[1] if len(parts) == 3 else parts[0]
            sequences[name] = ""
        elif name:
            sequences[name] += line.strip()
    return sequences


def edit(sequence: str, kind: str, rng: random.Random):
    """The edited sequence and the planted substitutions (position, reference, variant)."""
    if kind == "substitutions":
        positions = sorted(rng.sample(range(1, len(sequence) + 1), rng.randint(1, 5)))
        planted = []
        residues = list(sequence)
        for position in positions:
            new = rng.choice([a for a in AMINO if a != sequence[position - 1]])
            planted.append((position, sequence[position - 1], new))
            residues[position - 1] = new
        return "".join(residues), planted
    place = rng.randint(1, len(sequence) - 5)
    if kind == "insertion":
        inserted = "".join(rng.choices(AMINO, k=2))
        return sequence[:place] + inserted + sequence[place:], []
    return sequence[:place] + sequence[place + 3 :], []


def rebuild(region: str, offset: int, call: dict) -> str:
    """The variant region implied by a reference region and the called differences.

    `offset` is the number of reference residues before the region. The differences do not
    overlap, so applying them from the end of the region backwards keeps the indices valid.
    """
    operations = [(s["position"], "substitution", s) for s in call["substitutions"]]
    operations += [(d["start"], "deletion", d) for d in call["deletions"]]
    operations += [
        (i["after_position"] + 0.5, "insertion", i) for i in call["insertions"]
    ]
    residues = list(region)
    for _, kind, item in sorted(operations, key=lambda op: -op[0]):
        if kind == "substitution":
            residues[item["position"] - 1 - offset] = item["variant"]
        elif kind == "deletion":
            del residues[item["start"] - 1 - offset : item["end"] - offset]
        else:
            index = item["after_position"] - offset
            residues[index:index] = list(item["residues"])
    return "".join(residues)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    rng = random.Random(arguments.seed)
    sequences = {
        a: s
        for a, s in read_fasta(arguments.fasta).items()
        if 50 <= len(s) <= 1000 and set(s) <= set(AMINO)
    }
    chosen = rng.sample(sorted(sequences), arguments.sample)
    kinds = ["substitutions", "insertion", "deletion"]
    cases = {}
    for index, accession in enumerate(chosen):
        kind = kinds[index % 3]
        variant, planted = edit(sequences[accession], kind, rng)
        cases[f"v{index}"] = (accession, kind, variant, planted)

    with tempfile.TemporaryDirectory() as directory:
        query = Path(directory) / "variants.faa"
        query.write_text("".join(f">{k}\n{v[2]}\n" for k, v in cases.items()))
        step = {
            "id": "call",
            "capability": "protein.call_substitutions",
            "params": {
                "variants": str(query),
                "database": "swissprot-2026_03",
                "threads": arguments.threads,
            },
        }
        outcome = api.run(
            {"id": "validate", "steps": [step]}, run_dir=Path(directory) / "run"
        )
        if not outcome.ok:
            raise SystemExit([s.error for s in outcome.steps.values() if s.error])
        calls = {c["id"]: c for c in outcome.outputs("call")["variants"]}

    groups = {
        k: {
            "n": 0,
            "reference": 0,
            "tied": 0,
            "region": 0,
            "ends_outside": 0,
            "exact": 0,
        }
        for k in kinds
    }
    failures = []
    for name, (accession, kind, variant, planted) in cases.items():
        group, call = groups[kind], calls[name]
        group["n"] += 1
        found = call["reference"]
        # an entry with the sequence of the original is as good a reference as the original
        right = found is not None and sequences[found] == sequences[accession]
        group["reference"] += right
        planted_size = {"substitutions": len(planted), "insertion": 2, "deletion": 3}[
            kind
        ]
        if found is not None and not right and call["differences"] == planted_size:
            group["tied"] += 1  # an entry as close to the variant as the original is
        outside = (
            call["unaligned_variant_start"]
            + call["unaligned_variant_end"]
            + call["unaligned_reference_start"]
            + call["unaligned_reference_end"]
        )
        group["ends_outside"] += outside > 0
        region = False
        if right:
            reference = sequences[found]
            start, end = (
                call["unaligned_reference_start"],
                call["unaligned_reference_end"],
            )
            expected = variant[
                call["unaligned_variant_start"] : len(variant)
                - call["unaligned_variant_end"]
            ]
            region = (
                rebuild(reference[start : len(reference) - end], start, call)
                == expected
            )
        group["region"] += region
        exact = False
        if kind == "substitutions" and right and region and outside == 0:
            called = {
                (s["position"], s["reference"], s["variant"])
                for s in call["substitutions"]
            }
            exact = (
                called == set(planted)
                and not call["insertions"]
                and not call["deletions"]
            )
        group["exact"] += exact
        if not right or not region:
            failures.append(
                {
                    "kind": kind,
                    "original": accession,
                    "found": found,
                    "planted": [list(p) for p in planted],
                    "notation": call["notation"],
                    "unaligned_variant": [
                        call["unaligned_variant_start"],
                        call["unaligned_variant_end"],
                    ],
                    "unaligned_reference": [
                        call["unaligned_reference_start"],
                        call["unaligned_reference_end"],
                    ],
                }
            )
    result = {
        "sample": len(cases),
        "seed": arguments.seed,
        "groups": groups,
        "failures": failures,
    }
    arguments.out.write_text(json.dumps(result, indent=2) + "\n")
    for kind, g in groups.items():
        print(
            f"{kind:14s} n={g['n']}  reference {g['reference']}  equally close {g['tied']}  aligned region {g['region']}  "
            f"ends outside {g['ends_outside']}  exact substitutions {g['exact']}"
        )


if __name__ == "__main__":
    main()
