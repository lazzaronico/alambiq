"""Synonymous recoding of a coding sequence under explicit constraints.

The solver is DNA Chisel. The translation under genetic code 11, the first and
last codons, every avoided motif and the GC bounds are enforced as
constraints, and then checked again here on the solver's output before the
sequence is reported, so a solver that silently violated one is caught.

The seed makes a run reproducible. DNA Chisel draws from NumPy's global random
state, so `design` seeds it and restores the caller's state afterwards; that
preserves the determinism the dedicated runner process used to provide without
disturbing anything else in this process.
"""

from __future__ import annotations

import importlib.metadata
import math
import re


def settings(parameters):
    value = {
        "avoid_patterns": [],
        "gc_min": 0.0,
        "gc_max": 1.0,
        "gc_window": None,
        **parameters,
    }
    if set(value) != {"seed", "avoid_patterns", "gc_min", "gc_max", "gc_window"}:
        raise ValueError(
            f"The parameters are {sorted(value)}; exactly seed, avoid_patterns, gc_min, "
            "gc_max and gc_window are accepted, and seed is required."
        )
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError(
            f"seed must be an integer from 0 to 4,294,967,295; found {value['seed']!r}."
        )
    patterns = value["avoid_patterns"]
    if (
        not isinstance(patterns, list)
        or len(patterns) > 20
        or any(
            not isinstance(p, str) or not re.fullmatch("[ACGT]{4,30}", p)
            for p in patterns
        )
        or len(set(patterns)) != len(patterns)
    ):
        raise ValueError(
            "avoid_patterns must be a list of at most 20 distinct motifs of 4 to 30 "
            "bases, each base A, C, G or T."
        )
    for key in ["gc_min", "gc_max"]:
        if (
            type(value[key]) not in (int, float)
            or not math.isfinite(value[key])
            or not 0 <= value[key] <= 1
        ):
            raise ValueError(
                f"{key} must be a finite fraction from 0 to 1; found {value[key]!r}."
            )
    if value["gc_min"] > value["gc_max"]:
        raise ValueError(
            f"gc_min must not exceed gc_max; found {value['gc_min']} and "
            f"{value['gc_max']}."
        )
    window = value["gc_window"]
    if window is not None and (type(window) is not int or not 3 <= window <= 6000):
        raise ValueError(
            f"gc_window must be null or an integer from 3 to 6,000 bases; found {window!r}."
        )
    return value


def validate_input(dna, protein, parameters):
    config = settings(parameters)
    if (
        not isinstance(dna, str)
        or not re.fullmatch("[ACGT]{60,6000}", dna)
        or len(dna) % 3
    ):
        raise ValueError(
            "The coding sequence must have 60 to 6,000 bases in whole codons, each base "
            "A, C, G or T."
        )
    if not isinstance(protein, str) or not re.fullmatch(
        "[ACDEFGHIKLMNPQRSTVWY]+", protein
    ):
        raise ValueError(
            "The expected protein must consist of the 20 standard amino acids only."
        )
    if len(dna) != 3 * (len(protein) + 1):
        raise ValueError(
            "The coding sequence must have 3 bases for each residue of the expected "
            f"protein and 3 for the terminal stop codon, {3 * (len(protein) + 1)} bases; "
            f"found {len(dna)}."
        )
    if config["gc_window"] is not None and config["gc_window"] > len(dna):
        raise ValueError(
            f"gc_window ({config['gc_window']} bases) exceeds the coding sequence "
            f"length ({len(dna)} bases)."
        )
    return config


def checks(dna, config):
    complement = str.maketrans("ACGT", "TGCA")
    matches = []
    for pattern in config["avoid_patterns"]:
        reverse = pattern.translate(complement)[::-1]
        for motif, strand in [(pattern, "+")] + (
            [(reverse, "-")] if reverse != pattern else []
        ):
            for i in range(len(dna) - len(motif) + 1):
                if dna.startswith(motif, i):
                    matches.append(
                        {
                            "pattern": pattern,
                            "start": i + 1,
                            "end": i + len(motif),
                            "strand": strand,
                        }
                    )
    window = config["gc_window"] or len(dna)
    windows = []
    values = []
    for i in range(len(dna) - window + 1):
        part = dna[i : i + window]
        gc = (part.count("G") + part.count("C")) / window
        values.append(gc)
        if not config["gc_min"] <= gc <= config["gc_max"]:
            windows.append({"start": i + 1, "end": i + window, "gc_fraction": gc})
    return {
        "motif_matches": matches,
        "gc_violations": windows,
        "gc_fraction": (dna.count("G") + dna.count("C")) / len(dna),
        "minimum_window_gc": min(values),
        "maximum_window_gc": max(values),
        "satisfied": not matches and not windows,
    }


def design(dna, protein, parameters):
    config = validate_input(dna, protein, parameters)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["dnachisel", "biopython", "numpy"]
    }
    import numpy as np
    from Bio.Data.CodonTable import TranslationError
    from Bio.Seq import Seq
    from dnachisel import (
        AvoidChanges,
        AvoidPattern,
        DnaOptimizationProblem,
        EnforceGCContent,
        EnforceTranslation,
    )

    try:
        translation = str(Seq(dna).translate(table=11, cds=True))
    except TranslationError as exc:
        raise ValueError(
            f"The coding sequence does not translate as a complete CDS under genetic code 11: {exc}"
        ) from exc
    if translation != protein:
        raise ValueError(
            "The coding sequence does not translate to expected_protein under genetic "
            "code 11."
        )
    constraints = [
        EnforceTranslation(genetic_table="Bacterial"),
        AvoidChanges(location=(0, 3)),
        AvoidChanges(location=(len(dna) - 3, len(dna))),
    ]
    constraints += [
        AvoidPattern(pattern, strand=0) for pattern in config["avoid_patterns"]
    ]
    constraints.append(
        EnforceGCContent(
            mini=config["gc_min"], maxi=config["gc_max"], window=config["gc_window"]
        )
    )
    entry_state = np.random.get_state()
    np.random.seed(config["seed"])
    try:
        problem = DnaOptimizationProblem(dna, constraints=constraints, logger=None)
        return _solve(problem, dna, protein, config, versions)
    finally:
        np.random.set_state(entry_state)


def _solve(problem, dna, protein, config, versions):
    """Resolve the constraints and check the solver's output independently."""
    from Bio.Seq import Seq
    from dnachisel import NoSolutionError

    result = {
        "schema_version": "0.1",
        "dna_before": dna,
        "protein": protein,
        "settings": config,
        "versions": versions,
        "genetic_code": 11,
        "terminal_codons_required": "unchanged",
        "before": checks(dna, config),
    }
    try:
        problem.resolve_constraints()
    except NoSolutionError as exc:
        # Retain a rejected intermediate, never expose it as an assembly input.
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "rejected_candidate": problem.sequence,
                "after": checks(problem.sequence, config),
            }
        )
        return result
    after = problem.sequence
    if (
        len(after) != len(dna)
        or after[:3] != dna[:3]
        or after[-3:] != dna[-3:]
        or str(Seq(after).translate(table=11, cds=True)) != protein
        or not checks(after, config)["satisfied"]
    ):
        raise ValueError(
            "The independent checks of the translation, the terminal codons, the motifs "
            "and the GC content rejected the solver output."
        )
    result.update(
        {
            "status": "completed",
            "dna_after": after,
            "after": checks(after, config),
            "changed_codons": [
                {
                    "residue": i // 3 + 1,
                    "before": dna[i : i + 3],
                    "after": after[i : i + 3],
                    "amino_acid": protein[i // 3],
                }
                for i in range(0, len(dna) - 3, 3)
                if dna[i : i + 3] != after[i : i + 3]
            ],
            "solver_summary": problem.constraints_text_summary(),
        }
    )
    return result
