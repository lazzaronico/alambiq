"""Compose coding or protein parts with explicit encoding and checked translation."""

import hashlib
import json
import re

from alambiq.capabilities.shared.sequences import parse_fasta
from alambiq.capabilities.shared.sources import source_text


def settings(parameters):
    if set(parameters) != {"identifier", "stop_codon"}:
        raise ValueError(
            f"The parameters are {sorted(parameters)}; exactly identifier and "
            "stop_codon are required."
        )
    if not isinstance(parameters["identifier"], str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", parameters["identifier"]
    ):
        raise ValueError(
            "identifier must be a FASTA identifier of 1 to 64 characters: letters, digits, "
            "underscores, periods and hyphens, starting with a letter or digit."
        )
    if parameters["stop_codon"] not in ("TAA", "TAG", "TGA"):
        raise ValueError(
            "stop_codon must be TAA, TAG or TGA, the stop codons of genetic code 11; "
            f"found {parameters['stop_codon']!r}."
        )
    return dict(parameters)


def read_parts(parts, protein_only=False):
    minimum = 1 if protein_only else 2
    if not isinstance(parts, (list, tuple)) or not minimum <= len(parts) <= 20:
        raise ValueError(
            f"parts must be a list of {minimum} to 20 ordered parts; "
            f"found {len(parts) if isinstance(parts, (list, tuple)) else type(parts).__name__}."
        )
    required = {"id", "role", "expected_protein"}
    if not protein_only:
        required.add("coding_dna")
    allowed = required if protein_only else required | {"remove_terminal_stop"}
    seen, result = set(), []
    for part in parts:
        if (
            not isinstance(part, dict)
            or not required <= set(part)
            or set(part) - allowed
        ):
            raise ValueError(
                f"Each part must have the fields {sorted(required)} and may have only "
                f"the fields {sorted(allowed)}."
            )
        if (
            not isinstance(part["id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", part["id"])
            or part["id"] in seen
        ):
            raise ValueError(
                "Part identifiers must be unique and consist of 1 to 64 characters: "
                "letters, digits, underscores, periods and hyphens, starting with a "
                "letter or digit."
            )
        seen.add(part["id"])
        if (
            part["role"] not in ("protein", "tag", "linker")
            or type(part.get("remove_terminal_stop", False)) is not bool
        ):
            raise ValueError(
                f"Part {part['id']}: role must be protein, tag or linker, and "
                "remove_terminal_stop must be true or false."
            )
        keys = (
            ("expected_protein",)
            if protein_only
            else ("coding_dna", "expected_protein")
        )
        texts = {key: source_text(part[key]) for key in keys}
        protein = parse_fasta(texts["expected_protein"])[1]
        dna = None if protein_only else parse_fasta(texts["coding_dna"])[1]
        if dna is not None and (
            not re.fullmatch("[ACGT]+", dna) or len(dna) % 3 or len(dna) > 6000
        ):
            raise ValueError(
                f"Part {part['id']} must have coding_dna of whole codons, each base A, C, "
                "G or T, with at most 6,000 bases."
            )
        if not re.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+", protein):
            raise ValueError(
                f"Part {part['id']} must have an expected_protein of the 20 standard "
                "amino acids only."
            )
        result.append((part, texts, dna, protein))
    return result


def read_codons(spec):
    """Read explicit choices; validate their genetic meaning with Biopython at execution."""

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"The codon choices list the amino acid {key} twice.")
            result[key] = value
        return result

    text = source_text(spec)
    choices = json.loads(text, object_pairs_hook=unique)
    if (
        not isinstance(choices, dict)
        or set(choices) != set("ACDEFGHIKLMNPQRSTVWY")
        or any(
            not isinstance(c, str) or not re.fullmatch("[ACGT]{3}", c)
            for c in choices.values()
        )
    ):
        raise ValueError(
            "The codon choices must be a JSON object with exactly one codon of 3 uppercase "
            "A, C, G or T bases for each of the 20 standard amino acids."
        )
    return text, choices


def runtime_version():
    try:
        import Bio
    except ImportError as exc:
        raise ValueError(
            "Install Biopython 1.88, which the construct and coding-design extras include"
        ) from exc
    if Bio.__version__ != "1.88":
        raise ValueError(
            f"construct.compose_coding requires Biopython 1.88; found {Bio.__version__}."
        )
    return Bio.__version__


def compose(parts, parameters, *, codon_choices=None):
    config = settings(parameters)
    parsed = read_parts(parts, protein_only=codon_choices is not None)
    version = runtime_version()
    from Bio.Data.CodonTable import TranslationError, unambiguous_dna_by_id
    from Bio.Seq import Seq

    encoding = None
    if codon_choices is not None:
        text, choices = read_codons(codon_choices)
        table = unambiguous_dna_by_id[11]
        if any(table.forward_table.get(codon) != aa for aa, codon in choices.items()):
            raise ValueError(
                "Every codon choice must encode its amino acid under genetic code 11."
            )
        encoding = {
            "method": "explicit-one-codon-per-amino-acid",
            "codons": choices,
            "source": {
                **{k: v for k, v in codon_choices.items() if k not in ("path", "text")},
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            },
        }

    dna, protein, records = "", "", []
    for index, (part, texts, coding, expected) in enumerate(parsed):
        if encoding is not None:
            coding = "".join(encoding["codons"][aa] for aa in expected)
        removed = None
        if part.get("remove_terminal_stop", False):
            if len(coding) <= 3 or coding[-3:] not in ("TAA", "TAG", "TGA"):
                raise ValueError(
                    f"Part {part['id']} has remove_terminal_stop set but does not end "
                    "in TAA, TAG or TGA preceded by at least 1 codon."
                )
            coding, removed = coding[:-3], coding[-3:]
        try:
            # Only the first codon of the entire fusion has initiator semantics.
            translated = (
                str(Seq(coding + config["stop_codon"]).translate(table=11, cds=True))
                if index == 0
                else str(Seq(coding).translate(table=11))
            )
        except TranslationError as exc:
            raise ValueError(
                f"Part {part['id']} does not translate under genetic code 11: {exc}"
            ) from exc
        if translated != expected:
            raise ValueError(
                f"The translation of part {part['id']} in the fusion differs from its "
                "expected_protein."
            )
        records.append(
            {
                "id": part["id"],
                "role": part["role"],
                "dna_start": len(dna) + 1,
                "dna_end": len(dna) + len(coding),
                "protein_start": len(protein) + 1,
                "protein_end": len(protein) + len(expected),
                "removed_stop_codon": removed,
                "sources": {
                    key: {
                        **{
                            k: v
                            for k, v in part[key].items()
                            if k not in ("path", "text")
                        },
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                    for key, text in texts.items()
                },
            }
        )
        dna += coding
        protein += expected
    dna += config["stop_codon"]
    if not 60 <= len(dna) <= 6000:
        raise ValueError(
            f"The composed insert must be 60 to 6,000 bp; found {len(dna)} bp."
        )
    if str(Seq(dna).translate(table=11, cds=True)) != protein:
        raise ValueError(
            "The translation of the composed insert differs from the concatenated "
            "expected proteins."
        )
    result = {
        "schema_version": "0.1",
        "identifier": config["identifier"],
        "genetic_code": 11,
        "versions": {"biopython": version},
        "dna": dna,
        "protein": protein,
        "parts": records,
        "stop_codon": config["stop_codon"],
        "coordinate_convention": "one-based-inclusive",
    }
    if encoding is not None:
        result["encoding"] = encoding
    return result
