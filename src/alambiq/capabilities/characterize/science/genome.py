"""Offline, explicit protein-to-GFF3 neighborhood inspection.

Projects supplied GFF3 features and their exact protein mappings into bounded
genomic neighborhoods.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import unquote

from alambiq.capabilities.shared.sequences import parse_panel

DISPLAY_TYPES = {"CDS", "tRNA", "rRNA", "ncRNA", "tmRNA"}


def _decode(value):
    if re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise ValueError("Malformed GFF3 percent escape")
    result = unquote(value, errors="strict")
    if any(ord(c) < 32 for c in result):
        raise ValueError(
            "Control characters are not supported in GFF3 identifiers/attributes"
        )
    return result


def _attributes(text):
    result = {}
    if text == ".":
        return result
    for item in text.rstrip(";").split(";"):
        if "=" not in item:
            raise ValueError("GFF3 attributes require key=value")
        key, value = item.split("=", 1)
        key = _decode(key)
        if not key or key in result:
            raise ValueError("GFF3 attribute keys must be nonempty and unique per line")
        result[key] = [_decode(v) for v in value.split(",")]
        if key == "ID" and (len(result[key]) != 1 or not result[key][0]):
            raise ValueError("Each GFF3 ID must be one nonempty value")
    return result


def parse_gff(text):
    lines = text.splitlines()
    if not lines or not re.fullmatch(r"##gff-version 3(?:\.\d+){0,2}", lines[0]):
        raise ValueError("Supply GFF3 with its version directive on the first line")
    regions, features, named, circular, counts = {}, [], {}, set(), Counter()
    fasta_start = None
    for number, line in enumerate(lines[1:], 2):
        if line == "##FASTA":
            fasta_start = number
            break
        if line.startswith("##sequence-region"):
            parts = line.split()
            if len(parts) != 4:
                raise ValueError("Malformed sequence-region directive")
            seqid, a, b = _decode(parts[1]), int(parts[2]), int(parts[3])
            if seqid in regions or not 1 <= a <= b:
                raise ValueError("Duplicate or invalid sequence-region extent")
            regions[seqid] = {"start": a, "end": b, "basis": "sequence-region"}
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9:
            raise ValueError(f"GFF3 line {number} needs nine tab-separated columns")
        seqid, source, kind, a, b, score, strand, phase, raw = fields
        seqid = _decode(seqid)
        a, b = int(a), int(b)
        if not seqid or not 1 <= a <= b or strand not in {"+", "-", ".", "?"}:
            raise ValueError(f"Invalid genomic coordinates/strand at line {number}")
        if phase not in {"0", "1", "2", "."} or (kind == "CDS" and phase == "."):
            raise ValueError("CDS features need an explicit phase 0, 1 or 2")
        if score != "." and not math.isfinite(float(score)):
            raise ValueError("GFF3 scores must be finite or missing")
        attrs = _attributes(raw)
        if attrs.get("Is_circular") == ["true"]:
            circular.add(seqid)
        counts[kind] += 1
        identifier = attrs.get("ID", [None])[0]
        # Anonymous features use their original line, never a guessed protein ID.
        key = "id:" + identifier if identifier is not None else f"line:{number}"
        if key in named:
            f = named[key]
            if (f["seqid"], f["type"], f["strand"], f["source"]) != (
                seqid,
                kind,
                strand,
                source,
            ):
                raise ValueError(
                    "Repeated feature ID has inconsistent locus/type/strand/source"
                )
            for attr, values in attrs.items():
                if attr in f["attributes"] and f["attributes"][attr] != values:
                    raise ValueError("Repeated feature ID has inconsistent attributes")
                f["attributes"][attr] = values
            if any(not (b < s["start"] or a > s["end"]) for s in f["segments"]):
                raise ValueError("Overlapping or duplicate segments for one feature ID")
            f["segments"].append({"start": a, "end": b, "phase": phase})
            f["source_lines"].append(number)
        else:
            f = {
                "key": key,
                "id": identifier,
                "seqid": seqid,
                "type": kind,
                "source": source,
                "strand": strand,
                "attributes": attrs,
                "segments": [{"start": a, "end": b, "phase": phase}],
                "source_lines": [number],
            }
            named[key] = f
            features.append(f)
        if sum(counts.values()) > 10000:
            raise ValueError("Inspect at most 10000 GFF3 feature rows per request")
    if fasta_start is not None:
        lengths, identifier = {}, None
        for line in lines[fasta_start:]:
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:].strip()
                identifier = header.split()[0] if header else None
                if not identifier or identifier in lengths:
                    raise ValueError(
                        "Embedded FASTA identifiers must be nonempty and unique"
                    )
                lengths[identifier] = 0
            else:
                if identifier is None or not re.fullmatch(
                    r"[ACGTRYKMSWBDHVNacgtrykmswbdhvn]+", line
                ):
                    raise ValueError("Invalid embedded nucleotide FASTA")
                lengths[identifier] += len(line)
        if not lengths or any(n == 0 for n in lengths.values()):
            raise ValueError("Embedded FASTA contains no sequence")
        for seqid, length in lengths.items():
            region = regions.get(seqid)
            if region and (region["start"] != 1 or region["end"] != length):
                raise ValueError(
                    "Embedded sequence length disagrees with sequence-region"
                )
            regions[seqid] = {
                "start": 1,
                "end": length,
                "basis": "embedded-fasta",
                "sequence_length": length,
            }
    for f in features:
        f["segments"].sort(key=lambda s: s["start"])
        f["start"] = min(s["start"] for s in f["segments"])
        f["end"] = max(s["end"] for s in f["segments"])
        region = regions.get(f["seqid"])
        if region and (f["start"] < region["start"] or f["end"] > region["end"]):
            raise ValueError(
                "Feature outside supplied region; origin-spanning features need a separate profile"
            )
    return {
        "features": features,
        "regions": regions,
        "circular_contigs": sorted(circular),
        "source_type_counts": dict(sorted(counts.items())),
    }


def context_settings(parameters, *, has_candidates=False):
    if not set(parameters) <= {
        "genome_id",
        "anchor_ids",
        "flank_bp",
        "protein_id_attribute",
    }:
        raise ValueError("Unknown genomic-context parameter")
    genome = parameters.get("genome_id")
    if not isinstance(genome, str) or not re.fullmatch(
        r"[A-Za-z0-9_.:-]{1,128}", genome
    ):
        raise ValueError("Supply an explicit genome_id (1–128 identifier characters)")
    flank = parameters.get("flank_bp", 5000)
    if type(flank) is not int or not 0 <= flank <= 100000:
        raise ValueError("flank_bp must be an integer from 0 to 100000")
    attr = parameters.get("protein_id_attribute")
    if attr not in {"ID", "protein_id", "locus_tag"}:
        raise ValueError(
            "Choose the exact protein_id_attribute: ID, protein_id or locus_tag"
        )
    anchors = parameters.get("anchor_ids")
    if has_candidates:
        if anchors is not None:
            raise ValueError("Choose candidate FASTA or anchor_ids, not both")
    elif (
        not isinstance(anchors, list)
        or not 1 <= len(anchors) <= 50
        or any(
            not isinstance(a, str) or not a or any(c.isspace() for c in a)
            for a in anchors
        )
        or len(set(anchors)) != len(anchors)
    ):
        raise ValueError("Supply 1–50 unique anchor_ids, or a candidate FASTA binding")
    return {
        "genome_id": genome,
        "flank_bp": flank,
        "protein_id_attribute": attr,
        "anchor_ids": anchors,
    }


def inspect_context(gff, proteins, parameters, *, candidates=None):
    settings = context_settings(parameters, has_candidates=candidates is not None)
    parsed = parse_gff(gff)
    panel = {p["id"]: p for p in parse_panel(proteins)}
    candidate_panel = parse_panel(candidates) if candidates is not None else []
    anchors = (
        [p["id"] for p in candidate_panel]
        if candidates is not None
        else settings["anchor_ids"]
    )
    if not 1 <= len(anchors) <= 50:
        raise ValueError("Inspect 1–50 candidate proteins per request")
    sequences = {p["id"]: p["sequence"] for p in candidate_panel}
    loci, by_contig = {}, {}
    for feature in parsed["features"]:
        feature["protein_ids"] = (
            feature["attributes"].get(settings["protein_id_attribute"], [])
            if feature["type"] == "CDS"
            else []
        )
        feature["protein_ids"] = list(dict.fromkeys(feature["protein_ids"]))
        feature["available_protein_ids"] = [
            p for p in feature["protein_ids"] if p in panel
        ]
        for protein in feature["protein_ids"]:
            loci.setdefault(protein, []).append(feature)
        if feature["type"] in DISPLAY_TYPES:
            by_contig.setdefault(feature["seqid"], []).append(feature)
    anchor_records, windows, included = [], [], set()
    visible_count = 0
    for identifier in anchors:
        matches = loci.get(identifier, [])
        status = (
            "mapped" if len(matches) == 1 else "ambiguous" if matches else "unmapped"
        )
        if (
            identifier in sequences
            and identifier in panel
            and sequences[identifier] != panel[identifier]["sequence"]
        ):
            status = "sequence_mismatch"
        elif identifier in sequences and identifier not in panel and matches:
            status = "sequence_unavailable"
        anchor_records.append(
            {
                "id": identifier,
                "status": status,
                "sequence_available": identifier in panel,
                "locus_keys": [f["key"] for f in matches],
            }
        )
        if status in {"sequence_mismatch", "sequence_unavailable"}:
            continue
        for anchor in matches:
            region = parsed["regions"].get(anchor["seqid"])
            raw_start, raw_end = (
                anchor["start"] - settings["flank_bp"],
                anchor["end"] + settings["flank_bp"],
            )
            start = max(region["start"] if region else 1, raw_start)
            end = min(region["end"], raw_end) if region else raw_end
            visible = sorted(
                (
                    f
                    for f in by_contig.get(anchor["seqid"], [])
                    if any(
                        s["end"] >= start and s["start"] <= end for s in f["segments"]
                    )
                ),
                key=lambda f: (f["start"], f["end"], f["key"]),
            )
            if len(visible) > 250:
                raise ValueError(
                    "Neighborhood exceeds 250 features; reduce flank_bp"
                )
            visible_count += len(visible)
            if len(windows) >= 100 or visible_count > 5000:
                raise ValueError(
                    "Inspection exceeds 100 loci or 5000 displayed feature occurrences; select fewer anchors or a smaller supplied subset"
                )
            is_circular = anchor["seqid"] in parsed["circular_contigs"]
            windows.append(
                {
                    "anchor_id": identifier,
                    "anchor_key": anchor["key"],
                    "seqid": anchor["seqid"],
                    "start": start,
                    "end": end,
                    "region": region,
                    "circular": is_circular,
                    "left_clipped": raw_start < region["start"] if region else None,
                    "right_clipped": raw_end > region["end"] if region else None,
                    "boundary_note": "Circular source: a window that reaches the origin ends there and does not wrap."
                    if is_circular
                    else "Window clipped at the supplied region bounds, which may be assembly or subset boundaries."
                    if region
                    else "Source bounds are unreported; displayed extent follows the supplied features.",
                    "features": visible,
                }
            )
            included.update(p for f in visible for p in f["available_protein_ids"])
    return {
        "schema_version": "0.1",
        "genome_id": settings["genome_id"],
        "coordinate_basis": "One-based inclusive genomic coordinates",
        "settings": settings,
        "source_type_counts": parsed["source_type_counts"],
        "displayed_types": sorted(DISPLAY_TYPES),
        "protein_count": len(panel),
        "anchors": anchor_records,
        "neighbourhoods": windows,
        "neighbourhood_proteins": [p for p in panel.values() if p["id"] in included],
        "mapping_note": "Exact supplied CDS attribute values to FASTA identifiers; sequences are the supplied proteins.",
    }
