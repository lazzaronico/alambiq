"""One explicit blunt-linearized vector plus CDS or annotated cassette, using Pydna."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re

from alambiq.capabilities.shared.sequences import parse_fasta


def settings(parameters):
    values = {
        "enzyme": "SmaI",
        "overlap": 35,
        "primer_min": 18,
        "target_tm": 60.0,
        **parameters,
    }
    if set(values) != {"enzyme", "overlap", "primer_min", "target_tm"}:
        unknown = sorted(set(values) - {"enzyme", "overlap", "primer_min", "target_tm"})
        raise ValueError(
            f"The parameters include {', '.join(unknown)}; the accepted parameters are "
            "enzyme, overlap, primer_min and target_tm."
        )
    if not isinstance(values["enzyme"], str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9]{1,20}", values["enzyme"]
    ):
        raise ValueError(
            "enzyme must be the name of a blunt-cutting restriction enzyme: "
            "2 to 21 letters and digits, starting with a letter."
        )
    for name, low, high in [("overlap", 25, 60), ("primer_min", 18, 35)]:
        if type(values[name]) is not int or not low <= values[name] <= high:
            raise ValueError(
                f"{name} must be an integer from {low} to {high} bases; "
                f"found {values[name]!r}."
            )
    if (
        type(values["target_tm"]) not in (float, int)
        or not math.isfinite(values["target_tm"])
        or not 45 <= values["target_tm"] <= 75
    ):
        raise ValueError(
            "target_tm must be a finite temperature from 45 to 75 °C; "
            f"found {values['target_tm']!r}."
        )
    return values


def runtime_versions():
    try:
        import Bio
        import pydna
    except ImportError as exc:
        raise ValueError(
            "Install the construct extra: pydna 5.5.16 and biopython 1.88"
        ) from exc
    if (pydna.__version__, Bio.__version__) != ("5.5.16", "1.88"):
        raise ValueError(
            "construct.assemble requires pydna 5.5.16 and biopython 1.88; found "
            f"pydna {pydna.__version__} and biopython {Bio.__version__}."
        )
    return {"pydna": pydna.__version__, "biopython": Bio.__version__}


def feature_record(feature):
    return {
        "type": feature.type,
        "parts": [
            {"start": int(part.start) + 1, "end": int(part.end), "strand": part.strand}
            for part in feature.location.parts
        ],
        "qualifiers": feature.qualifiers,
    }


def cassette_cds(record):
    """Validate a supplied single-CDS cassette; the feature roles are the supplied labels."""
    if (
        record.annotations.get("topology") != "linear"
        or not 1 <= len(record.features) <= 30
    ):
        raise ValueError(
            "The cassette must be a linear GenBank record with 1 to 30 features; found "
            f"topology {record.annotations.get('topology')} and {len(record.features)} features."
        )
    for feature in record.features:
        if (
            feature.location is None
            or len(feature.location.parts) != 1
            or any(
                type(p.start).__name__ != "ExactPosition"
                or type(p.end).__name__ != "ExactPosition"
                or p.ref is not None
                or not 0 <= int(p.start) < int(p.end) <= len(record)
                for p in feature.location.parts
            )
            or {"bf_source_feature", "bf_insert_feature"} & feature.qualifiers.keys()
        ):
            raise ValueError(
                "Each cassette feature must have one exact location inside the cassette "
                "and no qualifier named bf_source_feature or bf_insert_feature."
            )
    coding = [f for f in record.features if f.type == "CDS"]
    if len(coding) != 1:
        raise ValueError(
            f"The cassette must contain exactly 1 CDS feature; found {len(coding)}."
        )
    cds = coding[0]
    if (
        cds.location.strand not in (-1, 1)
        or cds.qualifiers.get("transl_table") not in (["1"], ["11"])
        or cds.qualifiers.get("codon_start", ["1"]) != ["1"]
        or {
            "transl_except",
            "exception",
            "ribosomal_slippage",
            "partial",
            "pseudo",
            "pseudogene",
        }
        & cds.qualifiers.keys()
    ):
        raise ValueError(
            "The cassette CDS must lie on strand +1 or -1, have transl_table 1 or 11 and "
            "codon_start 1, and have none of the qualifiers transl_except, exception, "
            "ribosomal_slippage, partial, pseudo and pseudogene."
        )
    return cds, int(cds.qualifiers["transl_table"][0])


def assemble_construct(
    vector_text,
    insert_text,
    protein_text,
    parameters,
    insert_features=(),
    *,
    insert_format="fasta",
):
    versions = runtime_versions()
    config = settings(parameters)
    from Bio import SeqIO
    from Bio.Restriction import RestrictionBatch
    from Bio.Seq import Seq
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from pydna.amplify import pcr
    from pydna.assembly2 import Assembly, gibson_overlap
    from pydna.design import assembly_fragments, primer_design
    from pydna.dseqrecord import Dseqrecord
    from pydna.tm import tm_default

    vector_record = SeqIO.read(io.StringIO(vector_text), "genbank")
    if (
        vector_record.annotations.get("topology") != "circular"
        or not 100 <= len(vector_record) <= 20000
        or not re.fullmatch("[ACGT]+", str(vector_record.seq).upper())
    ):
        raise ValueError(
            "The vector must be one circular GenBank record of 100 to 20,000 bases, each "
            f"A, C, G or T; found topology {vector_record.annotations.get('topology')} "
            f"and {len(vector_record)} bases."
        )
    if insert_format not in ("fasta", "genbank") or (
        insert_format == "genbank" and insert_features
    ):
        raise ValueError(
            "The insert must be a coding FASTA or an annotated GenBank cassette. A GenBank "
            "cassette takes its features from the GenBank record, so insert_features must "
            "be empty."
        )
    insert_record = SeqIO.read(io.StringIO(insert_text), insert_format)
    dna = str(insert_record.seq).upper()
    if not 60 <= len(dna) <= 6000 or not re.fullmatch("[ACGT]+", dna):
        raise ValueError(
            "The insert must be one sequence of 60 to 6,000 bases, each A, C, G or T; "
            f"found {len(dna)} bases."
        )
    protein_id, expected = parse_fasta(protein_text)
    source_cds, genetic_code = (
        cassette_cds(insert_record) if insert_format == "genbank" else (None, 11)
    )
    coding_dna = source_cds.extract(Seq(dna)) if source_cds is not None else Seq(dna)
    try:
        translated = str(coding_dna.translate(table=genetic_code, cds=True))
    except ValueError as exc:
        raise ValueError(
            f"The insert does not translate as a complete CDS: {exc}"
        ) from exc
    if translated != expected:
        raise ValueError(
            f"The insert translation under genetic code {genetic_code} differs from "
            "expected_protein."
        )
    if source_cds is not None and source_cds.qualifiers.get(
        "translation", [translated]
    ) != [translated]:
        raise ValueError(
            "The translation qualifier of the cassette CDS differs from the translation "
            "of its DNA."
        )
    if not isinstance(insert_features, (list, tuple)) or len(insert_features) > 30:
        raise ValueError("insert_features must be a list of at most 30 regions.")
    insert = Dseqrecord(
        insert_record if source_cds is not None else dna, name="coding_insert"
    )
    insert_context = None
    if source_cds is not None:
        insert_context = {
            "format": "genbank",
            "length": len(dna),
            "genetic_code": genetic_code,
            "cds": feature_record(source_cds)["parts"][0],
            "source_feature_count": len(insert.features),
        }
        for index, feature in enumerate(insert.features):
            feature.qualifiers["bf_insert_feature"] = [str(index)]
            if feature.type == "CDS":
                feature.qualifiers["translation"] = [translated]
    else:
        insert.features.append(
            SeqFeature(
                SimpleLocation(0, len(dna), strand=1),
                type="CDS",
                qualifiers={
                    "label": ["supplied coding insert"],
                    "translation": [translated],
                    "transl_table": ["11"],
                    "protein_id": [protein_id],
                },
            )
        )
    for item in insert_features:
        if (
            not isinstance(item, dict)
            or set(item) != {"label", "start", "end"}
            or type(item["start"]) is not int
            or type(item["end"]) is not int
            or not 1 <= item["start"] <= item["end"] <= len(dna)
            or not isinstance(item["label"], str)
            or not 1 <= len(item["label"]) <= 120
            or any(ord(c) < 32 for c in item["label"])
        ):
            raise ValueError(
                "Each entry of insert_features must have the fields label, start and end "
                "only: a label of 1 to 120 characters without control characters, and "
                "integer one-based inclusive positions with "
                f"1 <= start <= end <= {len(dna)}."
            )
        insert.features.append(
            SeqFeature(
                SimpleLocation(item["start"] - 1, item["end"], strand=1),
                type="misc_feature",
                qualifiers={
                    "label": [item["label"]],
                    "note": ["User-supplied insert region"],
                },
            )
        )
    vector = Dseqrecord(vector_record, circular=True)
    if len(vector.features) > 500:
        raise ValueError(
            f"The vector has {len(vector.features)} features; at most 500 are accepted."
        )
    vector.features = [f for f in vector.features if f.location is not None]
    if len(vector.features) != len(vector_record.features):
        raise ValueError(
            "The vector contains a feature without a location; every vector feature "
            "needs an exact location."
        )
    originals = []
    for index, feature in enumerate(vector.features):
        if any(
            type(p.start).__name__ != "ExactPosition"
            or type(p.end).__name__ != "ExactPosition"
            or p.ref is not None
            or not 0 <= int(p.start) < int(p.end) <= len(vector)
            for p in feature.location.parts
        ):
            raise ValueError(
                "A vector feature location is not an exact position range inside the "
                "vector; every vector feature location must be exact."
            )
        originals.append(feature_record(feature))
        # A previous product used as backbone is a new source, not this insert.
        feature.qualifiers.pop("bf_insert_feature", None)
        feature.qualifiers["bf_source_feature"] = [str(index)]
    try:
        enzyme = next(iter(RestrictionBatch([config["enzyme"]])))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{config['enzyme']} is not a restriction enzyme known to Biopython."
        ) from exc
    sites = len(enzyme.search(vector_record.seq, linear=False))
    if not enzyme.is_blunt() or sites != 1:
        raise ValueError(
            f"{config['enzyme']} must cut bluntly and cut the vector exactly once; "
            f"found {sites} sites and a "
            f"{'blunt' if enzyme.is_blunt() else 'non-blunt'} cut."
        )
    linear = vector.linearize(enzyme)
    amplicon = primer_design(
        insert, limit=config["primer_min"], target_tm=config["target_tm"]
    )
    if getattr(amplicon, "template", None) is None:
        raise ValueError(
            f"No primer pair with annealing footprints of at least {config['primer_min']} "
            "bases could be designed for this insert at a target melting temperature of "
            f"{config['target_tm']:g} °C. Lower primer_min, change target_tm or use a "
            "longer insert."
        )
    # These are preliminary untailed design guides, not the final oligos.
    # assembly_fragments adds the final tailed-primer binding annotations.
    amplicon.template.features = [
        f
        for f in amplicon.template.features
        if f.type != "primer_bind" or "bf_insert_feature" in f.qualifiers
    ]
    fragments = assembly_fragments(
        [linear, amplicon], overlap=config["overlap"], circular=True
    )
    amplified = fragments[1]
    checked = pcr(
        amplified.forward_primer,
        amplified.reverse_primer,
        insert,
        limit=config["primer_min"],
    )
    if str(checked.seq).upper() != str(amplified.seq).upper():
        raise ValueError(
            "The simulated PCR product differs from the designed insert fragment."
        )
    products = Assembly(
        fragments,
        limit=config["overlap"],
        algorithm=gibson_overlap,
        use_fragment_order=False,
        use_all_fragments=True,
    ).assemble_circular(max_assemblies=50)
    unique = {product.seguid(): product for product in products}
    if len(unique) != 1:
        raise ValueError(
            f"The assembly gave {len(unique)} distinct circular sequences; exactly 1 is "
            "required, so no product is selected."
        )
    expected_dna = str(linear.seq).upper() + dna
    product = next(iter(unique.values())).synced(
        Dseqrecord(expected_dna, circular=True)
    )
    if str(product.seq).upper() != expected_dna:
        raise ValueError(
            "The assembled sequence differs from the complete linearized backbone "
            "followed by the insert."
        )
    # Remove duplicate PCR annotation records and replace the old whole-vector
    # source feature; it is not the source of the newly inserted bases.
    seen, features, retained, retained_insert = set(), [], set(), set()
    for feature in product.features:
        key = json.dumps(feature_record(feature), sort_keys=True)
        insert_id = feature.qualifiers.get("bf_insert_feature")
        if (feature.type == "source" and not insert_id) or key in seen:
            continue
        seen.add(key)
        source_id = feature.qualifiers.get("bf_source_feature")
        if source_id:
            index = int(source_id[0])
            retained.add(index)
            if (
                str(feature.extract(Seq(str(product.seq)))).upper()
                != str(vector_record.features[index].extract(vector_record.seq)).upper()
            ):
                raise ValueError(
                    "The sequence of a vector feature changed when the feature was "
                    "transferred to the product."
                )
        if insert_id:
            index = int(insert_id[0])
            if (
                index in retained_insert
                or feature.extract(Seq(str(product.seq))).upper()
                != insert_record.features[index].extract(Seq(dna)).upper()
            ):
                raise ValueError(
                    "The sequence of a cassette feature changed, or the feature was "
                    "duplicated, when the feature was transferred to the product."
                )
            retained_insert.add(index)
        features.append(feature)
    if source_cds is not None and retained_insert != set(
        range(len(insert_record.features))
    ):
        raise ValueError("The product lacks one or more features of the cassette.")
    product.features = features
    cds = [
        f
        for f in features
        if f.type == "CDS"
        and (
            "bf_insert_feature" in f.qualifiers
            if source_cds is not None
            else f.qualifiers.get("protein_id") == [protein_id]
        )
    ]
    if (
        len(cds) != 1
        or str(
            cds[0]
            .extract(Seq(str(product.seq)))
            .translate(table=genetic_code, cds=True)
        )
        != expected
    ):
        raise ValueError(
            "The product has no single CDS that translates to expected_protein."
        )
    product.id = product.name = "bf_construct"
    product.description = (
        "Explicit vector and annotated cassette; simulated overlap assembly"
        if insert_context
        else "Explicit vector and coding insert; simulated overlap assembly"
    )
    product.annotations.update(
        {"molecule_type": "DNA", "topology": "circular", "date": "01-JAN-2000"}
    )
    product.annotations["comment"] = (
        "Computational assembly. LOCUS date is a fixed serialization placeholder; execution timestamps are retained in the run record."
    )
    stream = io.StringIO()
    SeqIO.write(product, stream, "genbank")
    primers = []
    for name, primer in [
        ("insert-forward", amplified.forward_primer),
        ("insert-reverse", amplified.reverse_primer),
    ]:
        primers.append(
            {
                "id": name,
                "sequence": str(primer.seq).upper(),
                "footprint": str(primer.footprint).upper(),
                "tail": str(primer.tail).upper(),
                "footprint_tm_degC": tm_default(primer.footprint),
            }
        )
    removed = [
        {
            "source_feature_index": i,
            **feature,
            "reason": "The source annotation of the whole original vector is not copied to the recombinant construct"
            if feature["type"] == "source"
            else "Not retained by linearization or assembly: the feature overlaps the edited region or cannot be transferred intact",
        }
        for i, feature in enumerate(originals)
        if i not in retained
    ]
    return {
        "schema_version": "0.1",
        "sequence": expected_dna,
        "sequence_sha256": hashlib.sha256(expected_dna.encode()).hexdigest(),
        "length": len(product),
        "topology": "circular",
        "origin": "First base of the linearized backbone; the annotated cassette follows the complete backbone"
        if insert_context
        else "First base of the linearized backbone; the coding insert follows the complete backbone",
        "translation": translated,
        "protein_id": protein_id,
        "protein_sequence_sha256": hashlib.sha256(translated.encode()).hexdigest(),
        "settings": config,
        "versions": versions,
        "unique_circular_products": len(unique),
        "raw_circular_products": len(products),
        "primers": primers,
        "features": [feature_record(f) for f in features],
        "removed_vector_features": removed,
        "genbank": stream.getvalue(),
        "linear_backbone_fasta": linear.format("fasta"),
        "pcr_product_fasta": amplified.format("fasta"),
        "tm_model": "pydna.tm.tm_default on the annealing footprint",
        **({"insert_context": insert_context} if insert_context else {}),
    }
