"""Annotations are transferred from the strongest hits that meet the thresholds."""

from __future__ import annotations

from alambiq.capabilities.characterize.science.transfer import select_hits, transfer


def hit(
    target,
    identity,
    coverage=1.0,
    evalue=1e-30,
    bits=200.0,
    target_coverage=1.0,
    **annotations,
):
    return {
        "target": target,
        "identity": identity,
        "coverage": coverage,
        "target_coverage": target_coverage,
        "evalue": evalue,
        "bits": bits,
        "annotations": annotations,
    }


THRESHOLDS = dict(
    min_identity=0.4,
    min_coverage=0.8,
    max_evalue=1e-5,
    max_hits=5,
    exclude_identical=False,
)


def test_hits_below_a_threshold_are_dropped():
    rows = [
        hit("A", 0.39),
        hit("B", 0.9, coverage=0.79),
        hit("C", 0.9, evalue=1e-4),
        hit("D", 0.9),
    ]
    assert [h["target"] for h in select_hits(rows, **THRESHOLDS)] == ["D"]


def test_hits_are_ordered_by_bit_score_then_accession_and_limited():
    rows = [hit("B", 0.9, bits=100), hit("A", 0.9, bits=100), hit("C", 0.9, bits=300)]
    chosen = select_hits(rows, **{**THRESHOLDS, "max_hits": 2})
    assert [h["target"] for h in chosen] == ["C", "A"]


def test_an_identical_hit_is_dropped_only_when_asked():
    rows = [hit("SELF", 1.0, bits=900), hit("REL", 0.7)]
    kept = select_hits(rows, **THRESHOLDS)
    dropped = select_hits(rows, **{**THRESHOLDS, "exclude_identical": True})
    assert [h["target"] for h in kept] == ["SELF", "REL"]
    assert [h["target"] for h in dropped] == ["REL"]


def test_a_full_identity_match_of_part_of_the_target_is_not_identical():
    rows = [hit("PART", 1.0, target_coverage=0.5)]
    assert select_hits(rows, **{**THRESHOLDS, "exclude_identical": True})


def test_a_term_carries_its_support_best_identity_and_sources():
    hits = [
        hit("A", 0.9, ec=["2.7.11.1"], location=["Cytoplasm"]),
        hit("B", 0.5, ec=["2.7.11.1", "3.1.3.16"], location=["Nucleus"]),
    ]
    terms = transfer(hits)
    assert terms["ec"][0] == {
        "term": "2.7.11.1",
        "name": "",
        "support": 2,
        "best_identity": 0.9,
        "sources": ["A", "B"],
    }
    assert [t["term"] for t in terms["ec"]] == ["2.7.11.1", "3.1.3.16"]
    assert [t["term"] for t in terms["location"]] == ["Cytoplasm", "Nucleus"]


def test_a_go_term_is_split_into_its_identifier_and_label():
    terms = transfer([hit("A", 0.9, go_function=["GO:0005524|ATP binding"])])
    assert terms["go_function"][0]["term"] == "GO:0005524"
    assert terms["go_function"][0]["name"] == "ATP binding"


def test_no_hits_give_empty_lists_in_every_category():
    assert transfer([]) == {
        "ec": [],
        "go_function": [],
        "go_process": [],
        "go_component": [],
        "location": [],
    }
