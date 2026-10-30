"""The differences between a variant and its reference are read from an alignment."""

from __future__ import annotations

import pytest

from alambiq.capabilities.characterize.science.substitutions import (
    call,
    notation,
    parse_variant,
)


def read(variant, reference, vs=1, ve=None, ts=1, te=None, vl=None, tl=None):
    return call(
        variant,
        reference,
        variant_start=vs,
        variant_end=ve or len(variant.replace("-", "")),
        variant_length=vl or len(variant.replace("-", "")),
        reference_start=ts,
        reference_end=te or len(reference.replace("-", "")),
        reference_length=tl or len(reference.replace("-", "")),
    )


def test_identical_sequences_have_no_differences():
    found = read("ACDEF", "ACDEF")
    assert found["substitutions"] == found["insertions"] == found["deletions"] == []


def test_a_substitution_is_named_by_reference_residue_and_position():
    found = read("ACWEF", "ACDEF")
    assert found["substitutions"] == [{"reference": "D", "position": 3, "variant": "W"}]
    assert notation(found["substitutions"]) == "D3W"


def test_several_substitutions_are_joined_with_colons():
    found = read("WCDEY", "ACDEF")
    assert notation(found["substitutions"]) == "A1W:F5Y"


def test_positions_follow_the_start_of_the_reference_alignment():
    found = read("CWE", "CDE", ts=11, te=13, vs=1, ve=3, tl=30)
    assert found["substitutions"][0]["position"] == 12
    assert found["unaligned_reference_start"] == 10
    assert found["unaligned_reference_end"] == 17


def test_an_insertion_is_placed_after_the_preceding_reference_residue():
    found = read("ACGGDEF", "AC--DEF")
    assert found["insertions"] == [{"after_position": 2, "residues": "GG"}]
    assert found["substitutions"] == [] and found["deletions"] == []


def test_a_deletion_gives_the_reference_residues_the_variant_lacks():
    found = read("AC---F", "ACDEXF")
    assert found["deletions"] == [{"start": 3, "end": 5, "residues": "DEX"}]


def test_an_insertion_and_a_deletion_side_by_side_are_kept_apart():
    found = read("ACGG-F", "AC--DF")
    assert found["insertions"] == [{"after_position": 2, "residues": "GG"}]
    assert found["deletions"] == [{"start": 3, "end": 3, "residues": "D"}]


def test_positions_count_across_insertions_and_deletions():
    found = read("AC--EFW", "ACDEEFF")
    assert found["deletions"] == [{"start": 3, "end": 4, "residues": "DE"}]
    assert found["substitutions"] == [{"reference": "F", "position": 7, "variant": "W"}]


def test_residues_outside_the_alignment_are_counted_at_each_end():
    found = read("CDE", "CDE", vs=4, ve=6, vl=10, ts=1, te=3, tl=3)
    assert found["unaligned_variant_start"] == 3
    assert found["unaligned_variant_end"] == 4


def test_strings_of_different_length_are_refused():
    with pytest.raises(ValueError, match="differ in length"):
        read("ACD", "AC")


SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


def test_a_variant_is_read_as_residue_position_residue_triples():
    assert parse_variant("K6N:T7A", SEQUENCE) == [("K", 6, "N"), ("T", 7, "A")]


def test_one_substitution_is_a_variant():
    assert parse_variant("I3V", SEQUENCE) == [("I", 3, "V")]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("K6", "is not a substitution"),
        ("6KN", "is not a substitution"),
        ("K6N,T7A", "is not a substitution"),
        ("K600N", "outside the sequence"),
        ("A6N", "is K in the sequence, not A"),
        ("K6K", "must be one of the 20 amino acids and differ"),
        ("K6X", "must be one of the 20 amino acids and differ"),
        ("K6N:K6R", "two substitutions are at one position"),
    ],
)
def test_a_malformed_or_inconsistent_variant_is_refused_with_the_reason(text, message):
    with pytest.raises(ValueError, match=message):
        parse_variant(text, SEQUENCE)


def test_more_than_fifty_substitutions_are_refused():
    residues = [r for r in SEQUENCE]
    too_many = ":".join(
        f"{residues[i]}{i + 1}{'A' if residues[i] != 'A' else 'C'}" for i in range(51)
    )
    with pytest.raises(ValueError, match="at most 50 substitutions"):
        parse_variant(too_many, SEQUENCE)
