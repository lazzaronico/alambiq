"""The annotation table of the Swiss-Prot resource is built from the UniProt flat file."""

from __future__ import annotations

import gzip

import pytest

from alambiq.engines import swissprot_annotations as table

FLAT_FILE = """\
ID   KINA_TEST               Reviewed;         263 AA.
AC   P00001; Q99999;
DT   01-JAN-2000, integrated into UniProtKB/Swiss-Prot.
DE   RecName: Full=Test protein kinase A {ECO:0000255|HAMAP-Rule:MF_1};
DE            EC=2.7.11.1 {ECO:0000255};
DE   AltName: Full=Other name;
DE   Includes:
DE     RecName: Full=Second activity;
DE              EC=3.1.3.16;
OS   Homo sapiens
OS   (Human).
CC   -!- FUNCTION: Does something.
CC   -!- SUBCELLULAR LOCATION: Cytoplasm {ECO:0000250}. Nucleus {ECO:0000250}.
CC       Cell membrane {ECO:0000250}; Multi-pass membrane protein {ECO:0000250}.
CC       Note=Also found elsewhere.
CC   -!- SIMILARITY: Belongs to a family.
DR   GO; GO:0005524; F:ATP binding; IEA:UniProtKB-KW.
DR   GO; GO:0004672; F:protein kinase activity; IDA:UniProtKB.
DR   GO; GO:0006468; P:protein phosphorylation; IEA:InterPro.
DR   GO; GO:0005737; C:cytoplasm; IDA:UniProtKB.
DR   Pfam; PF00069; Pkinase; 1.
SQ   SEQUENCE   263 AA;  29735 MW;  B4840739BF7D4121 CRC64;
     MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RKLLDYKEWS PPRVQVECPK
//
ID   MULT_TEST               Reviewed;          30 AA.
AC   P00004;
DE   RecName: Full=A protein with two location comments;
OS   Homo sapiens (Human).
CC   -!- SUBCELLULAR LOCATION: [Isoform 1]: Cell membrane {ECO:0000250}; Single-pass
CC       type I membrane protein {ECO:0000250}. Endosome {ECO:0000250}.
CC       Note=Found in vesicles.
CC   -!- SUBCELLULAR LOCATION: [Isoform 2]: Secreted {ECO:0000250}.
CC   -!- TISSUE SPECIFICITY: Widely expressed.
SQ   SEQUENCE   30 AA;  3000 MW;  0000000000000000 CRC64;
     MKKKKKKKKK
//
ID   NOEC_TEST               Reviewed;          10 AA.
AC   P00002;
DE   RecName: Full=A protein with no annotations;
OS   Escherichia coli.
SQ   SEQUENCE   10 AA;  1000 MW;  0000000000000000 CRC64;
     MKKKKKKKKK
//
ID   ISOF_TEST               Reviewed;          20 AA.
AC   P00003;
DE   SubName: Full=Only a submitted name;
OS   Mus musculus (Mouse).
CC   -!- SUBCELLULAR LOCATION: [Isoform 1]: Secreted {ECO:0000305}.
CC       [Isoform 2]: Cytoplasm.
SQ   SEQUENCE   20 AA;  2000 MW;  0000000000000000 CRC64;
     MKKKKKKKKK
//
"""


@pytest.fixture
def flat_file(tmp_path):
    path = tmp_path / "uniprot_sprot.dat.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(FLAT_FILE)
    return path


def rows(flat_file, tmp_path):
    output = tmp_path / "annotations.tsv"
    assert table.build(flat_file, output) == 4
    return table.read(output, {"P00001", "P00002", "P00003", "P00004"})


def test_the_first_accession_names_the_entry(flat_file, tmp_path):
    assert sorted(rows(flat_file, tmp_path)) == ["P00001", "P00002", "P00003", "P00004"]


def test_a_row_carries_name_organism_length_and_ec_numbers(flat_file, tmp_path):
    row = rows(flat_file, tmp_path)["P00001"]
    assert row["name"] == "Test protein kinase A"
    assert row["organism"] == "Homo sapiens (Human)"
    assert row["length"] == 263
    assert row["ec"] == ["2.7.11.1", "3.1.3.16"]


def test_go_terms_are_split_by_aspect(flat_file, tmp_path):
    row = rows(flat_file, tmp_path)["P00001"]
    assert row["go_function"] == [
        "GO:0005524|ATP binding",
        "GO:0004672|protein kinase activity",
    ]
    assert row["go_process"] == ["GO:0006468|protein phosphorylation"]
    assert row["go_component"] == ["GO:0005737|cytoplasm"]


def test_locations_are_the_first_term_of_each_sentence_without_notes(
    flat_file, tmp_path
):
    assert rows(flat_file, tmp_path)["P00001"]["location"] == [
        "Cytoplasm",
        "Nucleus",
        "Cell membrane",
    ]


def test_isoform_prefixes_are_dropped_from_locations(flat_file, tmp_path):
    assert rows(flat_file, tmp_path)["P00003"]["location"] == ["Secreted", "Cytoplasm"]


def test_an_entry_without_annotations_has_empty_lists(flat_file, tmp_path):
    row = rows(flat_file, tmp_path)["P00002"]
    assert row["ec"] == row["go_function"] == row["location"] == []


def test_a_submitted_name_is_not_a_recommended_name(flat_file, tmp_path):
    assert rows(flat_file, tmp_path)["P00003"]["name"] == ""


def test_read_returns_only_the_requested_accessions(flat_file, tmp_path):
    output = tmp_path / "annotations.tsv"
    table.build(flat_file, output)
    assert list(table.read(output, {"P00002", "P12345"})) == ["P00002"]


def test_a_table_with_other_columns_is_refused(tmp_path):
    path = tmp_path / "other.tsv"
    path.write_text("accession\tname\nP1\tx\n")
    with pytest.raises(ValueError, match="has the columns"):
        table.read(path, {"P1"})


def test_every_location_comment_of_an_entry_contributes(flat_file, tmp_path):
    assert rows(flat_file, tmp_path)["P00004"]["location"] == [
        "Cell membrane",
        "Endosome",
        "Secreted",
    ]
