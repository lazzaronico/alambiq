"""protein.annotate_function transfers annotations from the closest reviewed relatives."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from alambiq.engines.swissprot_annotations import COLUMNS

AMINO = "ACDEFGHIKLMNPQRSTVWY"
MMSEQS = os.environ.get("ALAMBIQ_ENV_MMSEQS")
BINARY = (Path(MMSEQS).parent / "mmseqs") if MMSEQS else None


def mmseqs_binary() -> str | None:
    if BINARY and BINARY.is_file():
        return str(BINARY)
    return shutil.which("mmseqs")


def protein(seed: int, length: int = 200) -> str:
    return "".join(random.Random(seed).choices(AMINO, k=length))


BASE = protein(1)


def mutate(sequence: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(
        rng.choice([a for a in AMINO if a != c]) if rng.random() < rate else c
        for c in sequence
    )


# accession: (sequence, EC numbers, molecular function terms, locations)
ENTRIES = {
    "P00010": (BASE, ["1.1.1.1"], ["GO:0004022|alcohol dehydrogenase"], ["Cytoplasm"]),
    "P00011": (
        mutate(BASE, 0.10, 11),
        ["1.1.1.1"],
        ["GO:0004022|alcohol dehydrogenase"],
        ["Cytoplasm"],
    ),
    "P00012": (
        mutate(BASE, 0.25, 12),
        ["1.1.1.1", "1.1.1.2"],
        ["GO:0004022|alcohol dehydrogenase"],
        ["Cytoplasm", "Nucleus"],
    ),
    "P00013": (
        mutate(BASE, 0.40, 13),
        ["2.7.7.7"],
        ["GO:0003887|DNA-directed DNA polymerase"],
        [],
    ),
    **{f"Q{i:05d}": (protein(100 + i, 300), [], [], []) for i in range(60)},
}


def build_resource(root: Path, entries) -> None:
    binary = mmseqs_binary()
    if binary is None:
        pytest.skip("mmseqs is not available to build a test database")
    root.mkdir()
    fasta = root.parent / "swissprot.faa"
    fasta.write_text(
        "".join(
            f">sp|{acc}|{acc}_TEST name\n{row[0]}\n" for acc, row in entries.items()
        )
    )
    subprocess.run(
        [binary, "createdb", str(fasta), str(root / "swissprot-db")],
        check=True,
        capture_output=True,
    )
    lines = ["\t".join(COLUMNS)]
    for acc, (sequence, ec, function, location) in entries.items():
        lines.append(
            "\t".join(
                [
                    acc,
                    f"Protein {acc}",
                    "Homo sapiens (Human)",
                    str(len(sequence)),
                    "; ".join(ec),
                    "; ".join(function),
                    "",
                    "",
                    "; ".join(location),
                ]
            )
        )
    (root / "annotations.tsv").write_text("\n".join(lines) + "\n")


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", MMSEQS or sys.executable)


@pytest.fixture
def toy(tmp_path, monkeypatch, engine):
    from alambiq.core import resources

    root = tmp_path / "swissprot"
    build_resource(root, ENTRIES)
    monkeypatch.setattr(resources, "verify", lambda name: root)
    monkeypatch.setattr(
        resources,
        "spec",
        lambda name: {"logical_id": "swissprot-2026_03-v1", "version": "2026_03"},
    )
    monkeypatch.setattr(resources, "identity", lambda name: "ab" * 32)
    return root


def query_file(tmp_path: Path, records: dict[str, str] | None = None) -> str:
    path = tmp_path / "q.faa"
    path.write_text(
        "".join(f">{k}\n{v}\n" for k, v in (records or {"query": BASE}).items())
    )
    return str(path)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {
        "id": "annotate",
        "capability": "protein.annotate_function",
        "params": params,
    }
    return execute({"id": "a", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def refusal(params: dict, tmp_path: Path) -> str:
    outcome = run(params, tmp_path)
    assert not outcome.ok
    return " ".join(x.error or "" for x in outcome.steps.values())


@pytest.mark.usefixtures("engine")
class TestRefusals:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"min_identity": 1.5}, "min_identity must be between 0 and 1"),
            ({"min_coverage": -0.1}, "min_coverage must be between 0 and 1"),
            ({"max_evalue": 0}, "max_evalue must be above 0"),
            ({"max_hits": 0}, "max_hits must be an integer from 1 to 50"),
            ({"threads": 65}, "threads must be an integer from 1 to 64"),
            ({"sensitivity": 9}, "sensitivity must be between 1 and 7.5"),
        ],
    )
    def test_a_setting_out_of_range_is_refused(self, tmp_path, change, message):
        params = {"proteins": query_file(tmp_path), **change}
        assert message in refusal(params, tmp_path)

    def test_a_protein_that_is_too_short_is_refused_with_its_name(self, tmp_path):
        params = {"proteins": query_file(tmp_path, {"tiny": "ACDE"})}
        message = refusal(params, tmp_path)
        assert "tiny" in message
        assert "10–5000" in message

    def test_a_repeated_identifier_is_refused(self, tmp_path):
        path = tmp_path / "dup.faa"
        path.write_text(f">a\n{BASE}\n>a\n{BASE}\n")
        assert "appears twice" in refusal({"proteins": str(path)}, tmp_path)


class TestAnnotation:
    @pytest.fixture
    def outcome(self, tmp_path, toy):
        return run_ok({"proteins": query_file(tmp_path), "threads": 2}, tmp_path)

    @pytest.fixture
    def annotation(self, outcome):
        return outcome.outputs("annotate")["proteins"][0]

    def test_the_sources_are_the_closest_relatives_strongest_first(self, annotation):
        accessions = [s["accession"] for s in annotation["sources"]]
        assert accessions[:3] == ["P00010", "P00011", "P00012"]
        assert "Q00000" not in accessions

    def test_a_source_carries_its_name_organism_and_identity(self, annotation):
        first = annotation["sources"][0]
        assert first["name"] == "Protein P00010"
        assert first["organism"] == "Homo sapiens (Human)"
        assert first["identity"] == pytest.approx(1.0)
        assert (
            0.0
            < annotation["sources"][2]["identity"]
            < annotation["sources"][1]["identity"]
        )

    def test_a_term_has_its_support_best_identity_and_sources(self, annotation):
        ec = {t["term"]: t for t in annotation["ec"]}
        assert ec["1.1.1.1"]["support"] == 3
        assert ec["1.1.1.1"]["sources"] == ["P00010", "P00011", "P00012"]
        assert ec["1.1.1.1"]["best_identity"] == pytest.approx(1.0)
        assert ec["1.1.1.2"]["support"] == 1
        assert annotation["ec"][0]["term"] == "1.1.1.1"

    def test_a_go_term_is_listed_with_its_label(self, annotation):
        term = annotation["go_function"][0]
        assert (term["term"], term["name"]) == ("GO:0004022", "alcohol dehydrogenase")

    def test_locations_are_listed_by_support(self, annotation):
        assert [t["term"] for t in annotation["location"]] == ["Cytoplasm", "Nucleus"]

    def test_the_record_names_the_database_and_the_settings(self, outcome):
        record = json.loads(Path(outcome.outputs("annotate")["record"]).read_text())
        assert record["database"] == "swissprot-2026_03-v1"
        assert record["settings"]["max_hits"] == 5
        assert record["proteins"][0]["ec"][0]["term"] == "1.1.1.1"

    def test_the_report_states_the_database_and_the_counts(self, outcome):
        from alambiq.capabilities.characterize.protein_annotate_function_report import (
            render,
        )
        from alambiq.core.registry import get

        result = outcome.steps["annotate"].result
        html = render(result)
        assert "1 of 1 proteins annotated" in html
        assert "2026_03" in html
        assert "1.1.1.1" in html
        assert get("protein.annotate_function") is not None


class TestRelativeRunDirectory:
    def test_a_relative_run_directory_reaches_the_engine_as_absolute_paths(
        self, tmp_path, toy, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        outcome = run_ok(
            {"proteins": query_file(tmp_path), "threads": 2}, Path("."), "relative-run"
        )
        assert outcome.outputs("annotate")["annotated"] == 1


class TestOptions:
    def test_excluding_identical_hits_drops_the_entry_of_the_protein(
        self, tmp_path, toy
    ):
        outcome = run_ok(
            {"proteins": query_file(tmp_path), "exclude_identical": True, "threads": 2},
            tmp_path,
        )
        sources = [
            s["accession"]
            for s in outcome.outputs("annotate")["proteins"][0]["sources"]
        ]
        assert sources[0] == "P00011"
        assert "P00010" not in sources

    def test_the_database_is_named_in_the_request(self, tmp_path, toy):
        outcome = run_ok(
            {
                "proteins": query_file(tmp_path),
                "database": "swissprot-2026_03",
                "threads": 2,
            },
            tmp_path,
        )
        assert outcome.outputs("annotate")["database_version"] == "2026_03"

    def test_max_hits_limits_the_sources(self, tmp_path, toy):
        outcome = run_ok(
            {"proteins": query_file(tmp_path), "max_hits": 2, "threads": 2}, tmp_path
        )
        annotation = outcome.outputs("annotate")["proteins"][0]
        assert len(annotation["sources"]) == 2
        assert {t["term"] for t in annotation["ec"]} == {"1.1.1.1"}

    def test_a_stricter_identity_threshold_removes_distant_sources(self, tmp_path, toy):
        outcome = run_ok(
            {"proteins": query_file(tmp_path), "min_identity": 0.95, "threads": 2},
            tmp_path,
        )
        assert [
            s["accession"]
            for s in outcome.outputs("annotate")["proteins"][0]["sources"]
        ] == ["P00010"]

    def test_a_protein_without_relatives_is_returned_empty_with_a_note(
        self, tmp_path, toy
    ):
        stranger = protein(999, 250)
        outcome = run_ok(
            {
                "proteins": query_file(tmp_path, {"query": BASE, "stranger": stranger}),
                "threads": 2,
            },
            tmp_path,
        )
        out = outcome.outputs("annotate")
        empty = next(p for p in out["proteins"] if p["id"] == "stranger")
        assert empty["sources"] == [] and empty["ec"] == []
        assert (out["annotated"], out["unmatched"]) == (1, 1)
        assert any("stranger" in n for n in outcome.steps["annotate"].notes)

    def test_a_hit_absent_from_the_annotation_table_is_an_error(self, tmp_path, toy):
        lines = (toy / "annotations.tsv").read_text().splitlines()
        (toy / "annotations.tsv").write_text(
            "\n".join(line for line in lines if not line.startswith("P00011\t")) + "\n"
        )
        message = refusal({"proteins": query_file(tmp_path), "threads": 2}, tmp_path)
        assert "absent from the annotation table" in message
