"""sequence.generate_msa: the A3M reader, the depth verdict, and real searches.

The A3M tests use alignments small enough to count by hand. The search tests
build a small MMseqs2 database of constructed relatives and point the pinned
UniRef50 resource at it, so the real engine runs end to end in seconds; one more
test searches the real UniRef50 release where it is installed.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from alambiq.capabilities.characterize.science.msa import (
    assess,
    canonicalize,
    read_a3m,
)

A3M = """>query
ACDEF
>r1
ACDEF
>r2
ACDEY
>r3
AC---
>r4
A-DEF
>r5
WCDEY
>r6
WCDEY
"""


def judge(text=A3M, query="ACDEF", **overrides):
    settings = {
        "minimum_query_coverage": 0.5,
        "minimum_alignment_sequences": 1,
        "minimum_diverse_sequences": 0,
        **overrides,
    }
    return assess(text, query, **settings)


class TestA3M:
    def test_records_are_read_in_order_with_wrapped_lines_joined(self):
        records = read_a3m(">a\nAC\nDE\n>b desc\nWWWW\n")
        assert records == [("a", "ACDE"), ("b desc", "WWWW")]

    def test_canonical_form_keeps_the_query_first_and_sorts_the_rest(self):
        text = canonicalize(
            ">q\nACDEF\n>z\nACDEY\n>b\nWCDEY\n>b\nWCDEY\n".replace(
                ">b\nWCDEY\n>b", ">b\nWCDEY\n>c"
            )
        )
        assert [name for name, _ in read_a3m(text)] == ["q", "b", "c", "z"]

    def test_canonical_form_does_not_depend_on_line_wrapping(self):
        assert canonicalize(">q\nACDEF\n>a\nWC\nDEY\n") == canonicalize(
            ">q\nACD\nEF\n>a\nWCDEY\n"
        )

    def test_a_repeated_identifier_is_refused(self):
        with pytest.raises(ValueError, match="unique"):
            canonicalize(">q\nACDEF\n>a\nACDEY\n>a\nACDEY\n")

    def test_sequence_before_a_header_is_refused(self):
        with pytest.raises(ValueError, match="precedes"):
            read_a3m("ACDEF\n>q\nACDEF\n")

    def test_an_empty_alignment_is_refused(self):
        with pytest.raises(ValueError, match="no sequence records"):
            read_a3m("# only a comment\n")


class TestVerdict:
    def test_the_counts_follow_from_the_rows(self):
        counts = judge()
        # Seven rows. Distinct aligned strings: ACDEF, ACDEY, AC---, A-DEF, WCDEY.
        assert counts["sequence_count"] == 7
        assert counts["unique_sequence_count"] == 5
        # Diverse: covers >= 0.5 of the query and is at most 90 % identical to it.
        # ACDEY (0.8 identical) and WCDEY (0.6) are; AC--- covers 0.4, and A-DEF is
        # 100 % identical over what it covers.
        assert counts["diverse_sequence_count"] == 2
        assert counts["query_coverage_fraction"] == 1.0
        assert counts["median_query_coverage_fraction"] == 1.0

    def test_depth_thresholds_decide_and_name_what_failed(self):
        assert judge(minimum_alignment_sequences=5, minimum_diverse_sequences=2)[
            "quality_passed"
        ]
        few = judge(minimum_alignment_sequences=6)
        assert few["reason_codes"] == ["msa-insufficient-unique-sequences"]
        narrow = judge(minimum_diverse_sequences=3)
        assert narrow["reason_codes"] == ["msa-insufficient-diverse-sequences"]
        assert not narrow["quality_passed"]

    def test_a_median_row_that_covers_too_little_fails_on_coverage(self):
        text = ">query\nACDEF\n" + "".join(f">s{i}\nA----\n" for i in range(5))
        verdict = judge(text)
        assert verdict["median_query_coverage_fraction"] == pytest.approx(0.2)
        assert "msa-insufficient-query-coverage" in verdict["reason_codes"]

    def test_lower_case_insertions_do_not_add_columns(self):
        verdict = judge(">query\nACDEF\n>x\nACwwDEF\n")
        assert verdict["query_coverage_fraction"] == 1.0
        assert verdict["unique_sequence_count"] == 1

    def test_a_first_row_that_is_not_the_query_is_refused(self):
        with pytest.raises(ValueError, match="does not reproduce"):
            judge(">query\nACDEY\n")

    def test_a_row_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="length does not match"):
            judge(">query\nACDEF\n>x\nACDE\n")


# --- searches --------------------------------------------------------------

AMINO = "ACDEFGHIKLMNPQRSTVWY"
MMSEQS = os.environ.get("ALAMBIQ_ENV_MMSEQS")
BINARY = (Path(MMSEQS).parent / "mmseqs") if MMSEQS else None


def mmseqs_binary() -> str | None:
    if BINARY and BINARY.is_file():
        return str(BINARY)
    return shutil.which("mmseqs")


def protein(seed: int, length: int = 150) -> str:
    return "".join(random.Random(seed).choices(AMINO, k=length))


BASE = protein(1)


def mutate(sequence: str, rate: float, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(
        rng.choice([a for a in AMINO if a != c]) if rng.random() < rate else c
        for c in sequence
    )


def relatives(count: int) -> dict[str, str]:
    return {
        f"fam{i:02d}": mutate(BASE, 0.15 + 0.35 * (i / max(count, 1)), 100 + i)
        for i in range(count)
    }


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", MMSEQS or sys.executable)


def toy_database(tmp_path: Path, monkeypatch, entries: dict[str, str]) -> Path:
    binary = mmseqs_binary()
    if binary is None:
        pytest.skip("mmseqs is not available to build a test database")
    root = tmp_path / "uniref"
    root.mkdir()
    fasta = tmp_path / "db.faa"
    fasta.write_text("".join(f">{k}\n{v}\n" for k, v in entries.items()))
    subprocess.run(
        [binary, "createdb", str(fasta), str(root / "uniref50-db")],
        check=True,
        capture_output=True,
    )
    from alambiq.core import resources

    monkeypatch.setattr(resources, "verify", lambda name: root)
    return root


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "msa", "capability": "sequence.generate_msa", "params": params}
    return execute({"id": "m", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def query_file(tmp_path: Path, records: dict[str, str] | None = None) -> str:
    path = tmp_path / "q.faa"
    path.write_text(
        "".join(f">{k}\n{v}\n" for k, v in (records or {"target": BASE}).items())
    )
    return str(path)


def rows(outcome, chain: int = 0) -> list[tuple[str, str]]:
    return read_a3m(
        Path(outcome.outputs("msa")["chains"][chain]["alignment"]).read_text()
    )


class TestToyDatabase:
    @pytest.fixture
    def outcome(self, tmp_path, monkeypatch, engine):
        entries = {
            **relatives(60),
            **{f"other{i}": protein(500 + i) for i in range(100)},
        }
        toy_database(tmp_path, monkeypatch, entries)
        return run_ok({"proteins": query_file(tmp_path), "threads": 2}, tmp_path)

    def test_a_deep_family_gives_a_generated_alignment_headed_by_the_query(
        self, outcome
    ):
        chain = outcome.outputs("msa")["chains"][0]
        assert (chain["decision"], chain["reason_codes"]) == ("generated", [])
        assert chain["length"] == 150
        assert rows(outcome)[0] == ("query", BASE)
        assert chain["diverse_sequences"] >= 16 and chain["unique_sequences"] >= 32

    def test_only_relatives_of_the_query_are_in_the_alignment(self, outcome):
        names = {name.split()[0] for name, _ in rows(outcome)[1:]}
        assert names and all(n.startswith("fam") for n in names)

    def test_the_stored_alignment_is_in_canonical_order(self, outcome):
        text = Path(outcome.outputs("msa")["chains"][0]["alignment"]).read_text()
        assert text == canonicalize(text)

    def test_the_record_names_the_database_and_that_nothing_is_paired(self, outcome):
        record = json.loads(Path(outcome.outputs("msa")["record"]).read_text())
        assert record["database"] == "uniref50-2026_02-v1"
        assert record["database_version"] == "2026_02"
        assert len(record["database_identity"]) == 64
        assert record["pairing"] == "unpaired-per-chain"
        assert record["engine_versions"]["mmseqs"]
        assert record["chains"][0]["query_sha256"]

    def test_the_alignment_feeds_the_msa_inspector(self, tmp_path, monkeypatch, engine):
        toy_database(tmp_path, monkeypatch, {**relatives(60), "other": protein(500)})
        from alambiq.api import run as execute

        outcome = execute(
            {
                "id": "chain",
                "steps": [
                    {
                        "id": "msa",
                        "capability": "sequence.generate_msa",
                        "params": {"proteins": query_file(tmp_path)},
                    },
                    {
                        "id": "inspect",
                        "capability": "sequence.inspect_msa",
                        "bind": {"alignment": "msa.chains.0.alignment"},
                        "params": {"format": "a3m"},
                    },
                ],
            },
            run_dir=tmp_path / "chain",
        )
        assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
        assert outcome.outputs("inspect")["columns"] == 150
        assert outcome.outputs("inspect")["query_id"] == "query"

    def test_the_report_states_the_database(
        self, outcome, tmp_path
    ):
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "UniRef50 2026_02" in page


class TestSearchSettings:
    def test_a_shallow_family_is_withheld_but_kept(self, tmp_path, monkeypatch, engine):
        toy_database(
            tmp_path,
            monkeypatch,
            {**relatives(5), **{f"o{i}": protein(600 + i) for i in range(20)}},
        )
        outcome = run_ok({"proteins": query_file(tmp_path)}, tmp_path)
        chain = outcome.outputs("msa")["chains"][0]
        assert chain["decision"] == "withheld"
        assert set(chain["reason_codes"]) >= {
            "msa-insufficient-unique-sequences",
            "msa-insufficient-diverse-sequences",
        }
        assert chain["sequences"] <= 6
        assert Path(chain["alignment"]).is_file()
        assert outcome.outputs("msa")["withheld"] == 1
        assert "withheld as too shallow" in " ".join(outcome.steps["msa"].notes)

    def test_lowering_the_depth_rules_lets_the_same_search_pass(
        self, tmp_path, monkeypatch, engine
    ):
        toy_database(tmp_path, monkeypatch, {**relatives(5), "o": protein(600)})
        outcome = run_ok(
            {
                "proteins": query_file(tmp_path),
                "minimum_alignment_sequences": 3,
                "minimum_diverse_sequences": 2,
            },
            tmp_path,
        )
        assert outcome.outputs("msa")["chains"][0]["decision"] == "generated"

    def test_a_near_identical_sequence_is_kept_not_filtered_out(
        self, tmp_path, monkeypatch, engine
    ):
        """MMseqs2's identity ceiling does not remove close hits, so none is offered."""
        twin = mutate(BASE, 0.01, 7)
        assert sum(a != b for a, b in zip(BASE, twin, strict=True)) <= 3
        toy_database(tmp_path, monkeypatch, {**relatives(40), "twin": twin})
        outcome = run_ok(
            {"proteins": query_file(tmp_path), "database": "uniref50-2026_02"}, tmp_path
        )
        assert "twin" in {n.split()[0] for n, _ in rows(outcome)}

    def test_the_row_cap_limits_the_alignment(self, tmp_path, monkeypatch, engine):
        toy_database(tmp_path, monkeypatch, relatives(60))
        outcome = run_ok(
            {"proteins": query_file(tmp_path), "maximum_sequences": 10}, tmp_path
        )
        assert len(rows(outcome)) <= 11

    def test_a_short_fragment_needs_a_lower_coverage_bar(
        self, tmp_path, monkeypatch, engine
    ):
        entries = {**relatives(30), "fragment": BASE[:45]}
        toy_database(tmp_path, monkeypatch, entries)
        default = run_ok({"proteins": query_file(tmp_path)}, tmp_path, "default")
        lenient = run_ok(
            {"proteins": query_file(tmp_path), "minimum_query_coverage": 0.2},
            tmp_path,
            "lenient",
        )
        assert "fragment" not in {n.split()[0] for n, _ in rows(default)}
        assert "fragment" in {n.split()[0] for n, _ in rows(lenient)}

    def test_fewer_iterations_never_find_more_than_more(
        self, tmp_path, monkeypatch, engine
    ):
        toy_database(tmp_path, monkeypatch, {**relatives(60), "o": protein(600)})
        one = run_ok(
            {"proteins": query_file(tmp_path), "iterations": 1}, tmp_path, "one"
        )
        three = run_ok(
            {"proteins": query_file(tmp_path), "iterations": 3}, tmp_path, "three"
        )
        assert len(rows(one)) <= len(rows(three))

    def test_a_stricter_evalue_and_a_lower_sensitivity_never_find_more(
        self, tmp_path, monkeypatch, engine
    ):
        toy_database(tmp_path, monkeypatch, {**relatives(60), "o": protein(600)})
        default = run_ok({"proteins": query_file(tmp_path)}, tmp_path, "default")
        strict = run_ok(
            {"proteins": query_file(tmp_path), "evalue": 1e-30, "sensitivity": 2.0},
            tmp_path,
            "strict",
        )
        assert len(rows(strict)) <= len(rows(default))

    def test_the_thread_count_does_not_change_the_alignment(
        self, tmp_path, monkeypatch, engine
    ):
        toy_database(tmp_path, monkeypatch, {**relatives(60), "o": protein(600)})
        one = run_ok({"proteins": query_file(tmp_path), "threads": 1}, tmp_path, "one")
        four = run_ok(
            {"proteins": query_file(tmp_path), "threads": 4}, tmp_path, "four"
        )
        assert rows(one) == rows(four)

    def test_each_chain_is_aligned_on_its_own(
        self, tmp_path, monkeypatch, engine
    ):
        other = protein(2, 120)
        entries = {
            **relatives(40),
            **{f"b{i:02d}": mutate(other, 0.2 + 0.01 * i, 300 + i) for i in range(40)},
        }
        toy_database(tmp_path, monkeypatch, entries)
        outcome = run_ok(
            {"proteins": query_file(tmp_path, {"a": BASE, "b": other})}, tmp_path
        )
        chains = outcome.outputs("msa")["chains"]
        assert [c["id"] for c in chains] == ["a", "b"]
        assert {n.split()[0][:3] for n, _ in rows(outcome, 0)[1:]} == {"fam"}
        assert {n.split()[0][:1] for n, _ in rows(outcome, 1)[1:]} == {"b"}


class TestRefusals:
    @pytest.fixture(autouse=True)
    def stand_in_engine(self, engine):
        """A refusal happens before the search runs, so the stand-in interpreter will do."""

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"sensitivity": 8}, "sensitivity must be between"),
            ({"evalue": 0}, "evalue must be between"),
            ({"minimum_query_coverage": 1.5}, "minimum_query_coverage must be between"),
            ({"iterations": 0}, "iterations must be an integer from 1 to 5"),
            ({"maximum_sequences": 0}, "maximum_sequences must be an integer"),
            ({"threads": 0}, "threads must be an integer from 1 to 64"),
        ],
    )
    def test_a_setting_outside_its_range_is_refused(self, tmp_path, changes, message):
        outcome = run({"proteins": query_file(tmp_path), **changes}, tmp_path)
        assert outcome.steps["msa"].status == "failed"
        assert message in outcome.steps["msa"].error

    def test_a_protein_that_is_too_short_is_named(self, tmp_path):
        outcome = run({"proteins": query_file(tmp_path, {"tiny": "MKV"})}, tmp_path)
        assert "tiny: Query requires 10–5000" in outcome.steps["msa"].error

    def test_more_than_eight_proteins_are_refused(self, tmp_path):
        many = {f"p{i}": protein(i, 30) for i in range(9)}
        outcome = run({"proteins": query_file(tmp_path, many)}, tmp_path)
        assert "At most 8 proteins" in outcome.steps["msa"].error

    def test_a_database_that_is_not_installed_says_how_to_get_it(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_RESOURCE_HOME", str(tmp_path / "nowhere"))
        monkeypatch.delenv("ALAMBIQ_RESOURCE_UNIREF50_2026_02", raising=False)
        outcome = run({"proteins": query_file(tmp_path)}, tmp_path)
        assert outcome.steps["msa"].status == "failed"
        assert "uniref50-2026_02 is not at" in outcome.steps["msa"].error


def uniref_installed() -> bool:
    from alambiq.core import resources

    return resources.location("uniref50-2026_02").is_dir()


@pytest.mark.skipif(
    not (uniref_installed() and os.environ.get("ALAMBIQ_TEST_REAL_UNIREF50")),
    reason="set ALAMBIQ_TEST_REAL_UNIREF50=1 with UniRef50 installed to run the full search (many minutes)",
)
class TestRealUniRef50:
    LYSOZYME = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"

    def test_a_textbook_protein_gets_a_deep_diverse_alignment(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", MMSEQS or sys.executable)
        outcome = run_ok(
            {"proteins": query_file(tmp_path, {"lysozyme": self.LYSOZYME})}, tmp_path
        )
        chain = outcome.outputs("msa")["chains"][0]
        # Lysozyme C is one of the most densely sampled protein families.
        assert chain["decision"] == "generated"
        assert chain["unique_sequences"] > 200
        assert rows(outcome)[0][1] == self.LYSOZYME
        assert outcome.outputs("msa")["database_version"] == "2026_02"
