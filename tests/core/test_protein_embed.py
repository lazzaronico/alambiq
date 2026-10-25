"""protein.embed with the real ESM C 300M model, on the CPU.

A language model has no answer known in advance, so these tests check what must
hold whatever the numbers are: the vectors have the model's shape, the pooled
vector is the mean of the residue vectors, a protein embeds the same alone and
in a panel, a one-residue change moves a protein less than an unrelated protein
does, and the record names the model and what ran. One regression anchor pins
the first components for ubiquitin so an unintended change is noticed; it was
taken from a verified run and is not an independent truth.

They run where the pinned esm environment and the verified checkpoint exist:
set ALAMBIQ_ENV_ESM to that environment's Python (the first run hashes the
1.3 GB checkpoint).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

ENV = os.environ.get("ALAMBIQ_ENV_ESM", "")


def installed() -> bool:
    from alambiq.core import resources

    return (
        bool(ENV) and Path(ENV).is_file() and resources.location("esmc-300m").is_dir()
    )


pytestmark = pytest.mark.skipif(
    not installed(),
    reason="set ALAMBIQ_ENV_ESM and install the esmc-300m resource to run",
)

UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
VARIANT = UBIQUITIN[:20] + "A" + UBIQUITIN[21:]  # residue 21 changed to alanine
LYSOZYME = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"


def fasta(tmp_path: Path, records: dict[str, str], name: str = "p.faa") -> str:
    path = tmp_path / name
    path.write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))
    return str(path)


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "embed", "capability": "protein.embed", "params": params}
    return execute({"id": "e", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


def stored(outcome) -> dict:
    return json.loads(Path(outcome.outputs("embed")["record"]).read_text())


def vector(outcome, name: str) -> list[float]:
    return next(
        p["pooled_vector"] for p in stored(outcome)["proteins"] if p["id"] == name
    )


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.hypot(*a) * math.hypot(*b)
    )


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_ESM", ENV)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("embed")
    records = {"ubiquitin": UBIQUITIN, "variant": VARIANT, "lysozyme": LYSOZYME}
    os.environ["ALAMBIQ_ENV_ESM"] = ENV
    outcome = run_ok({"proteins": fasta(tmp_path, records)}, tmp_path)
    return outcome, tmp_path


class TestModelOutput:
    def test_the_record_names_the_model_and_what_ran(self, panel):
        outcome, _ = panel
        out = outcome.outputs("embed")
        assert out["model"] == "biohub/ESMC-300M"
        assert out["revision"] == "a59b831785f907e96e6a246b1d142bfb76df31ee"
        assert out["license"] == "MIT-with-third-party-notices"
        assert (out["dimension"], out["device"]) == (960, "cpu")
        record = stored(outcome)
        assert (
            record["weights_sha256"]
            == "0772d8fe64bb25e14fe6f23b80e3c9a7d215d0da3c6cba5bd356d7c0e0bb22cc"
        )
        assert record["layer"] == "last-hidden-state-post-norm"
        assert record["pooling"] == "mean-of-residues"
        assert record["input_tracks"] == ["sequence"]
        assert set(out["engine_versions"]) == {"esm", "torch", "transformers", "numpy"}

    def test_each_protein_has_a_vector_of_the_models_width(self, panel):
        outcome, _ = panel
        assert [p["id"] for p in stored(outcome)["proteins"]] == [
            "ubiquitin",
            "variant",
            "lysozyme",
        ]
        for p in stored(outcome)["proteins"]:
            assert len(p["pooled_vector"]) == 960
            assert all(math.isfinite(v) for v in p["pooled_vector"])

    def test_the_pooled_vector_is_the_mean_of_the_residue_vectors(self, panel):
        import numpy as np

        outcome, _ = panel
        archive = np.load(outcome.outputs("embed")["residue_embeddings"])
        for p in stored(outcome)["proteins"]:
            residues = archive[p["residue_array"]]
            # One 960-vector per residue, with the boundary tokens left out.
            assert residues.shape == (len(p["sequence"]), 960)
            assert np.allclose(residues.mean(axis=0), p["pooled_vector"], atol=1e-5)

    def test_a_one_residue_change_moves_a_protein_less_than_an_unrelated_protein(
        self, panel
    ):
        outcome, _ = panel
        original = vector(outcome, "ubiquitin")
        assert cosine(original, vector(outcome, "variant")) > cosine(
            original, vector(outcome, "lysozyme")
        )

    def test_the_sequence_hash_is_of_the_sequence(self, panel):
        import hashlib

        outcome, _ = panel
        for p in stored(outcome)["proteins"]:
            assert (
                p["sequence_sha256"]
                == hashlib.sha256(p["sequence"].encode()).hexdigest()
            )

    def test_regression_anchor_for_ubiquitin(self, panel):
        """Pinned from the first verified run (CPU, esm 3.4.0, torch 2.11.0); not independent truth."""
        outcome, _ = panel
        first = vector(outcome, "ubiquitin")[:3]
        norm = math.hypot(*vector(outcome, "ubiquitin"))
        assert first == pytest.approx(ANCHOR_FIRST_THREE, abs=1e-3)
        assert norm == pytest.approx(ANCHOR_NORM, rel=1e-3)

    def test_the_report_names_the_model(self, panel):
        outcome, tmp_path = panel
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "biohub/ESMC-300M" in page


class TestBatchingAndReuse:
    def test_a_protein_embeds_the_same_alone_as_in_a_panel(self, panel, tmp_path):
        outcome, _ = panel
        alone = run_ok(
            {"proteins": fasta(tmp_path, {"ubiquitin": UBIQUITIN})}, tmp_path
        )
        a, b = vector(alone, "ubiquitin"), vector(outcome, "ubiquitin")
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 1e-5

    def test_the_thread_count_does_not_change_the_answer_beyond_rounding(
        self, panel, tmp_path
    ):
        outcome, _ = panel
        threaded = run_ok(
            {"proteins": fasta(tmp_path, {"ubiquitin": UBIQUITIN}), "threads": 1},
            tmp_path,
        )
        a, b = vector(threaded, "ubiquitin"), vector(outcome, "ubiquitin")
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) < 1e-4

    def test_running_the_same_step_again_reuses_it(self, tmp_path):
        params = {"proteins": fasta(tmp_path, {"ubiquitin": UBIQUITIN})}
        first = run_ok(params, tmp_path, "same")
        second = run_ok(params, tmp_path, "same")
        assert (first.steps["embed"].status, second.steps["embed"].status) == (
            "completed",
            "reused",
        )
        assert vector(first, "ubiquitin") == vector(second, "ubiquitin")

    def test_changing_the_sequence_runs_it_again(self, tmp_path):
        run_ok(
            {"proteins": fasta(tmp_path, {"ubiquitin": UBIQUITIN})}, tmp_path, "same"
        )
        changed = run_ok(
            {"proteins": fasta(tmp_path, {"ubiquitin": VARIANT})}, tmp_path, "same"
        )
        assert changed.steps["embed"].status == "completed"


class TestChain:
    def test_embeddings_feed_the_explorer_and_the_selector_reads_their_format(
        self, tmp_path
    ):
        from alambiq.api import run as execute

        records = {"ubiquitin": UBIQUITIN, "variant": VARIANT, "lysozyme": LYSOZYME}
        outcome = execute(
            {
                "id": "chain",
                "steps": [
                    {
                        "id": "embed",
                        "capability": "protein.embed",
                        "params": {"proteins": fasta(tmp_path, records)},
                    },
                    {
                        "id": "explore",
                        "capability": "protein.explore_embeddings",
                        "bind": {"embeddings": "embed.record"},
                        "params": {"representatives": 2},
                    },
                ],
            },
            run_dir=tmp_path / "chain",
        )
        assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
        explored = outcome.outputs("explore")
        assert explored["model"] == "biohub/ESMC-300M"
        nearest = {n["id"]: n["neighbour"] for n in explored["nearest"]}
        assert nearest["ubiquitin"] == "variant" and nearest["variant"] == "ubiquitin"
        # Two representatives from a panel with one outlier: the outlier is one of them.
        assert "lysozyme" in explored["recommended"]


class TestRefusals:
    def test_a_protein_over_the_length_limit_is_named(self, tmp_path):
        outcome = run({"proteins": fasta(tmp_path, {"long": "A" * 1025})}, tmp_path)
        assert "long: expected 1–1024" in outcome.steps["embed"].error

    def test_a_character_that_is_not_a_residue_is_refused(self, tmp_path):
        outcome = run({"proteins": fasta(tmp_path, {"bad": "MKV*"})}, tmp_path)
        assert outcome.steps["embed"].status == "failed"

    def test_more_than_sixty_four_proteins_are_refused(self, tmp_path):
        many = {f"p{i}": "MKVLA" for i in range(65)}
        outcome = run({"proteins": fasta(tmp_path, many)}, tmp_path)
        assert "At most 64 proteins" in outcome.steps["embed"].error

    def test_a_thread_count_out_of_range_is_refused(self, tmp_path):
        outcome = run(
            {"proteins": fasta(tmp_path, {"p": "MKVLAAGG"}), "threads": 0}, tmp_path
        )
        assert "threads must be an integer from 1 to 32" in outcome.steps["embed"].error

    def test_an_unavailable_device_is_an_error_not_a_fallback(self, tmp_path):
        outcome = run(
            {"proteins": fasta(tmp_path, {"p": "MKVLAAGG"}), "device": "cuda"}, tmp_path
        )
        assert outcome.steps["embed"].status == "failed"
        assert (
            "CUDA is unavailable; no implicit device fallback"
            in outcome.steps["embed"].error
        )

    def test_a_model_that_is_not_offered_is_refused(self, tmp_path):
        outcome = run(
            {"proteins": fasta(tmp_path, {"p": "MKVLAAGG"}), "model": "esm2-650m"},
            tmp_path,
        )
        assert outcome.steps["embed"].status == "failed"
        assert "model" in outcome.steps["embed"].error


ANCHOR_FIRST_THREE = [-0.001842, -0.017289, 0.019839]
ANCHOR_NORM = 0.951085
