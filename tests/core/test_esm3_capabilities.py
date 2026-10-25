"""protein.predict_tracks, protein.complete and ESM3 embeddings, with the real model.

Ubiquitin is the reference protein: its secondary structure is known from the
crystal structure (1UBQ): strands at 1-7, 10-17, 40-45, 48-49 and 66-72, one
alpha helix at 23-34, and a flexible C-terminal tail. A prediction from sequence
alone should recover the helix and the main strands, and ubiquitin is the
textbook ubiquitin-like domain. Beyond that, a language model has no answer
known in advance, so the completion tests check what must hold whatever the
model writes: the prompt is what the regions imply, fixed residues are never
changed, a seed reproduces its proposal, and every rejection has a reason.

They run where the pinned esm environment and the verified ESM3-small checkpoint
exist: set ALAMBIQ_ENV_ESM to that environment's Python (the first run hashes
the 4.2 GB checkpoint).
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

import pytest

from alambiq.capabilities.characterize.science.completion import (
    build_prompt,
    check_candidate,
)

ENV = os.environ.get("ALAMBIQ_ENV_ESM", "")
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def installed() -> bool:
    from alambiq.core import resources

    return (
        bool(ENV) and Path(ENV).is_file() and resources.location("esm3-small").is_dir()
    )


needs_model = pytest.mark.skipif(
    not installed(),
    reason="set ALAMBIQ_ENV_ESM and install the esm3-small resource to run",
)


def fasta(tmp_path: Path, records: dict[str, str], name: str = "p.faa") -> str:
    path = tmp_path / name
    path.write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))
    return str(path)


def run(capability: str, params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "s", "capability": capability, "params": params}
    return execute({"id": "c", "steps": [step]}, run_dir=tmp_path / name)


def run_ok(capability: str, params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(capability, params, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    if ENV:
        monkeypatch.setenv("ALAMBIQ_ENV_ESM", ENV)


# --- the independent checks need no model ----------------------------------


class TestPrompt:
    def test_the_regions_become_underscores_exactly_where_they_say(self):
        built = build_prompt(UBIQUITIN, [(2, 4), (40, 44)], None)
        assert (
            built["prompt"]
            == UBIQUITIN[0] + "___" + UBIQUITIN[4:39] + "_____" + UBIQUITIN[44:]
        )
        assert built["mutable_positions"] == [2, 3, 4, 40, 41, 42, 43, 44]

    def test_the_default_number_of_steps_is_eight_or_the_masked_count(self):
        assert build_prompt(UBIQUITIN, [(2, 4)], None)["num_steps"] == 3
        assert build_prompt(UBIQUITIN, [(2, 30)], None)["num_steps"] == 8
        assert build_prompt(UBIQUITIN, [(2, 30)], 12)["num_steps"] == 12

    @pytest.mark.parametrize(
        ("regions", "steps", "message"),
        [
            ([], None, "at least one region"),
            ([(0, 3)], None, "must lie within"),
            ([(70, 80)], None, "must lie within"),
            ([(5, 3)], None, "must lie within"),
            ([(2, 6), (6, 9)], None, "must not overlap"),
            ([(1, 76)], None, "at least one residue outside the regions"),
            ([(2, 4)], 4, "num_steps"),
            ([(2, 4)], 0, "num_steps"),
        ],
    )
    def test_a_request_the_model_should_never_see_is_refused(
        self, regions, steps, message
    ):
        with pytest.raises(ValueError, match=message):
            build_prompt(UBIQUITIN, regions, steps)

    def test_an_ambiguous_residue_is_refused(self):
        with pytest.raises(ValueError, match="unambiguous"):
            build_prompt("MKVX" * 5, [(2, 3)], None)


class TestIndependentChecks:
    mutable = [2, 3, 4]
    sequence = "MKVLAAGG"

    def check(self, generated):
        return check_candidate(generated, self.sequence, self.mutable, 7, "proposal-1")

    def test_a_proposal_that_changed_only_masked_positions_is_accepted_with_its_changes(
        self,
    ):
        accepted = self.check("MAVLAAGG")
        assert accepted["status"] == "completed"
        assert accepted["mutations"] == [{"position": 2, "before": "K", "after": "A"}]
        assert accepted["unchanged_from_input"] is False

    def test_a_proposal_that_rewrote_nothing_is_flagged(self):
        assert self.check(self.sequence)["unchanged_from_input"] is True

    @pytest.mark.parametrize(
        ("generated", "reason"),
        [
            ("MKVLAAGA", "changed a fixed residue"),
            ("MKVLAAG", "length differs"),
            ("MK_LAAGG", "missing, masked or nonstandard"),
            ("MKXLAAGG", "missing, masked or nonstandard"),
            (None, "missing, masked or nonstandard"),
        ],
    )
    def test_a_proposal_that_broke_a_rule_is_rejected_with_the_reason(
        self, generated, reason
    ):
        rejected = self.check(generated)
        assert rejected["status"] == "rejected" and reason in rejected["reason"]

    def test_a_rejected_fixed_residue_change_still_lists_what_changed(self):
        assert self.check("MKVLAAGA")["mutations"][0]["position"] == 8


# --- predicted tracks -------------------------------------------------------


@needs_model
class TestTracks:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("tracks")
        os.environ["ALAMBIQ_ENV_ESM"] = ENV
        return run_ok(
            "protein.predict_tracks",
            {"protein": fasta(tmp_path, {"ubiquitin": UBIQUITIN})},
            tmp_path,
        ), tmp_path

    def labels(self, outcome) -> str:
        return outcome[0].outputs("s")["secondary_structure"]

    def test_there_is_one_label_per_residue_from_the_eight_classes(self, outcome):
        out = outcome[0].outputs("s")
        assert len(out["secondary_structure"]) == 76 == out["length"]
        assert set(out["secondary_structure"]) <= set("GHITEBSC")
        assert sum(out["secondary_counts"].values()) == 76

    def test_ubiquitins_one_alpha_helix_is_found_where_the_crystal_structure_has_it(
        self, outcome
    ):
        helix = self.labels(outcome)[22:34]  # residues 23-34
        assert helix.count("H") >= 10, helix

    def test_the_main_strands_are_found(self, outcome):
        labels = self.labels(outcome)
        assert labels[0:7].count("E") >= 5, labels[0:7]  # strand 1-7
        assert labels[9:17].count("E") >= 5, labels[9:17]  # strand 10-17
        assert labels[65:72].count("E") >= 5, labels[65:72]  # strand 66-72

    def test_the_flexible_tail_is_not_called_helix_or_strand(self, outcome):
        assert not set(self.labels(outcome)[72:]) & set("HE")

    def test_ubiquitin_is_called_a_ubiquitin_like_domain_over_most_of_its_length(
        self, outcome
    ):
        regions = outcome[0].outputs("s")["regions"]
        domain = [r for r in regions if r["label"] == "IPR000626"]
        assert domain and domain[0]["end"] - domain[0]["start"] >= 60
        assert all(1 <= r["start"] <= r["end"] <= 76 for r in regions)

    def test_the_probability_arrays_agree_with_the_labels(self, outcome):
        import numpy as np

        result, _ = outcome
        arrays = np.load(result.outputs("s")["arrays"])
        probabilities = arrays["secondary_structure_probability"]
        assert probabilities.shape == (76, 8)
        assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        assert "".join(
            "GHITEBSC"[i] for i in probabilities.argmax(axis=1)
        ) == self.labels(outcome)
        record = json.loads(Path(result.outputs("s")["record"]).read_text())
        assert np.allclose(
            record["secondary_structure"]["max_probability"], probabilities.max(axis=1)
        )

    def test_the_record_says_it_is_a_prediction_from_sequence_alone(self, outcome):
        record = json.loads(Path(outcome[0].outputs("s")["record"]).read_text())
        assert record["origin"] == "model-predicted" and record["input_tracks"] == [
            "sequence"
        ]
        assert record["model"] == "biohub/esm3-sm-open-v1"
        assert record["decoding"]["interpro_threshold"] == 0.1
        assert record["coordinate_system"] == "one-based-inclusive"
        assert len(record["resource_identity"]) == 64

    def test_the_same_protein_gives_the_same_tracks_again(self, outcome, tmp_path):
        again = run_ok(
            "protein.predict_tracks",
            {"protein": fasta(tmp_path, {"ubiquitin": UBIQUITIN})},
            tmp_path,
        )
        assert again.outputs("s")["secondary_structure"] == self.labels(outcome)

    def test_the_report_lists_the_predicted_regions(self, outcome):
        result, tmp_path = outcome
        page = result.report(tmp_path / "report.html").read_text()
        assert "IPR000626" in page

    def test_a_protein_over_the_length_limit_is_refused(self, tmp_path):
        outcome = run(
            "protein.predict_tracks",
            {"protein": fasta(tmp_path, {"long": "A" * 513})},
            tmp_path,
        )
        assert "1–512 protein residues" in outcome.steps["s"].error

    def test_a_thread_count_out_of_range_is_refused(self, tmp_path):
        outcome = run(
            "protein.predict_tracks",
            {"protein": fasta(tmp_path, {"p": UBIQUITIN}), "threads": 99},
            tmp_path,
        )
        assert "threads must be an integer from 1 to 32" in outcome.steps["s"].error

    def test_an_unavailable_device_is_an_error_not_a_fallback(self, tmp_path):
        outcome = run(
            "protein.predict_tracks",
            {"protein": fasta(tmp_path, {"p": UBIQUITIN}), "device": "cuda"},
            tmp_path,
        )
        assert (
            "CUDA is unavailable; no implicit device fallback"
            in outcome.steps["s"].error
        )


# --- completion -------------------------------------------------------------

DESIGNED = "".join(random.Random(5).choices(AMINO, k=80))
REGIONS = [{"start": 20, "end": 35}]


@needs_model
class TestComplete:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("complete")
        os.environ["ALAMBIQ_ENV_ESM"] = ENV
        params = {
            "protein": fasta(tmp_path, {"designed": DESIGNED}),
            "regions": REGIONS,
            "seed": 11,
            "candidates": 4,
        }
        return run_ok("protein.complete", params, tmp_path), tmp_path, params

    def test_the_prompt_is_the_input_with_the_masked_positions_blanked(self, outcome):
        out = outcome[0].outputs("s")
        assert out["prompt"] == DESIGNED[:19] + "_" * 16 + DESIGNED[35:]
        assert out["masked"] == 16 and out["length"] == 80

    def test_each_proposal_has_its_own_seed_counting_up_from_the_requested_one(
        self, outcome
    ):
        proposals = outcome[0].outputs("s")["proposals"]
        assert [p["seed"] for p in proposals] == [11, 12, 13, 14]
        assert [p["id"] for p in proposals] == [f"proposal-{i}" for i in range(1, 5)]

    def test_no_accepted_proposal_changed_a_fixed_residue(self, outcome):
        accepted = [
            p
            for p in outcome[0].outputs("s")["proposals"]
            if p["status"] == "completed"
        ]
        assert accepted
        for p in accepted:
            assert len(p["sequence"]) == 80
            assert (
                p["sequence"][:19] == DESIGNED[:19]
                and p["sequence"][35:] == DESIGNED[35:]
            )
            assert all(20 <= m["position"] <= 35 for m in p["mutations"])
            assert p["sequence"][19:35] != "_" * 16

    def test_a_model_asked_to_invent_sixteen_residues_of_a_random_protein_changes_some(
        self, outcome
    ):
        accepted = [
            p
            for p in outcome[0].outputs("s")["proposals"]
            if p["status"] == "completed"
        ]
        assert any(p["mutations"] for p in accepted)

    def test_the_accepted_proposals_come_back_as_a_fasta(self, outcome):
        out = outcome[0].outputs("s")
        text = Path(out["fasta"]).read_text()
        accepted = [p for p in out["proposals"] if p["status"] == "completed"]
        assert text == "".join(f">{p['id']}\n{p['sequence']}\n" for p in accepted)
        assert out["accepted"] == len(accepted)

    def test_a_repeat_of_an_earlier_proposal_is_named(self, outcome):
        proposals = outcome[0].outputs("s")["proposals"]
        seen: dict[str, str] = {}
        for p in proposals:
            if p["status"] != "completed":
                continue
            assert p["duplicate_of"] == seen.get(p["sequence"])
            seen.setdefault(p["sequence"], p["id"])

    def test_the_same_seed_reproduces_the_same_proposals(self, outcome, tmp_path):
        _, _, params = outcome
        again = run_ok(
            "protein.complete",
            {**params, "protein": fasta(tmp_path, {"designed": DESIGNED})},
            tmp_path,
        )
        first = [p["sequence"] for p in outcome[0].outputs("s")["proposals"]]
        assert [p["sequence"] for p in again.outputs("s")["proposals"]] == first

    def test_the_record_names_the_model_and_the_generation_settings(self, outcome):
        record = json.loads(Path(outcome[0].outputs("s")["record"]).read_text())
        assert record["model"] == "biohub/esm3-sm-open-v1"
        assert record["generation"]["num_steps"] == 8
        assert record["input"]["mutable_positions"] == list(range(20, 36))
        assert len(record["resource_identity"]) == 64

    def test_a_protein_the_model_knows_by_heart_is_restored_and_the_repeat_is_flagged(
        self, tmp_path
    ):
        outcome = run_ok(
            "protein.complete",
            {
                "protein": fasta(tmp_path, {"ubiquitin": UBIQUITIN}),
                "regions": [{"start": 2, "end": 4}],
                "seed": 1,
                "candidates": 3,
            },
            tmp_path,
        )
        notes = " ".join(outcome.steps["s"].notes)
        proposals = outcome.outputs("s")["proposals"]
        # Observed with this checkpoint: ESM3 restores a famous protein's masked residues.
        # Whatever it writes, anything that rewrote nothing must be flagged as such.
        for p in proposals:
            if p["status"] == "completed" and not p["mutations"]:
                assert p["unchanged_from_input"] is True
        if any(p["unchanged_from_input"] for p in proposals):
            assert "changed no residue" in notes

    def test_the_accepted_proposals_feed_an_esm3_embedding(self, tmp_path):
        from alambiq.api import run as execute

        params = {
            "protein": fasta(tmp_path, {"designed": DESIGNED}),
            "regions": REGIONS,
            "seed": 11,
            "candidates": 2,
        }
        outcome = execute(
            {
                "id": "chain",
                "steps": [
                    {
                        "id": "complete",
                        "capability": "protein.complete",
                        "params": params,
                    },
                    {
                        "id": "embed",
                        "capability": "protein.embed",
                        "bind": {"proteins": "complete.fasta"},
                        "params": {"model": "esm3-small"},
                    },
                ],
            },
            run_dir=tmp_path / "chain",
        )
        assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
        embedded = outcome.outputs("embed")
        assert (
            embedded["model"] == "biohub/esm3-sm-open-v1"
            and embedded["dimension"] == 1536
        )
        assert len(embedded["proteins"]) == outcome.outputs("complete")["accepted"]

    def test_the_report_shows_the_prompt(self, outcome):
        result, tmp_path, _ = outcome
        page = result.report(tmp_path / "report.html").read_text()
        assert "_" * 16 in page


class TestCompleteRefusals:
    @pytest.fixture(autouse=True)
    def stand_in_engine(self, monkeypatch):
        # A refusal happens before the model loads, so any interpreter will do; without
        # one configured the step would fail on the missing environment instead.
        monkeypatch.setenv("ALAMBIQ_ENV_ESM", ENV or sys.executable)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"regions": []}, "at least one region"),
            ({"regions": [{"start": 1, "end": 76}]}, "at least one residue outside the regions"),
            (
                {"regions": [{"start": 2, "end": 9}, {"start": 8, "end": 12}]},
                "must not overlap",
            ),
            ({"regions": [{"start": 70, "end": 90}]}, "must lie within"),
            ({"candidates": 9}, "candidates must be an integer from 1 to 8"),
            ({"candidates": 0}, "candidates must be an integer from 1 to 8"),
            ({"seed": -1}, "seed must be an integer"),
            ({"temperature": 0}, "temperature must be above 0"),
            ({"temperature": 2.5}, "temperature must be above 0"),
            ({"top_p": 1.5}, "top_p must be above 0"),
            ({"threads": 0}, "threads must be an integer from 1 to 32"),
            ({"num_steps": 20}, "num_steps"),
        ],
    )
    def test_a_request_the_model_should_never_see_is_refused_before_it_loads(
        self, tmp_path, changes, message
    ):
        params = {
            "protein": fasta(tmp_path, {"ubiquitin": UBIQUITIN}),
            "regions": [{"start": 2, "end": 9}],
            "seed": 1,
            **changes,
        }
        outcome = run("protein.complete", params, tmp_path)
        assert outcome.steps["s"].status == "failed"
        assert message in outcome.steps["s"].error


# --- ESM3 embeddings --------------------------------------------------------


@needs_model
class TestEsm3Embedding:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("esm3embed")
        os.environ["ALAMBIQ_ENV_ESM"] = ENV
        variant = UBIQUITIN[:20] + "A" + UBIQUITIN[21:]
        records = {"ubiquitin": UBIQUITIN, "variant": variant, "designed": DESIGNED}
        return run_ok(
            "protein.embed",
            {"proteins": fasta(tmp_path, records), "model": "esm3-small"},
            tmp_path,
        )

    def record(self, outcome) -> dict:
        return json.loads(Path(outcome.outputs("s")["record"]).read_text())

    def test_the_record_names_this_model_and_its_own_layer_and_width(self, outcome):
        record = self.record(outcome)
        assert record["model"] == "biohub/esm3-sm-open-v1"
        assert record["layer"] == "last-hidden-state-pre-final-norm"
        assert record["dimension"] == 1536
        assert (
            record["weights_sha256"]
            == "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9"
        )
        assert all(len(p["pooled_vector"]) == 1536 for p in record["proteins"])

    def test_the_pooled_vector_is_the_mean_of_the_residue_vectors(self, outcome):
        import numpy as np

        archive = np.load(outcome.outputs("s")["residue_embeddings"])
        for p in self.record(outcome)["proteins"]:
            residues = archive[p["residue_array"]]
            assert residues.shape == (len(p["sequence"]), 1536)
            assert np.allclose(residues.mean(axis=0), p["pooled_vector"], atol=1e-5)

    def test_a_one_residue_change_moves_a_protein_less_than_an_unrelated_one(
        self, outcome
    ):
        by_id = {p["id"]: p["pooled_vector"] for p in self.record(outcome)["proteins"]}

        def cosine(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True)) / (
                math.hypot(*a) * math.hypot(*b)
            )

        assert cosine(by_id["ubiquitin"], by_id["variant"]) > cosine(
            by_id["ubiquitin"], by_id["designed"]
        )

    def test_a_protein_over_the_esm3_limit_is_refused_though_esmc_would_accept_it(
        self, tmp_path
    ):
        outcome = run(
            "protein.embed",
            {"proteins": fasta(tmp_path, {"long": "A" * 600}), "model": "esm3-small"},
            tmp_path,
        )
        assert (
            "expected 1–512 ungapped protein residues for esm3-small"
            in outcome.steps["s"].error
        )

    def test_vectors_from_the_two_models_are_refused_in_one_exploration(
        self, outcome, tmp_path
    ):
        esmc_env = os.environ.get("ALAMBIQ_ENV_ESM")
        assert esmc_env
        esmc = run_ok(
            "protein.embed",
            {
                "proteins": fasta(tmp_path, {"other": UBIQUITIN[:40]}),
                "model": "esmc-300m",
            },
            tmp_path,
            "esmc",
        )
        from alambiq.api import run as execute

        explored = execute(
            {
                "id": "x",
                "steps": [
                    {
                        "id": "explore",
                        "capability": "protein.explore_embeddings",
                        "params": {
                            "embeddings": outcome.outputs("s")["record"],
                            "more_embeddings": [esmc.outputs("s")["record"]],
                        },
                    }
                ],
            },
            run_dir=tmp_path / "x",
        )
        assert "different representation space" in explored.steps["explore"].error
        assert "model" in explored.steps["explore"].error
