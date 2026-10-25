"""enzyme.retrieve_candidates and enzyme.characterize_candidates, end to end.

The panel is built so every expectation follows from how it was made, not from
running the code:

    R1      a 120-residue protein, the literature reference
    R2      R1 with a third of its residues changed, none at positions 29-31
            or 74-76: a distant relative that keeps the two features
    close   R1 with one change at position 5: the same protein
    mutated R1 with the residue at feature position 30 changed
    partial the first 90 residues of R1: covers 75 % of the reference
    gapped  R1 with residues 74-76 removed: feature 75 falls in the deletion
    far     an unrelated 120-residue protein
    N1      another unrelated 120-residue protein, the hard negative

The two features are R1's residues at positions 30 and 75.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from pathlib import Path

import pytest

AMINO = "ACDEFGHIKLMNPQRSTVWY"


def random_protein(seed: int, length: int = 120) -> str:
    return "".join(random.Random(seed).choices(AMINO, k=length))


R1 = random_protein(1)


def change(sequence: str, position: int) -> str:
    """Replace the residue at a one-based position with a different one."""
    old = sequence[position - 1]
    new = "W" if old != "W" else "F"
    return sequence[: position - 1] + new + sequence[position:]


def distant(sequence: str, seed: int, keep: set[int]) -> str:
    rng = random.Random(seed)
    out = list(sequence)
    for index in range(len(out)):
        if index + 1 not in keep and rng.random() < 0.33:
            out[index] = rng.choice([a for a in AMINO if a != out[index]])
    return "".join(out)


KEEP = {29, 30, 31, 74, 75, 76}
R2 = distant(R1, 2, KEEP)
REFERENCES = {"R1": R1, "R2": R2}
CANDIDATES = {
    "close": change(R1, 5),
    "mutated": change(R1, 30),
    "partial": R1[:90],
    "gapped": R1[:73] + R1[76:],
    "far": random_protein(3),
    "N1": random_protein(4),
}
FEATURES = [
    {"position": 30, "residue": R1[29], "role": "nucleophile", "citation": "Test 2020"},
    {"position": 75, "residue": R1[74], "role": "proton donor", "citation": "Test 2020"},
]


# A configured ALAMBIQ_ENV_MMSEQS runs these tests in that real engine
# environment; otherwise the interpreter running the tests stands in for it.
CONFIGURED = os.environ.get("ALAMBIQ_ENV_MMSEQS", "")


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", CONFIGURED or sys.executable)


def write_fasta(tmp_path: Path, name: str, records: dict[str, str]) -> str:
    path = tmp_path / name
    path.write_text("".join(f">{k}\n{v}\n" for k, v in records.items()))
    return str(path)


def retrieve_step(tmp_path: Path, **params) -> dict:
    return {
        "id": "retrieve",
        "capability": "enzyme.retrieve_candidates",
        "params": {
            "references": write_fasta(tmp_path, "refs.faa", params.pop("references", REFERENCES)),
            "candidates": write_fasta(tmp_path, "cand.faa", params.pop("candidates", CANDIDATES)),
            "positive_controls": ["R1", "R2"],
            "negative_controls": ["N1"],
            **params,
        },
    }


def characterize_step(**params) -> dict:
    return {
        "id": "characterize",
        "capability": "enzyme.characterize_candidates",
        "bind": {"retrieval": "retrieve.record"},
        "params": {"reference_id": "R1", "features": FEATURES, **params},
    }


def run(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "enzyme", "steps": steps}, run_dir=tmp_path / name)


def run_ok(steps: list[dict], tmp_path: Path, name: str = "run"):
    outcome = run(steps, tmp_path, name)
    assert outcome.ok, [x.error for x in outcome.steps.values() if x.error]
    return outcome


needs_mmseqs = pytest.mark.skipif(
    not (os.environ.get("ALAMBIQ_ENV_MMSEQS") or shutil.which("mmseqs")),
    reason="set ALAMBIQ_ENV_MMSEQS or put mmseqs on PATH to run the search",
)

@needs_mmseqs
class TestRetrieval:
    @pytest.fixture
    def outcome(self, tmp_path):
        return run_ok([retrieve_step(tmp_path)], tmp_path)

    def test_every_protein_is_ranked_once(self, outcome):
        ranked = outcome.outputs("retrieve")["ranked"]
        assert sorted(r["identifier"] for r in ranked) == sorted(REFERENCES | CANDIDATES)
        assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))

    def test_a_near_copy_of_a_reference_ranks_above_everything_unrelated(self, outcome):
        ranked = outcome.outputs("retrieve")["ranked"]
        order = [r["identifier"] for r in ranked]
        by_id = {r["identifier"]: r for r in ranked}
        assert by_id["close"]["best_reference"] == "R1"
        assert by_id["close"]["identity"] > 0.98
        # The two unrelated proteins are the bottom two.
        assert set(order[-2:]) == {"far", "N1"}
        assert order.index("close") < order.index("R2")

    def test_the_classes_follow_from_the_controls_named(self, outcome):
        by_id = {r["identifier"]: r for r in outcome.outputs("retrieve")["ranked"]}
        assert by_id["R1"]["evidence_class"] == "positive-control"
        assert by_id["N1"]["evidence_class"] == "hard-negative-control"
        assert by_id["close"]["evidence_class"] == "unresolved-candidate"

    def test_a_reference_is_not_scored_against_itself(self, outcome):
        by_id = {r["identifier"]: r for r in outcome.outputs("retrieve")["ranked"]}
        assert by_id["R1"]["best_reference"] == "R2"
        assert by_id["R2"]["best_reference"] == "R1"

    def test_controls_that_separate_cleanly_say_so(self, outcome):
        out = outcome.outputs("retrieve")
        assert out["positive_control_recall"] == 1.0
        assert out["all_hard_negatives_below_positives"] is True
        assert out["control_auroc"] == 1.0
        assert out["hard_negative_margin_bits"] == pytest.approx(
            out["minimum_positive_bits"] - out["maximum_hard_negative_bits"], abs=1e-6
        )
        assert out["hard_negative_margin_bits"] > 0

    def test_the_length_only_baseline_cannot_separate_equal_length_proteins(self, outcome):
        out = outcome.outputs("retrieve")
        # Every control is 120 residues, so length ties every comparison.
        assert out["length_only_baseline_auroc"] == 0.5
        assert out["control_auroc_lift_over_length"] == 0.5

    def test_a_hard_negative_that_is_a_near_copy_defeats_the_controls_and_is_flagged(
        self, tmp_path
    ):
        candidates = {**CANDIDATES, "decoy": change(R1, 7)}
        outcome = run_ok(
            [retrieve_step(tmp_path, candidates=candidates, negative_controls=["decoy"])],
            tmp_path,
        )
        out = outcome.outputs("retrieve")
        # The decoy aligns to R1 almost perfectly; R1 and R2 only to each other.
        assert out["all_hard_negatives_below_positives"] is False
        assert out["control_auroc"] == 0.0
        assert out["control_auroc_lift_over_length"] < 0
        notes = " ".join(outcome.steps["retrieve"].notes)
        assert "as high as the lowest positive" in notes
        assert "no better than protein length" in notes

    def test_a_positive_control_below_the_bit_threshold_is_not_recovered(self, tmp_path):
        outcome = run_ok([retrieve_step(tmp_path, positive_min_bits=10_000)], tmp_path)
        assert outcome.outputs("retrieve")["positive_control_recall"] == 0.0
        assert "of the positive controls reached 10000 bits" in " ".join(
            outcome.steps["retrieve"].notes
        )

    def test_a_strict_evalue_leaves_unrelated_proteins_without_any_alignment(self, tmp_path):
        outcome = run_ok([retrieve_step(tmp_path, evalue=1e-5)], tmp_path)
        by_id = {r["identifier"]: r for r in outcome.outputs("retrieve")["ranked"]}
        for name in ("far", "N1"):
            assert by_id[name]["bits"] == 0.0
            assert by_id[name]["best_reference"] is None
        assert by_id["close"]["best_reference"] == "R1"

    def test_the_report_gives_the_controls(self, tmp_path):
        outcome = run_ok([retrieve_step(tmp_path)], tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "AUROC" in page
        assert "protein length alone" in page


class TestRetrievalRefusals:
    @pytest.mark.parametrize(
        ("change_to", "message"),
        [
            ({"references": {"R1": R1}}, "at least two proteins"),
            ({"candidates": {**CANDIDATES, "R1": R1}}, "overlap"),
            ({"positive_controls": ["close"]}, "must be one of the references"),
            ({"negative_controls": ["R1"]}, "must be one of the candidates"),
            ({"positive_controls": []}, "name at least one positive"),
            ({"negative_controls": []}, "name at least one positive"),
            ({"positive_controls": ["R1", "R1"]}, "named twice"),
            ({"candidates": {"bad/id": R1[:20], "N1": R1[:20]}}, "must use only letters"),
            ({"candidates": {"X1": "MK*V", "N1": R1[:20]}}, "uppercase amino-acid"),
            ({"sensitivity": 0}, "sensitivity must be between 1 and 7.5"),
            ({"sensitivity": 0.9}, "sensitivity must be between 1 and 7.5"),
            ({"sensitivity": 7.6}, "sensitivity must be between 1 and 7.5"),
            ({"sensitivity": float("nan")}, "sensitivity must be between 1 and 7.5"),
            ({"evalue": 0}, "evalue must be positive"),
            ({"positive_min_bits": -1}, "must not be negative"),
        ],
    )
    def test_the_request_is_refused_in_plain_words(self, tmp_path, change_to, message):
        outcome = run([retrieve_step(tmp_path, **change_to)], tmp_path)
        assert outcome.steps["retrieve"].status == "failed"
        assert message in outcome.steps["retrieve"].error


@needs_mmseqs
class TestCharacterization:
    @pytest.fixture
    def outcome(self, tmp_path):
        return run_ok([retrieve_step(tmp_path), characterize_step()], tmp_path)

    @pytest.fixture
    def decisions(self, outcome):
        return {a["identifier"]: a for a in outcome.outputs("characterize")["assessments"]}

    def test_a_near_copy_that_keeps_both_features_advances(self, decisions):
        close = decisions["close"]
        assert close["decision"] == "advance"
        assert close["reason_codes"] == ["retrieval-and-literature-features-compatible"]
        assert (close["mapped_feature_fraction"], close["conserved_feature_fraction"]) == (1.0, 1.0)

    def test_a_substituted_feature_residue_is_rejected_as_not_conserved(self, decisions):
        mutated = decisions["mutated"]
        assert mutated["decision"] == "reject"
        assert mutated["reason_codes"] == ["literature-features-not-conserved"]
        assert mutated["conserved_feature_fraction"] == 0.5
        first = mutated["residues"][0]
        assert (first["reference_position"], first["expected_residue"]) == (30, R1[29])
        assert first["observed_residue"] == "W" and first["conserved"] is False
        assert mutated["residues"][1]["conserved"] is True

    def test_a_fragment_covering_too_little_of_the_reference_is_withheld(self, decisions):
        partial = decisions["partial"]
        assert partial["decision"] == "withhold"
        assert partial["reason_codes"] == ["reference-coverage-outside-applicability"]
        assert partial["query_coverage"] == pytest.approx(0.75, abs=0.02)
        # Both features lie inside the first 90 residues, so both mapped.
        assert partial["mapped_feature_fraction"] == 1.0

    def test_a_feature_in_a_deletion_is_unmapped_and_the_protein_withheld(self, decisions):
        gapped = decisions["gapped"]
        assert gapped["decision"] == "withhold"
        assert gapped["reason_codes"] == ["literature-features-not-fully-mapped"]
        assert gapped["mapped_feature_fraction"] == 0.5
        deleted = gapped["residues"][1]
        assert (deleted["reference_position"], deleted["target_position"]) == (75, None)
        assert deleted["observed_residue"] is None

    def test_a_position_after_a_deletion_is_renumbered_in_the_candidate(self, decisions):
        # Feature 30 is before the deletion, so it keeps its number.
        assert decisions["gapped"]["residues"][0]["target_position"] == 30

    def test_unrelated_proteins_without_an_alignment_are_rejected_for_that_reason_alone(
        self, decisions
    ):
        for name in ("far", "N1"):
            assert decisions[name]["decision"] == "reject"
            assert decisions[name]["reason_codes"] == ["no-reference-alignment"]

    def test_the_references_are_judged_with_the_candidates(self, decisions):
        # R1 against itself keeps both features; R2 was built to keep them.
        for name in ("R1", "R2"):
            assert decisions[name]["decision"] == "advance"

    def test_the_controls_show_the_policy_behaves(self, outcome):
        out = outcome.outputs("characterize")
        assert out["positive_control_advance_rate"] == 1.0
        assert out["negative_control_rejection_rate"] == 1.0
        assert out["unresolved_candidates_advanced"] == 1
        assert out["advanced"] == ["R1", "R2", "close"]
        assert out["withheld"] == ["gapped", "partial"]
        assert out["rejected"] == ["N1", "far", "mutated"]

    def test_a_looser_policy_changes_the_decision_it_governs(self, tmp_path):
        outcome = run_ok(
            [retrieve_step(tmp_path), characterize_step(minimum_conserved_feature_fraction=0.5)],
            tmp_path,
        )
        decisions = {a["identifier"]: a["decision"] for a in outcome.outputs("characterize")["assessments"]}
        # Half the features are conserved, which now meets the bar.
        assert decisions["mutated"] == "advance"

    def test_allowing_an_unmapped_feature_turns_a_withheld_protein_into_a_rejected_one(
        self, tmp_path
    ):
        outcome = run_ok(
            [retrieve_step(tmp_path), characterize_step(minimum_mapped_feature_fraction=0.5)],
            tmp_path,
        )
        gapped = {a["identifier"]: a for a in outcome.outputs("characterize")["assessments"]}["gapped"]
        # Half the features are now allowed to be unmapped, but an unmapped
        # feature is not a conserved one, and the default requires all conserved.
        assert gapped["decision"] == "reject"
        assert gapped["reason_codes"] == ["literature-features-not-conserved"]

    def test_a_strict_evalue_leaves_unrelated_proteins_with_no_alignment_to_the_reference(
        self, tmp_path
    ):
        outcome = run_ok(
            [retrieve_step(tmp_path), characterize_step(evalue=1e-5)], tmp_path
        )
        far = {a["identifier"]: a for a in outcome.outputs("characterize")["assessments"]}["far"]
        assert far["reason_codes"] == ["no-reference-alignment"]
        assert (far["identity"], far["query_coverage"]) == (0.0, 0.0)
        assert all(r["target_position"] is None for r in far["residues"])

    def test_a_policy_that_lets_a_known_negative_through_is_flagged(self, tmp_path):
        outcome = run_ok(
            [
                retrieve_step(tmp_path, negative_controls=["mutated"]),
                characterize_step(minimum_conserved_feature_fraction=0.5),
            ],
            tmp_path,
        )
        assert outcome.outputs("characterize")["negative_control_rejection_rate"] == 0.0
        assert "of the hard negatives were rejected" in " ".join(
            outcome.steps["characterize"].notes
        )

    def test_the_stored_record_holds_every_decision_and_the_features(self, outcome):
        record = json.loads(Path(outcome.outputs("characterize")["record"]).read_text())
        assert len(record["assessments"]) == 8
        assert [f["position"] for f in record["features"]] == [30, 75]
        assert record["engine_versions"]["mmseqs"]

    def test_the_report_lists_each_residue(self, tmp_path):
        outcome = run_ok([retrieve_step(tmp_path), characterize_step()], tmp_path)
        page = outcome.report(tmp_path / "report.html").read_text()
        assert "mutated: residues" in page


@needs_mmseqs
class TestCharacterizationRefusals:
    @pytest.mark.parametrize(
        ("params", "message"),
        [
            ({"reference_id": "close"}, "must be one of the references"),
            (
                {"features": [{"position": 30, "residue": "W", "role": "x", "citation": "y"}]},
                "not W",
            ),
            ({"features": [{"position": 500, "residue": "A", "role": "x", "citation": "y"}]}, "no residue"),
            ({"features": FEATURES + FEATURES[:1]}, "must be unique"),
            ({"features": []}, "at least one reference feature"),
            (
                {"features": [{"position": 30, "residue": R1[29], "role": "x", "citation": " "}]},
                "needs a role and the citation",
            ),
            ({"minimum_query_coverage": 1.5}, "between 0 and 1"),
            ({"minimum_retrieval_bits": -5}, "must not be negative"),
            ({"sensitivity": 0.5}, "sensitivity must be between 1 and 7.5"),
            ({"sensitivity": 8}, "sensitivity must be between 1 and 7.5"),
            ({"evalue": 0}, "evalue must be positive"),
        ],
    )
    def test_a_request_that_cannot_be_trusted_is_refused(self, tmp_path, params, message):
        outcome = run([retrieve_step(tmp_path), characterize_step(**params)], tmp_path)
        assert outcome.steps["characterize"].status == "failed"
        assert message in outcome.steps["characterize"].error
