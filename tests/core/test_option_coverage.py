"""Request options that no other test ran, each with an answer known without the code.

Every field of every request is run by at least one test, and the conformance
suite enforces it. These cover the engineering, construct, recoding and
operation options that were declared but never exercised. Writing them found a
real defect: an unachievable primer setting crashed the assembly with an
internal error instead of saying what could not be done.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
from importlib.util import find_spec
from pathlib import Path

import pytest

from core.test_construct_capabilities import PROTEIN, PROTEIN_DNA, VECTOR, fasta
from core.test_engineering_capabilities import panel_candidates, response

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "operation"
HEAD = (
    "measurement_id,sample_id,condition_id,metric,value,unit,"
    "independent_id,technical_id,role,status\n"
)
SELECT_FIT = {"metric": "activity", "condition_id": "standard", "unit": "umol/min/mg"}

needs_sklearn = pytest.mark.skipif(not find_spec("sklearn"), reason="needs scikit-learn")
needs_scipy = pytest.mark.skipif(not find_spec("scipy"), reason="needs scipy")
needs_pydna = pytest.mark.skipif(not find_spec("pydna"), reason="needs the construct extra")
needs_chisel = pytest.mark.skipif(not find_spec("dnachisel"), reason="needs dnachisel")


def run(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    return execute({"id": "options", "steps": steps}, run_dir=tmp_path / name)


def run_ok(steps: list[dict], tmp_path: Path, name: str = "run"):
    outcome = run(steps, tmp_path, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def stored(outcome, step: str) -> dict:
    return json.loads(Path(outcome.outputs(step)["record"]).read_text())


# ----------------------------------------------------------------- candidate.fit


def write_panel(tmp_path: Path, candidates: list[dict]) -> Path:
    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"schema_version": "0.1", "candidates": candidates}))
    return path


def write_assay(tmp_path: Path, candidates: list[dict], units: int = 0) -> Path:
    """Measurements for every non-pool candidate; `units` > 0 adds replicates."""
    from alambiq.capabilities.measure.science.inspection import inspect_assay

    factors = [0.95, 1.0, 1.05][:units]  # their mean is exactly 1
    rows = []
    for c in candidates:
        if c["partition"] == "pool":
            continue
        value = response(c["sequence"].count("A"))
        if units:
            rows += [
                f"m-{c['sample_id']}-{u},{c['sample_id']},standard,activity,"
                f"{value * f},umol/min/mg,u{u},t0,sample,measured\n"
                for u, f in enumerate(factors)
            ]
        else:
            rows.append(
                f"m-{c['sample_id']},{c['sample_id']},standard,activity,{value},"
                "umol/min/mg,,,sample,measured\n"
            )
    record = inspect_assay(
        HEAD + "".join(rows),
        {"standard": "Synthetic standard condition"},
        "preparation" if units else None,
    )
    path = tmp_path / ("assay-reps.json" if units else "assay.json")
    path.write_text(json.dumps(record))
    return path


def write_embeddings(tmp_path: Path, candidates: list[dict], vector) -> Path:
    path = tmp_path / "embeddings.json"
    path.write_text(
        json.dumps(
            {
                "model": "synthetic",
                "revision": "test",
                "layer": "none",
                "pooling": "none",
                "input_tracks": ["sequence"],
                "dimension": 2,
                "proteins": [
                    {
                        "id": c["sample_id"],
                        "sequence": c["sequence"],
                        "pooled_vector": vector(c),
                    }
                    for c in candidates
                ],
            }
        )
    )
    return path


def alanine_vector(candidate: dict) -> list[float]:
    """Two coordinates carrying the alanine fraction; never the zero vector."""
    fraction = candidate["sequence"].count("A") / 60
    return [fraction, 1.5 - fraction]


def fit_step(panel: Path, assay: Path, **params) -> dict:
    return {
        "id": "model",
        "capability": "candidate.fit",
        "params": {"panel": str(panel), "assay": str(assay), **SELECT_FIT, **params},
    }


@needs_sklearn
class TestCandidateFitOptions:
    @pytest.fixture
    def candidates(self) -> list[dict]:
        return panel_candidates()

    def test_embeddings_replace_the_sequence_composition_as_features(
        self, candidates, tmp_path
    ):
        panel = write_panel(tmp_path, candidates)
        assay = write_assay(tmp_path, candidates)
        embeddings = write_embeddings(tmp_path, candidates, alanine_vector)
        outcome = run_ok(
            [
                fit_step(
                    panel, assay, features="pooled-embeddings", embeddings=str(embeddings)
                )
            ],
            tmp_path,
        )
        out = outcome.outputs("model")
        # The response is linear in the first coordinate, so the fit is near exact.
        assert out["model"]["rmse"] < 1.5
        assert out["model"]["rmse"] < out["baseline"]["rmse"] / 10
        representation = stored(outcome, "model")["source"]["representation"]
        assert representation["kind"] == "pooled-embeddings"
        assert representation["model"] == "synthetic"

    def test_sequence_composition_can_be_named_explicitly(self, candidates, tmp_path):
        panel = write_panel(tmp_path, candidates)
        assay = write_assay(tmp_path, candidates)
        outcome = run_ok(
            [fit_step(panel, assay, features="sequence-composition-v1")], tmp_path
        )
        kind = stored(outcome, "model")["source"]["representation"]["kind"]
        assert kind == "sequence-composition-v1"

    def test_embeddings_are_required_by_embedding_features_and_only_by_them(
        self, candidates, tmp_path
    ):
        panel = write_panel(tmp_path, candidates)
        assay = write_assay(tmp_path, candidates)
        embeddings = write_embeddings(tmp_path, candidates, alanine_vector)
        missing = run([fit_step(panel, assay, features="pooled-embeddings")], tmp_path, "a")
        assert missing.steps["model"].status == "failed"
        assert "embeddings" in missing.steps["model"].error
        extra = run([fit_step(panel, assay, embeddings=str(embeddings))], tmp_path, "b")
        assert extra.steps["model"].status == "failed"
        assert "embeddings" in extra.steps["model"].error

    def test_replicates_are_fitted_through_their_unit_means(self, candidates, tmp_path):
        panel = write_panel(tmp_path, candidates)
        replicated = write_assay(tmp_path, candidates, units=3)
        outcome = run_ok(
            [fit_step(panel, replicated, observation_basis="independent-means")],
            tmp_path,
        )
        # The three units average to the true value exactly, so the fit is as
        # good as one made on the values themselves.
        assert outcome.outputs("model")["model"]["rmse"] < 1.0

    def test_replicated_measurements_need_the_independent_means_basis(
        self, candidates, tmp_path
    ):
        panel = write_panel(tmp_path, candidates)
        replicated = write_assay(tmp_path, candidates, units=3)
        outcome = run([fit_step(panel, replicated)], tmp_path)
        assert outcome.steps["model"].status == "failed"
        assert "independent-means" in outcome.steps["model"].error


# --------------------------------------------------------------- candidate.select

BASE = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"
NEIGHBOUR = "G" + BASE[1:]  # one position different from BASE
STRANGER = "W" * len(BASE)  # different at nearly every position


def selection_panel() -> list[dict]:
    rows = [
        ("t1", "ACACACACACACACACACACACACACACAC", "train", "g1"),
        ("t2", "DEDEDEDEDEDEDEDEDEDEDEDEDEDEDE", "train", "g2"),
        ("p1", BASE, "pool", "gp1"),
        ("p2", NEIGHBOUR, "pool", "gp2"),
        ("p3", STRANGER, "pool", "gp3"),
    ]
    return [
        {
            "sample_id": name,
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "group_id": group,
            "partition": partition,
        }
        for name, sequence, partition, group in rows
    ]


def select_step(panel: Path, **params) -> dict:
    return {
        "id": "pick",
        "capability": "candidate.select",
        "params": {"panel": str(panel), "count": 2, "seed": 3, "strategy": "random", **params},
    }


def chosen(outcome) -> set[str]:
    return {c["sample_id"] for c in outcome.outputs("pick")["selected"]}


class TestCandidateSelectOptions:
    @pytest.fixture
    def panel(self, tmp_path) -> Path:
        return write_panel(tmp_path, selection_panel())

    def test_only_eligible_candidates_are_chosen(self, panel, tmp_path):
        outcome = run_ok([select_step(panel, eligible_ids=["p1", "p3"])], tmp_path)
        # Two places, two eligible candidates: exactly those.
        assert chosen(outcome) == {"p1", "p3"}
        assert outcome.outputs("pick")["reasons"]["researcher-eligibility"] == 1

    def test_excluded_candidates_are_never_chosen(self, panel, tmp_path):
        outcome = run_ok([select_step(panel, exclude_ids=["p3"])], tmp_path)
        assert chosen(outcome) == {"p1", "p2"}

    def test_an_identifier_that_is_not_in_the_panel_is_refused(self, panel, tmp_path):
        outcome = run([select_step(panel, exclude_ids=["nope"])], tmp_path)
        assert outcome.steps["pick"].status == "failed"
        assert "distinct sample_id values" in outcome.steps["pick"].error

    @pytest.mark.parametrize("seed", range(8))
    def test_diverse_selection_by_sequence_never_takes_both_near_duplicates(
        self, panel, tmp_path, seed
    ):
        outcome = run_ok(
            [select_step(panel, strategy="diverse", diversity="sequence", seed=seed)],
            tmp_path,
            f"s{seed}",
        )
        picked = chosen(outcome)
        # p1 and p2 differ at one position, p3 at nearly all: a diverse pair
        # always includes the stranger and keeps at most one of the neighbours.
        assert "p3" in picked
        assert not {"p1", "p2"} <= picked

    @pytest.mark.parametrize("seed", range(8))
    def test_diverse_selection_by_embedding_follows_the_vectors(
        self, panel, tmp_path, seed
    ):
        vectors = {
            "t1": [0.5, 0.5], "t2": [0.6, 0.4],
            "p1": [1.0, 0.0], "p2": [0.99, 0.1], "p3": [0.0, 1.0],
        }
        embeddings = write_embeddings(
            tmp_path, selection_panel(), lambda c: vectors[c["sample_id"]]
        )
        outcome = run_ok(
            [
                select_step(
                    panel,
                    strategy="diverse",
                    diversity="embedding-cosine",
                    embeddings=str(embeddings),
                    seed=seed,
                )
            ],
            tmp_path,
            f"e{seed}",
        )
        picked = chosen(outcome)
        # By cosine, p1 and p2 point almost the same way and p3 is orthogonal.
        assert "p3" in picked and not {"p1", "p2"} <= picked

    def test_diversity_must_be_stated_for_a_diverse_strategy(self, panel, tmp_path):
        outcome = run([select_step(panel, strategy="diverse")], tmp_path)
        assert outcome.steps["pick"].status == "failed"
        assert "sequence or embedding-cosine" in outcome.steps["pick"].error

    def test_embeddings_belong_only_to_embedding_diversity(self, panel, tmp_path):
        embeddings = write_embeddings(tmp_path, selection_panel(), lambda c: [1.0, 1.0])
        outcome = run([select_step(panel, embeddings=str(embeddings))], tmp_path)
        assert outcome.steps["pick"].status == "failed"
        assert "are accepted only with the strategies diverse" in outcome.steps["pick"].error or (
            "are accepted only with the strategies diverse" in outcome.steps["pick"].error
        )


# -------------------------------------------------------------- construct.assemble


def assemble_step(tmp_path: Path, **params) -> dict:
    return {
        "id": "build",
        "capability": "construct.assemble",
        "params": {
            "vector": str(VECTOR),
            "insert": fasta(tmp_path, "insert.fa", "insert", PROTEIN_DNA),
            "expected_protein": fasta(tmp_path, "insert.faa", "insert", PROTEIN),
            **params,
        },
    }


def cassette_text() -> str:
    """A linear GenBank cassette: noncoding flanks around the coding sequence."""
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from Bio.SeqRecord import SeqRecord

    prefix, suffix = "GACCTAGACGGTACGCTTTAGCGTC", "GCTGACTGACGTATCCTACGGTCA"
    record = SeqRecord(Seq(prefix + PROTEIN_DNA + suffix), id="cassette", name="cassette")
    record.annotations = {"molecule_type": "DNA", "topology": "linear", "date": "01-JAN-2000"}
    end = len(prefix) + len(PROTEIN_DNA)
    record.features = [
        SeqFeature(SimpleLocation(0, len(prefix), strand=1), type="misc_feature",
                   qualifiers={"label": ["upstream context"]}),
        SeqFeature(SimpleLocation(len(prefix), end, strand=1), type="CDS",
                   qualifiers={"label": ["test CDS"], "codon_start": ["1"],
                               "transl_table": ["11"], "translation": [PROTEIN]}),
        SeqFeature(SimpleLocation(end, len(record.seq), strand=1), type="misc_feature",
                   qualifiers={"label": ["downstream context"]}),
    ]
    stream = io.StringIO()
    SeqIO.write(record, stream, "genbank")
    return stream.getvalue()


@needs_pydna
class TestAssembleOptions:
    def labels(self, outcome) -> list[str]:
        return [
            f["qualifiers"].get("label", [""])[0] for f in stored(outcome, "build")["features"]
        ]

    def span(self, outcome, label: str) -> tuple[int, int]:
        """One-based, inclusive start and end of the named feature in the product."""
        (feature,) = [
            f for f in stored(outcome, "build")["features"]
            if f["qualifiers"].get("label", [""])[0] == label
        ]
        (part,) = feature["parts"]
        return part["start"], part["end"]

    def test_an_annotated_cassette_brings_its_own_features(self, tmp_path):
        path = tmp_path / "cassette.gb"
        path.write_text(cassette_text())
        outcome = run_ok(
            [assemble_step(tmp_path, insert=str(path), annotated_cassette=True)], tmp_path
        )
        # The whole cassette, flanks included, is placed into the vector: the
        # vector, a 25 base upstream flank, the 63 base coding sequence and a
        # 24 base downstream flank.
        from Bio import SeqIO

        vector = len(SeqIO.read(VECTOR, "genbank"))
        assert outcome.outputs("build")["length"] == vector + 25 + len(PROTEIN_DNA) + 24
        assert "upstream context" in self.labels(outcome)
        # The coding sequence sits after the vector and the 25 base flank, and
        # spans exactly its 63 bases.
        assert self.span(outcome, "test CDS") == (vector + 25 + 1, vector + 25 + 63)

    def test_the_cassette_preset_is_the_same_as_the_flag(self, tmp_path):
        path = tmp_path / "cassette.gb"
        path.write_text(cassette_text())
        with_flag = run_ok(
            [assemble_step(tmp_path, insert=str(path), annotated_cassette=True)], tmp_path, "a"
        )
        step = assemble_step(tmp_path, insert=str(path))
        step["preset"] = "blunt-cassette"
        with_preset = run_ok([step], tmp_path, "b")
        assert with_preset.outputs("build")["length"] == with_flag.outputs("build")["length"]
        assert with_preset.outputs("build")["sequence_sha256"] == with_flag.outputs("build")["sequence_sha256"]

    def test_features_named_for_a_plain_insert_appear_in_the_product(self, tmp_path):
        features = [{"label": "my coding region", "start": 1, "end": 63}]
        outcome = run_ok([assemble_step(tmp_path, insert_features=features)], tmp_path)
        from Bio import SeqIO

        vector = len(SeqIO.read(VECTOR, "genbank"))
        # Annotated at the insert's own coordinates, which start right after the vector.
        assert self.span(outcome, "my coding region") == (vector + 1, vector + 63)
        plain = run_ok([assemble_step(tmp_path)], tmp_path, "plain")
        assert outcome.outputs("build")["features"] == plain.outputs("build")["features"] + 1
        assert "my coding region" not in self.labels(plain)

    @pytest.mark.parametrize("overlap", [25, 35, 60])
    def test_each_primer_tail_is_as_long_as_the_overlap(self, overlap, tmp_path):
        outcome = run_ok([assemble_step(tmp_path, overlap=overlap)], tmp_path, f"o{overlap}")
        tails = [len(p["sequence"]) - len(p["footprint"]) for p in outcome.outputs("build")["primers"]]
        assert tails == [overlap, overlap]

    def test_a_higher_minimum_lengthens_the_annealing_footprints(self, tmp_path):
        default = run_ok([assemble_step(tmp_path)], tmp_path, "d")
        longer = run_ok([assemble_step(tmp_path, primer_min=25)], tmp_path, "l")
        assert min(len(p["footprint"]) for p in default.outputs("build")["primers"]) < 25
        assert min(len(p["footprint"]) for p in longer.outputs("build")["primers"]) >= 25

    @pytest.mark.parametrize("target", [65.0, 70.0])
    def test_footprint_melting_temperatures_follow_the_target(self, target, tmp_path):
        outcome = run_ok([assemble_step(tmp_path, target_tm=target)], tmp_path, f"t{target}")
        for primer in outcome.outputs("build")["primers"]:
            assert primer["footprint_tm_degC"] == pytest.approx(target, abs=1.0)

    def test_settings_that_cannot_be_met_are_refused_in_plain_words(self, tmp_path):
        for params in ({"primer_min": 30}, {"target_tm": 75.0}):
            outcome = run([assemble_step(tmp_path, **params)], tmp_path, str(params))
            assert outcome.steps["build"].status == "failed"
            error = outcome.steps["build"].error
            assert "could be designed" in error and "NoneType" not in error


# -------------------------------------------------------------- dna.optimize_coding


@needs_chisel
class TestRecodingWindow:
    def test_every_window_ends_inside_the_gc_bounds(self, tmp_path):
        from Bio.Seq import Seq

        rng = random.Random(5)
        codons = [a + b + c for a in "ACGT" for b in "ACGT" for c in "ACGT"
                  if str(Seq(a + b + c).translate()) != "*"]
        dna = "ATG" + "".join(rng.choice(codons) for _ in range(498)) + "TAA"
        protein = str(Seq(dna).translate(table=11, cds=True))
        coding, expected = fasta(tmp_path, "c.fa", "c", dna), fasta(tmp_path, "c.faa", "c", protein)
        outcome = run_ok(
            [
                {
                    "id": "design",
                    "capability": "dna.optimize_coding",
                    "params": {
                        "coding_dna": coding, "expected_protein": expected, "seed": 7,
                        "gc_min": 0.4, "gc_max": 0.6, "gc_window": 60,
                    },
                }
            ],
            tmp_path,
        )
        out = outcome.outputs("design")
        assert out["status"] == "completed"
        # The random sequence drifts outside the bounds somewhere; the result must not.
        assert out["before"]["gc_violations"] > 0 and out["after"]["gc_violations"] == 0

        after = stored(outcome, "design")["dna_after"]
        # Checked here independently of the solver's own report.
        for start in range(len(after) - 59):
            window = after[start : start + 60]
            gc = (window.count("G") + window.count("C")) / 60
            assert 0.4 <= gc <= 0.6, (start, gc)
        assert str(Seq(after).translate(table=11, cds=True)) == protein


# ----------------------------------------------------------------------- operation


@needs_scipy
class TestOperationSolvers:
    S0, K, T = 0.01, 0.2, 10.0

    @pytest.mark.parametrize("solver", ["Radau", "DOP853"])
    def test_either_integrator_reproduces_first_order_decay(self, solver, tmp_path):
        outcome = run_ok(
            [
                {
                    "id": "s",
                    "capability": "operation.simulate",
                    "params": {"definition": str(FIXTURES / "first-order.json"), "solver": solver},
                }
            ],
            tmp_path,
            solver,
        )
        final = outcome.outputs("s")["final_concentrations_M"]["S"]
        assert final == pytest.approx(self.S0 * math.exp(-self.K * self.T), rel=1e-5)

    def test_the_fit_recovers_the_rate_with_the_explicit_integrator(self, tmp_path):
        outcome = run_ok(
            [
                {
                    "id": "f",
                    "capability": "operation.fit",
                    "params": {"definition": str(FIXTURES / "first-order-fit.json"), "solver": "DOP853"},
                }
            ],
            tmp_path,
        )
        (capacity,) = outcome.outputs("f")["parameters"]
        assert capacity["value"] == pytest.approx(self.K * self.S0, rel=1e-4)

    def test_an_integrator_that_is_not_offered_is_refused(self, tmp_path):
        outcome = run(
            [
                {
                    "id": "s",
                    "capability": "operation.simulate",
                    "params": {"definition": str(FIXTURES / "first-order.json"), "solver": "LSODA"},
                }
            ],
            tmp_path,
        )
        assert outcome.steps["s"].status == "failed"
        assert "'Radau'" in outcome.steps["s"].error
