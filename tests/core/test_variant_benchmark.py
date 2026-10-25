"""The variant-proposal benchmark: its statistics, its stored results, and its reproducibility.

The metrics are checked against independent calculations (scipy's rank correlation, an
enrichment worked out by hand, the spread of a random ranking), the committed results against
their own arithmetic and against the table on the methods page, and, where the esm environment
is configured, a small dataset is scored twice through the real model and must give the same
numbers.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/benchmark_variant_proposal.py"
RESULTS = ROOT / "docs/methods/variant-proposal-benchmark.json"
PAGE = ROOT / "docs/methods/variant-proposal.md"


def tool():
    spec = importlib.util.spec_from_file_location("benchmark_variant_proposal", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench():
    return tool()


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text())


class TestStatistics:
    def test_the_rank_correlation_is_scipys(self, bench):
        scipy_stats = pytest.importorskip("scipy.stats")
        rng = random.Random(3)
        a = [rng.gauss(0, 1) for _ in range(60)]
        b = [x + rng.gauss(0, 1) for x in a]
        assert bench.spearman(a, b) == pytest.approx(
            scipy_stats.spearmanr(a, b)[0], abs=1e-12
        )

    def test_ties_share_their_mean_rank(self, bench):
        scipy_stats = pytest.importorskip("scipy.stats")
        a = [1, 1, 2, 3, 3, 3, 4]
        b = [2, 1, 1, 4, 5, 3, 6]
        assert bench.spearman(a, b) == pytest.approx(
            scipy_stats.spearmanr(a, b)[0], abs=1e-12
        )

    def test_a_perfect_ranking_is_one_and_a_reversed_one_minus_one(self, bench):
        x = list(range(20))
        assert bench.spearman(x, x) == pytest.approx(1.0)
        assert bench.spearman(x, x[::-1]) == pytest.approx(-1.0)

    def test_enrichment_counts_the_best_quarter_among_the_best_tenth(self, bench):
        measured = [float(i) for i in range(100)]  # the best quarter is 75..99
        perfect = bench.enrichment(measured, measured)
        assert perfect == {"top": 10, "hits": 10, "fraction": 1.0, "enrichment": 4.0}
        worst = bench.enrichment([-m for m in measured], measured)
        assert worst["hits"] == 0 and worst["enrichment"] == 0.0

    def test_the_top_tenth_is_at_least_five_variants(self, bench):
        measured = [float(i) for i in range(20)]
        assert bench.enrichment(measured, measured)["top"] == 5

    def test_the_bootstrap_interval_contains_the_estimate(self, bench):
        rng = random.Random(4)
        a = [rng.gauss(0, 1) for _ in range(80)]
        b = [x + rng.gauss(0, 1.5) for x in a]
        low, high = bench.bootstrap_interval(a, b)
        assert low < bench.spearman(a, b) < high and high - low < 0.5

    def test_a_random_ranking_has_no_correlation_and_the_expected_spread(self, bench):
        dataset = {
            "higher_is_better": True,
            "variants": [{"value": float(i)} for i in range(100)],
        }
        out = bench.random_baseline(dataset, shuffles=400, seed=5)
        assert abs(out["spearman"]) < 0.02
        assert out["spearman_sd"] == pytest.approx(1 / math.sqrt(99), rel=0.15)
        assert out["top_tenth"]["enrichment"] == pytest.approx(1.0, abs=0.15)

    def test_blosum_prefers_similar_residues(self, bench):
        matrix = bench.blosum()
        assert matrix[("I", "V")] > matrix[("I", "D")]
        assert all(matrix[(a, b)] == matrix[(b, a)] for a, b in matrix)
        assert matrix[("W", "W")] == 11


class TestStoredResults:
    def test_three_or_more_datasets_each_with_the_model_and_both_baselines(
        self, results
    ):
        assert len(results["datasets"]) >= 3
        for dataset in results["datasets"]:
            names = [m["method"] for m in dataset["methods"]]
            assert names == [
                "ESM C 300M log-likelihood ratio",
                "BLOSUM62 score of the substitution",
                "random ranking",
            ]
            assert dataset["variants"] > 20 and dataset["source"]

    def test_each_estimate_lies_inside_its_own_interval(self, results):
        for dataset in results["datasets"]:
            for method in dataset["methods"]:
                if "spearman_95_interval" in method:
                    low, high = method["spearman_95_interval"]
                    assert -1 <= low <= method["spearman"] <= high <= 1

    def test_the_random_ranking_sits_at_zero_within_its_spread(self, results):
        for dataset in results["datasets"]:
            random_row = dataset["methods"][2]
            assert abs(random_row["spearman"]) < 3 * random_row["spearman_sd"]
            assert random_row["top_tenth"]["enrichment"] == pytest.approx(1.0, abs=0.1)

    def test_the_table_on_the_methods_page_is_the_one_these_results_give(
        self, bench, results
    ):
        assert bench.markdown_table(results) in PAGE.read_text()

    def test_the_page_says_what_the_numbers_do(self, results):
        text = " ".join(PAGE.read_text().split())
        by = {d["id"]: d["methods"][0]["spearman"] for d in results["datasets"]}
        assert by["atta-transaminase"] < 0 and "inverted" in text
        assert all(
            by[k] > 0.3
            for k in ("bslipa-t50", "glucokinase-activity", "glucokinase-abundance")
        )
        assert "The results apply to the proteins and properties of the datasets in the table" in text


ENV = os.environ.get("ALAMBIQ_ENV_ESM", "")


def esm_ready() -> bool:
    from alambiq.core import resources

    return (
        bool(ENV) and Path(ENV).is_file() and resources.location("esmc-300m").is_dir()
    )


@pytest.mark.skipif(
    not esm_ready(),
    reason="set ALAMBIQ_ENV_ESM and install esmc-300m to score a dataset",
)
class TestReproducibility:
    def test_a_dataset_scored_twice_gives_identical_results(
        self, bench, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ALAMBIQ_ENV_ESM", ENV)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        pytest.importorskip("subprocess")
        import subprocess

        outputs = []
        for name in ("a", "b"):
            out = tmp_path / f"{name}.json"
            done = subprocess.run(
                [
                    os.sys.executable,
                    str(TOOL),
                    "--studies",
                    os.environ.get(
                        "ALAMBIQ_STUDIES", str(ROOT.parent / "alambiq-studies")
                    ),
                    "--out",
                    str(out),
                    "--only",
                    "atta-transaminase",
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            assert done.returncode == 0, done.stderr[-2000:]
            outputs.append(json.loads(out.read_text()))
        assert outputs[0] == outputs[1]
        stored = next(
            d
            for d in json.loads(RESULTS.read_text())["datasets"]
            if d["id"] == "atta-transaminase"
        )
        again = outputs[0]["datasets"][0]
        for one, other in zip(stored["methods"], again["methods"], strict=True):
            assert one["spearman"] == pytest.approx(other["spearman"], abs=5e-4)
