"""reaction.explore_routes with the real DORAnet backend, on cases solved by hand.

The base case is methyl acetate plus water, one operator that hydrolyses an ester:
there is exactly one route, one step, giving acetic acid and methanol.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("doranet")
pytest.importorskip("rdkit")

HYDROLYSIS_SMARTS = "[C:1](=[O:2])[O:3][C:4].[OH2:5]>>[C:1](=[O:2])[OH:5].[OH:3][C:4]"


def hydrolysis() -> dict:
    return {
        "id": "ester-conversion",
        "title": "Methyl acetate hydrolysis",
        "feedstocks": [
            {"id": "ester", "smiles": "COC(C)=O", "amount": 1, "role": "substrate"},
            {"id": "water", "smiles": "O", "amount": 1, "role": "cosubstrate"},
        ],
        "target": {"smiles": "CC(=O)O", "amount": 1},
        "extent_per_step": 1,
        "recover_feedstocks": [],
        "operators": [
            {
                "id": "ester-hydrolysis",
                "name": "Ester hydrolysis",
                "smarts": HYDROLYSIS_SMARTS,
                "source": {
                    "kind": "researcher-specified",
                    "citation": "Explicit ester + water hydrolysis operator",
                },
                "catalyst_requirements": [
                    "A carboxylesterase accepting the supplied ester"
                ],
                "enzyme_evidence": [],
            }
        ],
        "bounds": {
            "generations": 2,
            "max_recipes": 30,
            "max_products_per_recipe": 20,
            "max_nodes": 60,
            "max_edges": 60,
            "max_route_steps": 4,
            "max_route_states": 500,
            "max_routes": 10,
            "max_atoms": 120,
        },
    }


def run(params: dict, tmp_path: Path, name: str = "run"):
    from alambiq.api import run as execute

    step = {"id": "routes", "capability": "reaction.explore_routes", "params": params}
    return execute({"id": "r", "steps": [step]}, run_dir=tmp_path / name)


def explore(params: dict, tmp_path: Path, name: str = "run"):
    outcome = run(params, tmp_path, name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def out(outcome) -> dict:
    return outcome.outputs("routes")


def record(outcome) -> dict:
    return json.loads(Path(out(outcome)["record"]).read_text())


def by_smiles(outcome, inventory: dict[str, str]) -> dict[str, str]:
    names = {m["chemical_id"]: m["canonical_smiles"] for m in out(outcome)["molecules"]}
    return {names[k]: v for k, v in inventory.items()}


class TestHydrolysis:
    def test_the_one_route_is_one_step_with_the_net_change_worked_out_by_hand(
        self, tmp_path
    ):
        outcome = explore(hydrolysis(), tmp_path)
        assert out(outcome)["target_status"] == "inventory-route-found"
        assert len(out(outcome)["routes"]) == 1
        route = out(outcome)["routes"][0]
        assert len(route["steps"]) == 1 and route["target_amount"] == "1"
        assert by_smiles(outcome, route["net_stoichiometry"]) == {
            "COC(C)=O": "-1",
            "O": "-1",
            "CC(=O)O": "1",
            "CO": "1",
        }
        assert route["balance_residuals"] == {}
        final = record(outcome)["routes"][0]["final_inventory"]
        assert by_smiles(outcome, final) == {"CC(=O)O": "1", "CO": "1"}

    def test_the_reaction_keeps_its_operator_and_catalyst_requirement(self, tmp_path):
        reaction = out(explore(hydrolysis(), tmp_path))["reactions"][0]
        assert reaction["operator_id"] == "ester-hydrolysis"
        assert reaction["catalyst_requirements"] == [
            "A carboxylesterase accepting the supplied ester"
        ]
        assert reaction["enzyme_references"] == 0

    def test_the_search_is_reproducible(self, tmp_path):
        first = record(explore(hydrolysis(), tmp_path, "a"))
        second = record(explore(hydrolysis(), tmp_path, "b"))
        assert first == second

    def test_the_backend_versions_are_recorded(self, tmp_path):
        versions = out(explore(hydrolysis(), tmp_path))["versions"]
        assert set(versions) == {"doranet", "rdkit"}

    def test_a_missing_feedstock_leaves_the_target_unreached_and_says_so(
        self, tmp_path
    ):
        params = hydrolysis()
        params["feedstocks"].pop()
        outcome = explore(params, tmp_path)
        assert out(outcome)["routes"] == [] and out(outcome)["reactions"] == []
        assert out(outcome)["target_status"] == "unreached"
        assert "unreached by these operators" in " ".join(
            outcome.steps["routes"].notes
        )

    def test_asking_for_more_than_the_inventory_holds_is_reachable_but_has_no_route(
        self, tmp_path
    ):
        params = hydrolysis()
        params["target"]["amount"] = 2
        outcome = explore(params, tmp_path)
        assert (
            out(outcome)["routes"] == []
            and out(outcome)["target_status"] == "network-reachable"
        )
        assert "no sequence of steps closes the inventory" in " ".join(
            outcome.steps["routes"].notes
        )

    def test_a_smaller_extent_per_step_scales_the_amounts_exactly(self, tmp_path):
        params = hydrolysis()
        params["feedstocks"][0]["amount"] = "1/2"
        params["feedstocks"][1]["amount"] = "1/2"
        params["target"]["amount"] = "1/2"
        params["extent_per_step"] = "1/2"
        outcome = explore(params, tmp_path)
        assert len(out(outcome)["routes"]) == 1
        assert out(outcome)["routes"][0]["steps"][0]["extent"] == "1/2"

    def test_a_target_already_in_the_feed_is_a_zero_step_route(self, tmp_path):
        params = hydrolysis()
        params["target"]["smiles"] = "O"
        route = out(explore(params, tmp_path))["routes"][0]
        assert route["steps"] == [] and route["net_stoichiometry"] == {}


class TestBalanceAndBounds:
    def test_an_operator_that_loses_atoms_is_rejected_and_cannot_propagate(
        self, tmp_path
    ):
        params = hydrolysis()
        params["operators"][0]["smarts"] = (
            "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH:3].[CH4:4]"
        )
        outcome = explore(params, tmp_path)
        assert out(outcome)["routes"] == []
        assert out(outcome)["rejected_reactions"] >= 1
        assert "H" in record(outcome)["rejected_reactions"][0]["balance_residuals"]
        assert len(out(outcome)["molecules"]) == 2

    def test_a_graph_bound_that_is_reached_is_reported_as_an_unfinished_search(
        self, tmp_path
    ):
        params = hydrolysis()
        params["bounds"]["max_nodes"] = 3
        outcome = explore(params, tmp_path)
        assert len(out(outcome)["molecules"]) <= 3
        assert "max_nodes" in out(outcome)["limits_reached"]
        assert "has not finished" in " ".join(outcome.steps["routes"].notes)

    def test_a_route_state_bound_that_is_reached_is_reported(self, tmp_path):
        params = hydrolysis()
        params["bounds"]["max_route_states"] = 1
        assert "max_route_states" in out(explore(params, tmp_path))["limits_reached"]

    def test_candidate_matching_is_bounded_before_expansion(self, tmp_path):
        params = hydrolysis()
        params["feedstocks"] = [
            {"id": f"feed-{i}", "smiles": s, "amount": 1, "role": "substrate"}
            for i, s in enumerate(("C", "CC", "CCC"))
        ]
        params["target"] = {"smiles": "CCCC", "amount": 1}
        params["bounds"].update(max_recipes=1, max_nodes=3)
        params["operators"][0]["smarts"] = "[C:1].[C:2].[C:3]>>[C:1][C:2][C:3]"
        outcome = explore(params, tmp_path)
        assert "recipe_candidates" in out(outcome)["limits_reached"]
        assert out(outcome)["recipes_evaluated"] == 0
        stored = record(outcome)
        assert (
            stored["recipe_candidates_considered"],
            stored["recipe_candidate_limit"],
        ) == (3, 3)
        assert len(out(outcome)["molecules"]) == 3


class TestChemistry:
    def test_a_cofactor_must_come_back_for_the_route_to_close(self, tmp_path):
        from alambiq.capabilities.investigate.science.routes import trace_routes

        compounds = [
            {"chemical_id": k, "canonical_smiles": s}
            for k, s in [
                ("feed", "CC=O"),
                ("product", "CCO"),
                ("reduced", "CC(C)O"),
                ("oxidized", "CC(=O)C"),
                ("donor", "CO"),
                ("waste", "C=O"),
            ]
        ]
        request = {
            "feedstocks": [
                {"id": "starting", "smiles": "CC=O", "amount": 1, "role": "substrate"},
                {"id": "shuttle", "smiles": "CC(O)C", "amount": 1, "role": "cofactor"},
                {"id": "donor", "smiles": "CO", "amount": 1, "role": "cosubstrate"},
            ],
            "target": {"smiles": "CCO", "amount": 1},
            "extent_per_step": 1,
            "recover_feedstocks": ["shuttle"],
            "bounds": hydrolysis()["bounds"],
        }
        reactions = [
            {
                "id": "conversion",
                "reactants": ["feed", "reduced"],
                "stoichiometry": {
                    "feed": "-1",
                    "reduced": "-1",
                    "product": "1",
                    "oxidized": "1",
                },
            },
            {
                "id": "regeneration",
                "reactants": ["donor", "oxidized"],
                "stoichiometry": {
                    "donor": "-1",
                    "oxidized": "-1",
                    "waste": "1",
                    "reduced": "1",
                },
            },
        ]
        routes, _ = trace_routes(request, compounds, reactions, set())
        assert len(routes) == 1 and len(routes[0]["steps"]) == 2
        assert "reduced" not in routes[0]["net_stoichiometry"]
        assert routes[0]["net_stoichiometry"]["donor"] == "-1"
        assert routes[0]["balance_residuals"] == {}
        # Without the donor the shuttle cannot be regenerated, so there is no route.
        request["feedstocks"].pop()
        assert trace_routes(request, compounds, reactions, set())[0] == []

    def test_two_stereochemical_outcomes_are_both_kept_and_the_target_is_matched_exactly(
        self, tmp_path
    ):
        params = hydrolysis()
        params["feedstocks"] = [
            {
                "id": "ketone",
                "smiles": "CC(=O)c1ccccc1",
                "amount": 1,
                "role": "substrate",
            },
            {
                "id": "donor",
                "smiles": "C[C@H](N)C(=O)O",
                "amount": 1,
                "role": "cosubstrate",
            },
        ]
        params["target"] = {"smiles": "C[C@H](N)c1ccccc1", "amount": 1}
        first = params["operators"][0]
        first.update(
            id="amine-a",
            name="Ketone aminotransfer",
            smarts="[C:1](=[O:2])([#6:3])[#6:4].[C;H1:5]([NH2:6])([#6:7])[#6:8]>>[C@H:1]([NH2:6])([*:3])[*:4].[CH0:5](=[O:2])([*:7])[*:8]",
            catalyst_requirements=[
                "PLP-dependent aminotransferase and supplied amino donor"
            ],
        )
        second = copy.deepcopy(first)
        second["id"] = "amine-b"
        second["smarts"] = second["smarts"].replace("[C@H:1]", "[C@@H:1]")
        params["operators"].append(second)
        outcome = explore(params, tmp_path)
        smiles = {m["canonical_smiles"] for m in out(outcome)["molecules"]}
        assert {"C[C@H](N)c1ccccc1", "C[C@@H](N)c1ccccc1"} <= smiles
        assert out(outcome)["routes"]
        for route in out(outcome)["routes"]:
            assert route["target_amount"] == "1" and route["balance_residuals"] == {}

    def test_a_cascade_keeps_oxygen_and_peroxide_explicit_and_needs_its_oxidant(
        self, tmp_path
    ):
        params = hydrolysis()
        params["feedstocks"].append(
            {"id": "oxygen", "smiles": "O=O", "amount": 1, "role": "cosubstrate"}
        )
        params["target"] = {"smiles": "C=O", "amount": 1}
        second = copy.deepcopy(params["operators"][0])
        second.update(
            id="alcohol-oxidase",
            name="Primary alcohol oxidation",
            smarts="[C;H3:1][OH:2].[O:3]=[O:4]>>[CH2:1]=[OH0:2].[OH:3][OH:4]",
            catalyst_requirements=["A primary alcohol oxidase accepting methanol"],
        )
        params["operators"].append(second)
        outcome = explore(params, tmp_path)
        route = out(outcome)["routes"][0]
        assert len(route["steps"]) == 2
        net = by_smiles(outcome, route["net_stoichiometry"])
        assert net["O=O"] == "-1" and net["OO"] == "1" and "CO" not in net
        assert route["balance_residuals"] == {}
        params["feedstocks"].pop()
        assert out(explore(params, tmp_path, "no-oxygen"))["routes"] == []

    def test_a_chiral_operator_matches_only_the_corresponding_site(self, tmp_path):
        params = hydrolysis()
        params["feedstocks"] = [
            {
                "id": "meso-diol",
                "smiles": "C[C@H](O)[C@H](O)C",
                "amount": 1,
                "role": "substrate",
            },
            {"id": "oxygen", "smiles": "O=O", "amount": 1, "role": "cosubstrate"},
        ]
        params["target"] = {"smiles": "CC(=O)[C@H](C)O", "amount": 1}
        params["bounds"]["generations"] = 1
        params["operators"][0]["smarts"] = (
            "[CH3:1][C@H:2]([OH:3])[#6:4].[O:5]=[O:6]>>[CH3:1][CH0:2](=[OH0:3])[*:4].[OH:5][OH:6]"
        )
        outcome = explore(params, tmp_path)
        ketones = [m for m in out(outcome)["molecules"] if m["formula"] == "C4H8O2"]
        assert len(ketones) == 1, [m["canonical_smiles"] for m in ketones]

    def test_an_enzyme_reference_attaches_only_to_its_exact_equation(self, tmp_path):
        params = hydrolysis()
        params["operators"][0]["enzyme_evidence"] = [
            {
                "citation": "Hand-written reference",
                "enzyme": "Esterase A",
                "reactants": ["COC(C)=O", "O"],
                "products": ["CC(=O)O", "CO"],
            },
            {
                "citation": "Another reaction",
                "enzyme": "Esterase B",
                "reactants": ["CCOC(C)=O", "O"],
                "products": ["CC(=O)O", "CCO"],
            },
        ]
        reaction = out(explore(params, tmp_path))["reactions"][0]
        assert reaction["enzyme_references"] == 1


class TestRefusals:
    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda r: r["bounds"].update(generations=True), "whole number"),
            (lambda r: r["bounds"].update(max_route_states=0), "max_route_states"),
            (lambda r: r["bounds"].update(generations=9), "generations"),
            (
                lambda r: r["feedstocks"].append(copy.deepcopy(r["feedstocks"][0])),
                "distinct",
            ),
            (lambda r: r["operators"][0].update(smarts="C>>N"), "unique positive map"),
            (
                lambda r: r["operators"][0].update(smarts="[C:1]>>[N:2]"),
                "both equation sides",
            ),
            (
                lambda r: r["operators"][0].update(smarts="[C:1][N:2]>>[N:1][C:2]"),
                "element identity",
            ),
            (lambda r: r["target"].update(amount=0), "Coefficient must be in"),
            (lambda r: r["feedstocks"][0].update(role="reagent"), "expected one of"),
            (lambda r: r.update(recover_feedstocks=["nothing"]), "recover_feedstocks"),
        ],
    )
    def test_a_search_that_cannot_mean_anything_is_refused_in_plain_words(
        self, tmp_path, change, message
    ):
        params = hydrolysis()
        change(params)
        outcome = run(params, tmp_path)
        assert outcome.steps["routes"].status == "failed"
        assert message in outcome.steps["routes"].error


class TestPresentation:
    def test_the_report_names_the_target_status_and_the_backend_versions(
        self, tmp_path
    ):
        page = explore(hydrolysis(), tmp_path).report(tmp_path / "r.html").read_text()
        assert "inventory-route-found" in page and "Ester hydrolysis" in page
        assert "Backend:" in page and "doranet" in page

    def test_a_search_that_stopped_at_a_bound_says_so_in_the_report(self, tmp_path):
        params = hydrolysis()
        params["bounds"]["max_nodes"] = 3
        page = explore(params, tmp_path).report(tmp_path / "r.html").read_text()
        assert "Bounds reached" in page and "max_nodes" in page
