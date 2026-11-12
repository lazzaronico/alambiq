"""structure.measure_ligand_site on structures built so that every distance is known.

The reference and the mobile structure are written by hand from a few alpha carbons, a zinc
and a three-atom ligand, and the mobile structure is the reference moved by a rigid motion, so
the fit has an RMSD of zero and the ligand returns to the position it was built at:

* a rigid motion of a single chain is undone, and the distance of a ligand atom to the zinc is
  the distance written down;
* two different chains that the mobile structure names the other way round are paired by the
  search, and the same pairing given as `chain_map` gives the same numbers;
* a dimer with a two-fold axis fits equally well in two assignments, which place the ligand at
  1.5 and 2.1 Å from the zinc of the subunit they put it in, and the result gives both as the
  range of the nearest distance;
* a target without a named ligand atom reports the nearest heavy atom of the ligand;
* the ligand's bonds are the ones the geometry has;
* selectors and targets that match nothing, or more than one residue, are refused;
* a real structure (181L) fitted on itself returns an RMSD of zero and a distance of zero
  from a ligand atom to the same atom of the reference.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gemmi")

from alambiq.capabilities.investigate.science import ligand_site  # noqa: E402

# PDB coordinates have three decimals, so a moved structure carries errors of about 5e-4 Å
TOLERANCE = 2e-3
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "structures"
CAPABILITY = "structure.measure_ligand_site"

CHAIN_1 = [(0, 0, 0), (3.8, 0, 0), (3.8, 3.8, 0), (0, 3.8, 2), (1, 2, 5), (-2, 1, 3)]
CHAIN_2 = [(10, 0, 0), (12, 1, 0), (14, -2, 1), (11, 3, 4), (9, 5, 2), (8, -1, 6)]
ZINC = (2.0, 1.0, 2.0)
LIGAND = {"C1": (3.0, 1.0, 2.0), "O1": (3.2, 2.2, 2.0), "C2": (4.4, 0.5, 2.0)}


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


MOTION = (rotation((1, 2, 3), 70), np.array([11.0, -4.0, 6.0]))


def move(points, motion=MOTION):
    matrix, shift = motion
    return [tuple(matrix @ np.asarray(p, dtype=float) + shift) for p in points]


def line(record, serial, name, residue, chain, number, xyz, element):
    shown = f" {name:<3}" if len(name) < 4 else name
    return (
        f"{record:<6}{serial:>5} {shown}{' '}{residue:>3} {chain}{number:>4}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2}"
    )


def pdb(chains, hetero=()):
    """A PDB text: `chains` is {chain: points}, `hetero` is (name, residue, chain, number, xyz, element)."""
    rows, serial = [], 1
    for chain, points in chains.items():
        for number, xyz in enumerate(points, start=1):
            rows.append(line("ATOM", serial, "CA", "ALA", chain, number, xyz, "C"))
            serial += 1
    for name, residue, chain, number, xyz, element in hetero:
        rows.append(line("HETATM", serial, name, residue, chain, number, xyz, element))
        serial += 1
    return "\n".join(rows) + "\nEND\n"


def ligand_rows(points, chain="L", residue="LIG"):
    elements = {"C1": "C", "O1": "O", "C2": "C"}
    return [(n, residue, chain, 1, points[n], elements[n]) for n in points]


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def make_pair(tmp_path, reference_chains, mobile_chains, ligand_at_reference=LIGAND):
    """The reference holds the zinc; the mobile structure is the reference moved, with the ligand."""
    zinc = [("ZN", "ZN", "A", 100, ZINC, "ZN")]
    reference = pdb(reference_chains, zinc)
    moved_ligand = dict(
        zip(ligand_at_reference, move(ligand_at_reference.values()), strict=True)
    )
    mobile = pdb(mobile_chains, ligand_rows(moved_ligand))
    return write(tmp_path, "reference.pdb", reference), write(
        tmp_path, "mobile.pdb", mobile
    )


def run(request, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {"id": "site", "capability": CAPABILITY, "params": request}
    return execute({"id": "site", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("site")


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def request(reference, mobile, targets, **extra):
    return {
        "reference": reference,
        "mobile": mobile,
        "ligand": {"chain_id": "L"},
        "targets": targets,
        **extra,
    }


class TestSingleChain:
    def test_motion_is_undone_and_the_distance_is_the_one_written_down(self, tmp_path):
        reference, mobile = make_pair(tmp_path, {"A": CHAIN_1}, {"A": move(CHAIN_1)})
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [
                        {
                            "label": "zinc",
                            "residue_name": "ZN",
                            "atom": "ZN",
                            "ligand_atom": "O1",
                        }
                    ],
                ),
                tmp_path,
            )
        )
        assert found["fit"]["rmsd_angstrom"] == pytest.approx(0.0, abs=TOLERANCE)
        assert found["fit"]["fitted_residues"] == 6
        assert found["fit"]["chain_assignment"] == {"A": "A"}
        assert found["fit"]["equivalent_fits"] == 1
        measured = found["measurements"][0]
        assert measured["nearest"]["distance_angstrom"] == pytest.approx(
            distance(LIGAND["O1"], ZINC), abs=TOLERANCE
        )
        assert measured["nearest"]["chain_id"] == "A"
        assert measured["nearest"]["residue_number"] == 100
        assert measured["nearest_range_angstrom"][0] == pytest.approx(
            measured["nearest_range_angstrom"][1]
        )
        centroid = np.mean([np.asarray(p) for p in LIGAND.values()], axis=0)
        assert found["ligand"]["centroid_angstrom"] == pytest.approx(
            list(centroid), abs=TOLERANCE
        )

    def test_no_named_atom_reports_the_nearest_heavy_atom(self, tmp_path):
        reference, mobile = make_pair(tmp_path, {"A": CHAIN_1}, {"A": move(CHAIN_1)})
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [{"label": "zinc", "residue_name": "ZN", "atom": "ZN"}],
                ),
                tmp_path,
            )
        )
        nearest = min(LIGAND, key=lambda n: distance(LIGAND[n], ZINC))
        measured = found["measurements"][0]["nearest"]
        assert measured["ligand_atom"] == nearest
        assert measured["distance_angstrom"] == pytest.approx(
            distance(LIGAND[nearest], ZINC), abs=TOLERANCE
        )

    def test_the_bonds_are_the_ones_the_geometry_has(self, tmp_path):
        reference, mobile = make_pair(tmp_path, {"A": CHAIN_1}, {"A": move(CHAIN_1)})
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [{"label": "z", "residue_name": "ZN", "atom": "ZN"}],
                ),
                tmp_path,
            )
        )
        bonds = {a["name"]: a["bonded_to"] for a in found["ligand"]["atoms"]}
        # C1-O1 is 1.2 Å and C1-C2 is 1.5 Å; O1-C2 is 2.4 Å and too long to be a bond
        assert bonds == {"C1": ["C2", "O1"], "O1": ["C1"], "C2": ["C1"]}

    def test_the_fitted_structure_places_the_ligand_where_it_was_built(self, tmp_path):
        reference, mobile = make_pair(tmp_path, {"A": CHAIN_1}, {"A": move(CHAIN_1)})
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [{"label": "z", "residue_name": "ZN", "atom": "ZN"}],
                ),
                tmp_path,
            )
        )
        placed = ligand_site.read_structure(
            Path(found["fitted_structure"]).read_text(), "placed"
        )
        atoms = {
            a.name: a.pos.tolist()
            for c in placed[0]
            for r in c
            if c.name == "L"
            for a in r
        }
        for name, xyz in LIGAND.items():
            assert atoms[name] == pytest.approx(list(xyz), abs=1e-2)


class TestChainAssignment:
    def swapped(self, tmp_path):
        """The mobile structure names its two different chains the other way round."""
        return make_pair(
            tmp_path,
            {"A": CHAIN_1, "B": CHAIN_2},
            {"A": move(CHAIN_2), "B": move(CHAIN_1)},
        )

    def test_the_search_finds_the_pairing(self, tmp_path):
        reference, mobile = self.swapped(tmp_path)
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [
                        {
                            "label": "zinc",
                            "residue_name": "ZN",
                            "atom": "ZN",
                            "ligand_atom": "O1",
                        }
                    ],
                ),
                tmp_path,
            )
        )
        assert found["fit"]["chain_assignment"] == {"A": "B", "B": "A"}
        assert found["fit"]["assignments_tested"] == 2
        assert found["fit"]["rmsd_angstrom"] == pytest.approx(0.0, abs=TOLERANCE)
        assert found["fit"]["equivalent_fits"] == 1
        assert found["measurements"][0]["nearest"][
            "distance_angstrom"
        ] == pytest.approx(distance(LIGAND["O1"], ZINC), abs=TOLERANCE)

    def test_chain_map_gives_the_same_numbers(self, tmp_path):
        reference, mobile = self.swapped(tmp_path)
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [
                        {
                            "label": "zinc",
                            "residue_name": "ZN",
                            "atom": "ZN",
                            "ligand_atom": "O1",
                        }
                    ],
                    chain_map={"A": "B", "B": "A"},
                ),
                tmp_path,
            )
        )
        assert found["fit"]["assignments_tested"] == 1
        assert found["measurements"][0]["nearest"][
            "distance_angstrom"
        ] == pytest.approx(distance(LIGAND["O1"], ZINC), abs=TOLERANCE)

    def test_a_wrong_chain_map_fits_badly(self, tmp_path):
        reference, mobile = self.swapped(tmp_path)
        found = ok(
            run(
                request(
                    reference,
                    mobile,
                    [{"label": "zinc", "residue_name": "ZN", "atom": "ZN"}],
                    chain_map={"A": "A", "B": "B"},
                ),
                tmp_path,
            )
        )
        assert found["fit"]["rmsd_angstrom"] > 1.0

    def test_a_wider_tolerance_admits_the_poorer_assignment(self, tmp_path):
        reference, mobile = self.swapped(tmp_path)
        target = [
            {"label": "zinc", "residue_name": "ZN", "atom": "ZN", "ligand_atom": "O1"}
        ]
        narrow = ok(run(request(reference, mobile, target), tmp_path, "narrow"))
        wide = ok(
            run(
                request(reference, mobile, target, equivalent_fit_angstrom=5.0),
                tmp_path,
                "wide",
            )
        )
        assert narrow["fit"]["equivalent_fits"] == 1
        assert wide["fit"]["equivalent_fits"] == 2
        assert wide["fit"]["rmsd_angstrom"] == pytest.approx(
            narrow["fit"]["rmsd_angstrom"]
        )
        # the chosen fit is the same, and the range now includes the poorer fit
        assert (
            wide["measurements"][0]["nearest"] == narrow["measurements"][0]["nearest"]
        )
        low, high = wide["measurements"][0]["nearest_range_angstrom"]
        assert low <= wide["measurements"][0]["nearest"]["distance_angstrom"] <= high
        assert high > low

    def test_a_dimer_with_a_two_fold_axis_has_two_equivalent_fits(self, tmp_path):
        # Chain B is chain A turned by 180 degrees about the z axis through (5, 5): the
        # identity assignment and the swap both fit with RMSD 0. The zinc of subunit B sits 0.6 Å
        # from the position the symmetry gives, so the ligand lies 1.5 Å from the zinc of A in
        # one assignment and 2.1 Å from the zinc of B in the other.
        def turn(point):
            return (10.0 - point[0], 10.0 - point[1], point[2])

        chain_b = [turn(p) for p in CHAIN_1]
        reference_text = pdb(
            {"A": CHAIN_1, "B": chain_b},
            [
                ("ZN", "ZN", "A", 100, ZINC, "ZN"),
                ("ZN", "ZN", "B", 100, tuple(np.add(turn(ZINC), (0.6, 0, 0))), "ZN"),
            ],
        )
        ligand = {
            "O1": (ZINC[0] + 1.5, ZINC[1], ZINC[2]),
            "C1": (ZINC[0] + 2.7, ZINC[1], ZINC[2]),
            "C2": (ZINC[0] + 4.2, ZINC[1], ZINC[2]),
        }
        moved = dict(zip(ligand, move(ligand.values()), strict=True))
        mobile_text = pdb({"A": move(CHAIN_1), "B": move(chain_b)}, ligand_rows(moved))
        reference = write(tmp_path, "reference.pdb", reference_text)
        mobile = write(tmp_path, "mobile.pdb", mobile_text)
        target = [
            {"label": "zinc", "residue_name": "ZN", "atom": "ZN", "ligand_atom": "O1"}
        ]
        outcome = run(request(reference, mobile, target), tmp_path)
        found = ok(outcome)
        assert found["fit"]["assignments_tested"] == 2
        assert found["fit"]["equivalent_fits"] == 2
        low, high = found["measurements"][0]["nearest_range_angstrom"]
        assert low == pytest.approx(1.5, abs=TOLERANCE)
        assert high == pytest.approx(2.1, abs=TOLERANCE)
        assert low <= found["measurements"][0]["nearest"]["distance_angstrom"] <= high
        assert any(
            "assignments of the chains fit" in n for n in outcome.steps["site"].notes
        )
        # naming the pairing leaves one fit
        chain_map = {"A": "A", "B": "B"}
        pinned = ok(
            run(
                request(reference, mobile, target, chain_map=chain_map),
                tmp_path,
                "pinned",
            )
        )
        assert pinned["fit"]["equivalent_fits"] == 1
        assert pinned["measurements"][0]["nearest"][
            "distance_angstrom"
        ] == pytest.approx(1.5, abs=TOLERANCE)


class TestRefusals:
    def base(self, tmp_path):
        return make_pair(tmp_path, {"A": CHAIN_1}, {"A": move(CHAIN_1)})

    def refused(self, tmp_path, params, fragment):
        outcome = run(params, tmp_path)
        assert not outcome.ok
        errors = " ".join(s.error or "" for s in outcome.steps.values())
        assert fragment in errors, errors

    def zinc(self):
        return [{"label": "zinc", "residue_name": "ZN", "atom": "ZN"}]

    def test_a_ligand_selector_that_matches_nothing(self, tmp_path):
        reference, mobile = self.base(tmp_path)
        params = request(reference, mobile, self.zinc())
        params["ligand"] = {"chain_id": "Q"}
        self.refused(tmp_path, params, "matches 0 residues")

    def test_a_ligand_selector_with_no_field(self, tmp_path):
        reference, mobile = self.base(tmp_path)
        params = request(reference, mobile, self.zinc())
        params["ligand"] = {}
        self.refused(tmp_path, params, "residue_name, a chain_id or both")

    def test_an_unknown_ligand_atom(self, tmp_path):
        reference, mobile = self.base(tmp_path)
        params = request(
            reference,
            mobile,
            [
                {
                    "label": "zinc",
                    "residue_name": "ZN",
                    "atom": "ZN",
                    "ligand_atom": "N9",
                }
            ],
        )
        self.refused(tmp_path, params, "no atom N9")

    def test_a_target_the_reference_does_not_have(self, tmp_path):
        reference, mobile = self.base(tmp_path)
        params = request(
            reference, mobile, [{"label": "x", "residue_name": "NAP", "atom": "C4N"}]
        )
        self.refused(tmp_path, params, "no atom C4N in a residue named NAP")

    def test_two_targets_with_one_label(self, tmp_path):
        reference, mobile = self.base(tmp_path)
        params = request(reference, mobile, self.zinc() + self.zinc())
        self.refused(tmp_path, params, "labels of the targets are not distinct")

    def test_different_numbers_of_chains_need_a_chain_map(self, tmp_path):
        reference, mobile = make_pair(
            tmp_path, {"A": CHAIN_1, "B": CHAIN_2}, {"A": move(CHAIN_1)}
        )
        self.refused(
            tmp_path, request(reference, mobile, self.zinc()), "give chain_map"
        )

    def test_residues_that_do_not_correspond(self, tmp_path):
        reference = write(
            tmp_path,
            "reference.pdb",
            pdb({"A": CHAIN_1}, [("ZN", "ZN", "A", 100, ZINC, "ZN")]),
        )
        shifted = pdb(
            {"A": move(CHAIN_1)},
            ligand_rows(dict(zip(LIGAND, move(LIGAND.values()), strict=True))),
        )
        shifted = shifted.replace("ALA A", "GLY A")
        mobile = write(tmp_path, "mobile.pdb", shifted)
        self.refused(
            tmp_path,
            request(reference, mobile, self.zinc()),
            "Fewer than 3 residues correspond",
        )

    def test_a_chain_map_that_names_a_missing_chain(self, tmp_path):
        reference, mobile = self.base(tmp_path)
        self.refused(
            tmp_path,
            request(reference, mobile, self.zinc(), chain_map={"A": "Z"}),
            "reference chain Z",
        )

    def test_a_file_that_is_not_a_structure(self, tmp_path):
        reference, _ = self.base(tmp_path)
        junk = write(tmp_path, "junk.txt", "not a structure\n")
        self.refused(tmp_path, request(reference, junk, self.zinc()), "holds no atoms")


class TestRealStructure:
    def test_a_structure_fitted_on_itself(self, tmp_path):
        path = str(FIXTURES / "181L.cif")
        found = ok(
            run(
                {
                    "reference": path,
                    "mobile": path,
                    "ligand": {"residue_name": "HED"},
                    "targets": [
                        {
                            "label": "same atom",
                            "residue_name": "HED",
                            "atom": "O1",
                            "ligand_atom": "O1",
                        }
                    ],
                },
                tmp_path,
            )
        )
        assert found["fit"]["rmsd_angstrom"] == pytest.approx(0.0, abs=1e-4)
        assert found["fit"]["chain_assignment"] == {"A": "A"}
        assert found["fit"]["mobile_coverage"] == pytest.approx(1.0)
        assert found["measurements"][0]["nearest"][
            "distance_angstrom"
        ] == pytest.approx(0.0, abs=1e-3)
        assert found["ligand"]["residue_name"] == "HED"
