"""simulation.compare_scalar_couplings against dihedrals constructed to be exact.

Trp-cage's residue 5 is not in a ring, so cutting the bond between its backbone
nitrogen and alpha carbon splits the molecule in two. Rotating the nitrogen side about that
bond by an angle phi changes the dihedral H-N-CA-HA by exactly phi and nothing
else; the side-chain dihedrals HA-CA-CB-HB2 and HA-CA-CB-HB3 sit on the other side and
stay bit-for-bit constant. Every expected coupling below is therefore computed with an
independent NumPy dihedral from the frames the test wrote, then passed through the Karplus
relation by hand:

* the calculated value is the mean of that coupling over all frames of all replicas
  (equal replica weights), and its error the spread of the replica means over sqrt(M);
* the residual against a chosen measurement, the RMSE and the mean absolute error follow;
* two replicas written in two segments are joined end to end;
* relabelling HB2 and HB3 in the topology changes the couplings by topology naming but
  not by IUPAC geometry assignment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
md = pytest.importorskip("mdtraj")
pytest.importorskip("gemmi")

from tests.core.test_simulation_inspect_trajectory import (  # noqa: E402, F401
    ok,
    run,
    selector,
    trpcage,
    write_dcd,
    write_state,
)

A, B, C = 9.5, -1.6, 1.8
RESIDUE = 5
T975_3 = 3.1824463


def dihedral(p) -> float:
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def karplus(theta: float, a=A, b=B, c=C) -> float:
    return a * np.cos(theta) ** 2 + b * np.cos(theta) + c


def rotate_about(points, origin, axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    r = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    return (points - origin) @ r.T + origin


class Molecule:
    """Trp-cage with a rotatable N-terminal side of residue 5."""

    def __init__(self, files, directory: Path):
        self.files = files
        self.directory = directory
        self.top = files["protein_topology"]
        self.base = files["protein_xyz"]
        self.n, _ = selector(self.top, RESIDUE, "N")
        self.ca, _ = selector(self.top, RESIDUE, "CA")
        neighbours = {a.index: set() for a in self.top.atoms}
        for a, b in self.top.bonds:
            neighbours[a.index].add(b.index)
            neighbours[b.index].add(a.index)
        neighbours[self.n].discard(self.ca)
        neighbours[self.ca].discard(self.n)
        side, pending = set(), [self.n]
        while pending:
            i = pending.pop()
            if i not in side:
                side.add(i)
                pending.extend(neighbours[i] - side)
        assert self.ca not in side, "the bond N-CA of residue 5 should split the chain"
        self.side = sorted(side)
        self.index = {
            name: selector(self.top, RESIDUE, name)[0]
            for name in ("H", "N", "CA", "HA", "CB", "HB2", "HB3")
        }

    def frames(self, angles) -> list:
        axis = self.base[self.ca] - self.base[self.n]
        out = []
        for phi in angles:
            xyz = self.base.copy()
            xyz[self.side] = rotate_about(
                self.base[self.side], self.base[self.n], axis, phi
            )
            out.append(xyz)
        return out

    def path(self, a: str, b: str, c: str, d: str, xyz) -> float:
        i = self.index
        return dihedral(xyz[[i[a], i[b], i[c], i[d]]])

    def coupling(self, name: str, xyz) -> float:
        path = {
            "hnha": ("H", "N", "CA", "HA"),
            "hahb2": ("HA", "CA", "CB", "HB2"),
            "hahb3": ("HA", "CA", "CB", "HB3"),
        }[name]
        return float(karplus(self.path(*path, xyz)))


@pytest.fixture(scope="module")
def molecule(trpcage, tmp_path_factory):  # noqa: F811
    return Molecule(trpcage, tmp_path_factory.mktemp("couplings"))


def write_replica(molecule, name, angles, split=None):
    """Write a replica as one segment, or as consecutive segments of the given sizes."""
    frames = molecule.frames(angles)
    sizes = split or [len(frames)]
    segments, start = [], 0
    for k, size in enumerate(sizes):
        stem = molecule.directory / f"{name}_{k}"
        write_dcd(stem.with_suffix(".dcd"), frames[start : start + size])
        write_state(stem.with_suffix(".csv"), size, extra_first=True)
        segments.append(
            {
                "trajectory": str(stem.with_suffix(".dcd")),
                "state_data": str(stem.with_suffix(".csv")),
            }
        )
        start += size
    return {"id": name, "segments": segments}, frames


def observation(molecule, key, atoms, measured, **extra):
    chain = {"chain_id": "A", "residue_number": RESIDUE, "residue_name": "GLN"}
    return {
        "id": key,
        "atoms": [{**chain, "atom_name": n} for n in atoms],
        "karplus": {"a_hz": A, "b_hz": B, "c_hz": C},
        "measured_hz": measured,
        "reported_error_hz": 0.5,
        "error_description": "the source's reported standard deviation",
        **extra,
    }


def reference(molecule, measured, temperature=300.0, **extra):
    return {
        "title": "Synthetic couplings",
        "citation": "constructed for this test",
        "license": "test fixture",
        "temperature_kelvin": temperature,
        "conditions": "none; made up",
        "observations": [
            observation(molecule, "hnha", ("H", "HA"), measured[0]),
            observation(molecule, "hahb2", ("HA", "HB2"), measured[1]),
            observation(molecule, "hahb3", ("HA", "HB3"), measured[2]),
        ],
        **extra,
    }


def step(molecule, replicas, ref, **params):
    return {
        "id": "jc",
        "capability": "simulation.compare_scalar_couplings",
        "params": {
            "topology": str(molecule.files["protein"]),
            "reference": ref,
            "replicas": replicas,
            **params,
        },
    }


def angles(seed, n=80):
    return 2 * np.pi * np.random.default_rng(seed).random(n)


def mean_temperature(sizes_per_replica) -> float:
    return float(
        np.mean(
            [
                300.0 + k
                for sizes in sizes_per_replica
                for size in sizes
                for k in range(1, size + 1)
            ]
        )
    )


class TestKnownDihedrals:
    @pytest.fixture(scope="class")
    def built(self, molecule, tmp_path_factory):
        replicas, frames = [], []
        for i, split in enumerate([[40, 40], None, None, None]):
            replica, fr = write_replica(molecule, f"rep{i}", angles(i), split)
            replicas.append(replica)
            frames.append(fr)
        expected = {
            key: np.array([[molecule.coupling(key, x) for x in fr] for fr in frames])
            for key in ("hnha", "hahb2", "hahb3")
        }
        means = {k: v.mean(axis=1) for k, v in expected.items()}
        measured = [
            means["hnha"].mean() + 0.5,
            means["hahb2"].mean() - 0.3,
            means["hahb3"].mean() + 0.2,
        ]
        temperature = mean_temperature([[40, 40], [80], [80], [80]])
        outcome = run(
            [
                step(
                    molecule,
                    replicas,
                    reference(molecule, measured, temperature),
                    checks={
                        "split_rhat_max": 1.1,
                        "combined_ess_min": 100.0,
                        "required_jointly_passing": 1,
                    },
                )
            ],
            tmp_path_factory.mktemp("jc"),
        )
        return outcome, expected, measured, temperature

    def test_each_calculated_coupling_is_the_mean_of_its_karplus_values(self, built):
        outcome, expected, _, _ = built
        rows = {r["id"]: r for r in ok(outcome, "jc")["couplings"]}
        for key in ("hnha", "hahb2", "hahb3"):
            assert rows[key]["calculated_hz"] == pytest.approx(
                expected[key].mean(), abs=2e-3
            )

    def test_the_error_is_the_spread_of_the_replica_means_over_root_m(self, built):
        outcome, expected, _, _ = built
        row = next(r for r in ok(outcome, "jc")["couplings"] if r["id"] == "hnha")
        means = expected["hnha"].mean(axis=1)
        assert row["replica_standard_error_hz"] == pytest.approx(
            means.std(ddof=1) / 2, abs=2e-3
        )

    def test_the_side_chain_couplings_do_not_move_when_the_backbone_turns(self, built):
        outcome, expected, _, _ = built
        for key in ("hahb2", "hahb3"):
            assert np.ptp(expected[key]) < 1e-6
        rows = {r["id"]: r for r in ok(outcome, "jc")["couplings"]}
        assert rows["hahb2"]["replica_standard_error_hz"] < 1e-3
        assert rows["hahb2"]["diagnostic"] == "constant"
        assert rows["hnha"]["diagnostic"] == "estimated"

    def test_the_residuals_rmse_and_mae_follow_from_the_measurements(self, built):
        outcome, expected, measured, _ = built
        out = ok(outcome, "jc")
        calc = np.array([expected[k].mean() for k in ("hnha", "hahb2", "hahb3")])
        residual = calc - np.array(measured)
        assert [r["residual_hz"] for r in out["couplings"]] == pytest.approx(
            residual.tolist(), abs=2e-3
        )
        assert out["rmse_hz"] == pytest.approx(np.sqrt(np.mean(residual**2)), abs=2e-3)
        assert out["mae_hz"] == pytest.approx(np.mean(np.abs(residual)), abs=2e-3)
        assert out["couplings"][0]["residual_over_reported_error"] == pytest.approx(
            residual[0] / 0.5, abs=5e-3
        )

    def test_two_segments_are_joined_end_to_end_into_one_replica(self, built):
        out = ok(built[0], "jc")
        assert out["replicas"] == ["rep0", "rep1", "rep2", "rep3"]
        assert out["frames_per_replica"] == 80 and out["trimmed_frames"] == 0
        arrays = np.load(out["arrays"])
        assert arrays["replica_rep0"].shape == (80, 3)
        assert arrays["observation_ids"].tolist() == ["hnha", "hahb2", "hahb3"]
        assert arrays["replica_rep0"][:, 0] == pytest.approx(
            built[1]["hnha"][0], abs=2e-3
        )

    def test_the_sampling_judgement_counts_the_couplings_that_pass(self, built):
        out = ok(built[0], "jc")
        status = {r["id"]: r["sampling"] for r in out["couplings"]}
        # The angle is uniform, so independent draws pass; a constant series cannot.
        assert status == {"hnha": "pass", "hahb2": "fail", "hahb3": "fail"}
        assert out["sampling_status"] == "pass"
        assert 1.0 <= out["couplings"][0]["split_rhat"] < 1.1

    def test_the_sampling_interval_and_temperature_come_from_the_state_data(
        self, built
    ):
        out = ok(built[0], "jc")
        assert out["sampling_interval_ps"] == pytest.approx(0.2)
        assert out["mean_sampled_temperature_kelvin"] == pytest.approx(built[3])
        assert out["measured_temperature_kelvin"] == pytest.approx(built[3])

    def test_the_details_hold_the_atom_paths(self, built):
        details = json.loads(Path(ok(built[0], "jc")["details"]).read_text())
        paths = {
            p["id"]: [a["atom_name"] for a in p["path"]] for p in details["atom_paths"]
        }
        assert paths == {
            "hnha": ["H", "N", "CA", "HA"],
            "hahb2": ["HA", "CA", "CB", "HB2"],
            "hahb3": ["HA", "CA", "CB", "HB3"],
        }
        assert details["statistics"]["replica_count"] == 4


class TestTemperatureAndChecks:
    def replicas(self, molecule, count=3):
        return [
            write_replica(molecule, f"t{i}", angles(20 + i))[0] for i in range(count)
        ]

    def test_couplings_measured_at_another_temperature_are_flagged(
        self, molecule, tmp_path
    ):
        replicas = self.replicas(molecule)
        outcome = run(
            [step(molecule, replicas, reference(molecule, [4.0, 5.0, 6.0], 250.0))],
            tmp_path,
        )
        assert outcome.ok
        assert "mixes temperatures" in " ".join(outcome.steps["jc"].notes)

    def test_no_checks_means_no_sampling_judgement(self, molecule, tmp_path):
        replicas = self.replicas(molecule)
        out = ok(
            run(
                [step(molecule, replicas, reference(molecule, [4.0, 5.0, 6.0], 340.5))],
                tmp_path,
            ),
            "jc",
        )
        assert out["sampling_status"] is None
        assert all(r["sampling"] is None for r in out["couplings"])

    def test_requiring_more_passing_couplings_than_can_pass_fails_the_sampling(
        self, molecule, tmp_path
    ):
        replicas = self.replicas(molecule)
        out = ok(
            run(
                [
                    step(
                        molecule,
                        replicas,
                        reference(molecule, [4.0, 5.0, 6.0], 340.5),
                        checks={
                            "split_rhat_max": 1.1,
                            "combined_ess_min": 100.0,
                            "required_jointly_passing": 3,
                        },
                    )
                ],
                tmp_path,
            ),
            "jc",
        )
        assert out["sampling_status"] == "fail"

    def test_frames_beyond_a_multiple_of_twenty_are_left_out_and_noted(
        self, molecule, tmp_path
    ):
        replicas = [
            write_replica(molecule, f"odd{i}", angles(40 + i, 90))[0] for i in range(2)
        ]
        outcome = run(
            [step(molecule, replicas, reference(molecule, [4.0, 5.0, 6.0]))], tmp_path
        )
        out = ok(outcome, "jc")
        assert out["frames_per_replica"] == 80 and out["trimmed_frames"] == 10
        assert "left out" in " ".join(outcome.steps["jc"].notes)

    def test_block_counts_are_used_and_recorded(self, molecule, tmp_path):
        replicas = self.replicas(molecule, 2)
        out = ok(
            run(
                [
                    step(
                        molecule,
                        replicas,
                        reference(molecule, [4.0, 5.0, 6.0]),
                        block_counts=[4, 8],
                    )
                ],
                tmp_path,
            ),
            "jc",
        )
        details = json.loads(Path(out["details"]).read_text())
        assert [
            b["block_count_per_replica"] for b in details["statistics"]["blocking"]
        ] == [4, 8]


class TestBetaMethylene:
    @pytest.fixture(scope="class")
    def swapped(self, molecule):
        """The same coordinates with the HB2 and HB3 labels of residue 5 exchanged."""
        traj = md.load_pdb(str(molecule.files["protein"]))
        for atom in traj.topology.atoms:
            if atom.residue.resSeq == RESIDUE and atom.name in ("HB2", "HB3"):
                atom.name = "HB3" if atom.name == "HB2" else "HB2"
        path = molecule.directory / "swapped.pdb"
        traj.save_pdb(str(path))
        return path

    def couplings(self, molecule, topology, naming, tmp_path):
        replicas = [
            write_replica(molecule, f"hb{i}", angles(60 + i))[0] for i in range(2)
        ]
        params = step(
            molecule, replicas, reference(molecule, [4.0, 5.0, 6.0]), atom_naming=naming
        )
        params["params"]["topology"] = str(topology)
        out = ok(run([params], tmp_path), "jc")
        return {r["id"]: r["calculated_hz"] for r in out["couplings"]}, out

    def test_topology_naming_follows_the_labels_so_relabelling_changes_the_answer(
        self, molecule, swapped, tmp_path
    ):
        original, _ = self.couplings(
            molecule, molecule.files["protein"], "topology", tmp_path / "a"
        )
        relabelled, _ = self.couplings(molecule, swapped, "topology", tmp_path / "b")
        assert abs(original["hahb2"] - relabelled["hahb2"]) > 0.05
        assert relabelled["hahb2"] == pytest.approx(original["hahb3"], abs=2e-3)
        assert relabelled["hahb3"] == pytest.approx(original["hahb2"], abs=2e-3)

    def test_iupac_assignment_gives_the_same_answer_whatever_the_labels(
        self, molecule, swapped, tmp_path
    ):
        original, first = self.couplings(
            molecule, molecule.files["protein"], "iupac-beta-methylene", tmp_path / "a"
        )
        relabelled, second = self.couplings(
            molecule, swapped, "iupac-beta-methylene", tmp_path / "b"
        )
        assert relabelled["hahb2"] == pytest.approx(original["hahb2"], abs=2e-3)
        assert relabelled["hahb3"] == pytest.approx(original["hahb3"], abs=2e-3)
        # Exactly one of the two files had to have its labels exchanged.
        assert first["swapped_beta_hydrogens"] + second["swapped_beta_hydrogens"] == 1
        assert first["atom_naming"] == "iupac-beta-methylene"

    def test_the_assignment_is_recorded_in_the_details(
        self, molecule, swapped, tmp_path
    ):
        _, out = self.couplings(molecule, swapped, "iupac-beta-methylene", tmp_path)
        details = json.loads(Path(out["details"]).read_text())
        (record,) = details["atom_naming"]["assignments"]
        assert record["residue_number"] == RESIDUE and "swapped" in record

    def test_a_residue_without_a_beta_methylene_is_refused(self, molecule, tmp_path):
        replicas = [write_replica(molecule, f"gly{i}", angles(i))[0] for i in range(2)]
        ref = reference(molecule, [4.0, 5.0, 6.0])
        for atom in ref["observations"][1]["atoms"]:
            if atom["atom_name"] == "HB2":
                atom.update(residue_number=10, residue_name="GLY")
        outcome = run(
            [step(molecule, replicas, ref, atom_naming="iupac-beta-methylene")],
            tmp_path,
        )
        assert not outcome.ok
        assert "requires one supported residue" in outcome.steps["jc"].error


class TestRefusals:
    def replicas(self, molecule, count=2, n=80):
        return [
            write_replica(molecule, f"r{i}_{n}", angles(80 + i, n))[0]
            for i in range(count)
        ]

    def fails(self, molecule, tmp_path, message, replicas=None, ref=None, **params):
        outcome = run(
            [
                step(
                    molecule,
                    replicas or self.replicas(molecule),
                    ref or reference(molecule, [4.0, 5.0, 6.0]),
                    **params,
                )
            ],
            tmp_path,
        )
        assert not outcome.ok
        assert message in outcome.steps["jc"].error

    def test_endpoints_with_no_three_bond_path_are_refused(self, molecule, tmp_path):
        ref = reference(molecule, [4.0, 5.0, 6.0])
        ref["observations"][0]["atoms"][0]["atom_name"] = "HB2"
        ref["observations"][0]["atoms"][1]["atom_name"] = "H"
        self.fails(
            molecule,
            tmp_path,
            "expected one three-bond coupling path; found 0",
            ref=ref,
        )

    def test_an_endpoint_that_matches_no_atom_is_refused(self, molecule, tmp_path):
        ref = reference(molecule, [4.0, 5.0, 6.0])
        ref["observations"][0]["atoms"][0]["atom_name"] = "ZZ9"
        self.fails(molecule, tmp_path, "exact endpoint has 0 topology matches", ref=ref)

    @pytest.mark.parametrize(
        ("edit", "message"),
        [
            (
                lambda r: r.update(temperature_kelvin=-1.0),
                "positive temperature_kelvin",
            ),
            (lambda r: r.update(citation="  "), "non-empty citation"),
            (lambda r: r.update(observations=[]), "1–5000 scalar-coupling"),
            (
                lambda r: r["observations"][0].update(error_description=""),
                "description of what that error means",
            ),
            (
                lambda r: r["observations"][0].update(reported_error_hz=-0.1),
                "non-negative reported error",
            ),
            (
                lambda r: r["observations"][1].update(id="hnha"),
                "ids must be unique",
            ),
            (
                lambda r: r["observations"][0]["karplus"].update(a_hz=float("nan")),
                "Karplus coefficients must be finite",
            ),
        ],
    )
    def test_a_reference_missing_what_a_comparison_needs_is_refused(
        self, molecule, tmp_path, edit, message
    ):
        ref = reference(molecule, [4.0, 5.0, 6.0])
        edit(ref)
        self.fails(molecule, tmp_path, message, ref=ref)

    def test_one_replica_is_refused(self, molecule, tmp_path):
        self.fails(
            molecule,
            tmp_path,
            "2–30 replicas are required",
            replicas=self.replicas(molecule, 1),
        )

    def test_replica_ids_must_be_unique_lowercase_names(self, molecule, tmp_path):
        replicas = self.replicas(molecule)
        replicas[1] = {**replicas[1], "id": replicas[0]["id"]}
        self.fails(molecule, tmp_path, "Replica ids must be unique", replicas=replicas)
        replicas[1] = {**replicas[1], "id": "Bad Name"}
        self.fails(molecule, tmp_path, "Replica ids must be unique", replicas=replicas)

    def test_replicas_of_different_lengths_are_refused(self, molecule, tmp_path):
        replicas = [
            write_replica(molecule, "len_a", angles(1, 80))[0],
            write_replica(molecule, "len_b", angles(2, 60))[0],
        ]
        self.fails(molecule, tmp_path, "equal numbers of frames", replicas=replicas)

    def test_replicas_that_are_too_short_are_refused(self, molecule, tmp_path):
        self.fails(
            molecule,
            tmp_path,
            "at least 40 frames",
            replicas=self.replicas(molecule, 2, 30),
        )

    def test_identical_replicas_are_refused(self, molecule, tmp_path):
        one, _ = write_replica(molecule, "twin", angles(5))
        self.fails(
            molecule,
            tmp_path,
            "identical samples",
            replicas=[one, {**one, "id": "twin2"}],
        )

    def test_segments_sampled_at_different_intervals_are_refused(
        self, molecule, tmp_path
    ):
        a, _ = write_replica(molecule, "iv_a", angles(7))
        b, _ = write_replica(molecule, "iv_b", angles(8))
        state = Path(b["segments"][0]["state_data"])
        lines = state.read_text().splitlines()
        rows = [lines[0]] + [
            ",".join(
                [str(1000 + 200 * i), str((1000 + 200 * i) * 0.002)]
                + row.split(",")[2:]
            )
            for i, row in enumerate(lines[1:])
        ]
        state.write_text("\n".join(rows) + "\n")
        self.fails(molecule, tmp_path, "same interval", replicas=[a, b])

    def test_a_segment_count_outside_one_to_ten_is_refused(self, molecule, tmp_path):
        replicas = self.replicas(molecule)
        replicas[0] = {**replicas[0], "segments": []}
        self.fails(molecule, tmp_path, "one to ten segments", replicas=replicas)

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"block_counts": [1]}, "block counts between 2 and 100"),
            ({"block_counts": [7]}, "do not divide the 80 frames"),
            (
                {
                    "checks": {
                        "split_rhat_max": 0.5,
                        "combined_ess_min": 10.0,
                        "required_jointly_passing": 1,
                    }
                },
                "split R-hat limit of at least 1",
            ),
            (
                {
                    "checks": {
                        "split_rhat_max": 1.1,
                        "combined_ess_min": 10.0,
                        "required_jointly_passing": 9,
                    }
                },
                "between 1 and the number of couplings",
            ),
        ],
    )
    def test_settings_that_cannot_be_honoured_are_refused(
        self, molecule, tmp_path, changes, message
    ):
        self.fails(molecule, tmp_path, message, **changes)

    def test_a_topology_with_other_atoms_than_the_trajectory_is_refused(
        self, molecule, tmp_path
    ):
        params = step(
            molecule, self.replicas(molecule), reference(molecule, [4.0, 5.0, 6.0])
        )
        params["params"]["topology"] = str(molecule.files["ligand_pdb"])
        outcome = run([params], tmp_path)
        assert not outcome.ok
        assert "disagree about the atoms" in outcome.steps["jc"].error
