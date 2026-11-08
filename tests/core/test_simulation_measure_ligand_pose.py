"""simulation.measure_ligand_pose against trajectories whose answers are known.

The cases build a small complex by hand: ten alpha carbons around a benzene ring that lies
in the xy plane with its center at the origin, and DCD frames made from it by exact
transformations, so each number the capability returns can be written down:

* a rigid rotation and translation of every atom leaves the ligand RMSD at zero, because
  the pocket fit removes it;
* a translation of the ligand by d in the protein frame gives an RMSD and a centroid
  shift of d;
* a benzene turned by 60 degrees about its ring normal coincides with itself, so the
  smallest RMSD over the symmetric pairings is zero, and turned by 30 degrees every atom
  moves by the chord 2 r sin(15 deg) of the ring of radius r = 1.4 Å;
* for toluene, which has two pairings, the RMSD of a turned pose equals the smaller of two
  sums that the test writes out atom by atom;
* a ligand translated by a whole box vector looks far away as stored and is put back by
  whole-molecule imaging;
* the number of fit atoms equals the number of alpha carbons that a brute-force distance
  count finds within the cutoff.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
md = pytest.importorskip("mdtraj")
pytest.importorskip("rdkit")

from alambiq.capabilities.investigate.science import ligand_pose  # noqa: E402

RING = 1.4
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
CELL = 60.0
CAPABILITY = "simulation.measure_ligand_pose"


def run(params, tmp_path, name="run"):
    from alambiq.api import run as execute

    step = {"id": "pose", "capability": CAPABILITY, "params": params}
    return execute({"id": "pose", "steps": [step]}, run_dir=tmp_path / name)


def ok(outcome):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("pose")


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    a = math.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * (k @ k)


def ring(radius=RING):
    return np.array(
        [
            [radius * math.cos(math.radians(60 * k)), radius * math.sin(math.radians(60 * k)), 0]
            for k in range(6)
        ]
    )


def pocket():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(10, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    radii = np.array([6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0])
    return vectors * radii[:, None]


class System:
    """Ten alpha carbons in chain A and a ligand residue in chain B."""

    def __init__(self, ligand_atoms: np.ndarray, residue="BNZ", hydrogens=False):
        self.protein = pocket()
        heavy = ligand_atoms
        coordinates = [*self.protein, *heavy]
        names = [f"C{i + 1}" for i in range(len(heavy))]
        elements = ["C"] * len(heavy)
        if hydrogens:
            centered = heavy - heavy.mean(axis=0)
            outward = centered / np.linalg.norm(centered, axis=1, keepdims=True)
            coordinates += list(heavy + outward * 1.08)
            names += [f"H{i + 1}" for i in range(len(heavy))]
            elements += ["H"] * len(heavy)
        self.xyz = np.array(coordinates, dtype=float)
        self.ligand = list(range(10, 10 + len(heavy)))
        self.all_ligand = list(range(10, len(self.xyz)))
        top = md.Topology()
        chain_a = top.add_chain()
        for i in range(10):
            residue_a = top.add_residue("ALA", chain_a, resSeq=i + 1)
            top.add_atom("CA", md.element.carbon, residue_a)
        chain_b = top.add_chain()
        residue_b = top.add_residue(residue, chain_b, resSeq=1)
        for name, element in zip(names, elements, strict=True):
            top.add_atom(
                name,
                md.element.hydrogen if element == "H" else md.element.carbon,
                residue_b,
            )
        self.topology = top

    def transformed(
        self, rigid=None, shift=None, turn=None, ligand_shift_cell=None
    ) -> np.ndarray:
        xyz = self.xyz.copy()
        lig = self.all_ligand
        if turn is not None:
            axis, degrees = turn
            xyz[lig] = xyz[lig] @ rotation(axis, degrees).T
        if shift is not None:
            xyz[lig] += np.asarray(shift, dtype=float)
        if ligand_shift_cell is not None:
            xyz[lig] += np.asarray(ligand_shift_cell, dtype=float) * CELL
        if rigid is not None:
            rot, translation = rigid
            xyz = xyz @ rot.T + translation
        return xyz

    def write(self, directory: Path, frames, cell=None, name="system"):
        topology = directory / f"{name}.pdb"
        reference = directory / f"{name}_reference.pdb"
        dcd = directory / f"{name}.dcd"
        for path, xyz in ((topology, frames[0]), (reference, self.xyz)):
            traj = md.Trajectory(np.asarray(xyz)[None] / 10, self.topology)
            if cell:
                traj.unitcell_lengths = np.full((1, 3), cell / 10)
                traj.unitcell_angles = np.full((1, 3), 90.0)
            traj.save_pdb(str(path))
        stacked = np.asarray(frames, dtype=np.float32)
        with md.formats.DCDTrajectoryFile(str(dcd), "w") as handle:
            if cell:
                handle.write(
                    stacked,
                    cell_lengths=np.full((len(stacked), 3), cell, dtype=np.float32),
                    cell_angles=np.full((len(stacked), 3), 90.0, dtype=np.float32),
                )
            else:
                handle.write(stacked)
        return str(topology), str(dcd), str(reference)


def request(files, smiles=BENZENE, **changes):
    topology, trajectory, reference = files
    params = {
        "topology": topology,
        "trajectory": trajectory,
        "reference_structure": reference,
        "ligand": {"chain_id": "B", "residue_number": 1, "residue_name": "BNZ"},
        "reference_ligand": {"chain_id": "B", "residue_number": 1, "residue_name": "BNZ"},
        "smiles": smiles,
        **changes,
    }
    return params


@pytest.fixture(scope="module")
def benzene(tmp_path_factory):
    system = System(ring())
    rigid = (rotation([1, 2, 3], 37.0), np.array([4.0, -2.0, 7.0]))
    frames = [
        system.transformed(),
        system.transformed(rigid=rigid),
        system.transformed(shift=[2.0, 0, 0]),
        system.transformed(turn=([0, 0, 1], 60.0)),
        system.transformed(turn=([0, 0, 1], 30.0)),
        system.transformed(rigid=rigid, shift=[0, 3.0, 0]),
    ]
    directory = tmp_path_factory.mktemp("benzene")
    return system, frames, system.write(directory, frames)


class TestKnownPoses:
    @pytest.fixture(scope="class")
    def result(self, benzene, tmp_path_factory):
        _, _, files = benzene
        return ok(run(request(files), tmp_path_factory.mktemp("run")))

    def test_each_frame_has_the_rmsd_that_its_construction_fixes(self, result):
        data = np.load(result["arrays"])
        chord = 2 * RING * math.sin(math.radians(15.0))
        expected = [0.0, 0.0, 2.0, 0.0, chord, 3.0]
        assert data["rmsd_angstrom"] == pytest.approx(expected, abs=2e-3)

    def test_the_centroid_shift_is_the_translation_of_the_ligand(self, result):
        data = np.load(result["arrays"])
        assert data["centroid_shift_angstrom"] == pytest.approx(
            [0.0, 0.0, 2.0, 0.0, 0.0, 3.0], abs=2e-3
        )

    def test_a_rigid_motion_of_everything_leaves_the_fit_exact(self, result):
        data = np.load(result["arrays"])
        assert data["fit_rmsd_angstrom"][1] == pytest.approx(0.0, abs=2e-3)

    def test_the_summary_follows_from_the_series(self, result):
        data = np.load(result["arrays"])["rmsd_angstrom"]
        assert result["frames"] == 6
        assert result["rmsd"]["mean"] == pytest.approx(float(data.mean()), abs=1e-9)
        assert result["rmsd"]["maximum"] == pytest.approx(3.0, abs=2e-3)
        assert result["rmsd"]["first"] == pytest.approx(0.0, abs=2e-3)
        assert result["rmsd"]["last"] == pytest.approx(3.0, abs=2e-3)

    def test_the_threshold_counts_frames_and_names_the_first_frame_beyond(self, result):
        assert result["threshold_angstrom"] == 2.0
        # Frame 2 is at 2.0 Å (inside by a rounding margin or not): count what the data say.
        data = np.load(result["arrays"])["rmsd_angstrom"]
        inside = int((data <= 2.0).sum())
        assert result["frames_within_threshold"] == inside
        assert result["fraction_within_threshold"] == pytest.approx(inside / 6)
        beyond = np.flatnonzero(data > 2.0)
        assert result["first_frame_beyond_threshold"] == int(beyond[0])

    def test_benzene_has_twelve_symmetric_pairings_and_six_heavy_atoms(self, result):
        assert result["pairings"] == 12
        assert result["heavy_atoms"] == 6


class TestRequestFields:
    def test_a_lower_threshold_changes_the_count(self, benzene, tmp_path):
        _, _, files = benzene
        out = ok(run(request(files, threshold_angstrom=1.0), tmp_path))
        assert out["threshold_angstrom"] == 1.0
        assert out["frames_within_threshold"] == 4
        assert out["first_frame_beyond_threshold"] == 2

    @pytest.mark.parametrize("cutoff", [8.0, 10.0, 12.5, 100.0])
    def test_the_fit_is_the_alpha_carbons_within_the_cutoff_of_the_ligand(
        self, benzene, tmp_path, cutoff
    ):
        system, _, files = benzene
        distance = [
            np.linalg.norm(system.xyz[i] - system.xyz[system.ligand], axis=1).min()
            for i in range(10)
        ]
        expected = sum(d <= cutoff for d in distance)
        assert 3 <= expected <= 10
        out = ok(run(request(files, fit_cutoff_angstrom=cutoff), tmp_path, f"c{cutoff}"))
        assert out["fit_atoms"] == expected
        assert out["fit_cutoff_angstrom"] == cutoff

    def test_the_chunk_size_does_not_change_the_numbers(self, benzene, tmp_path):
        _, _, files = benzene
        one = ok(run(request(files, chunk_frames=1), tmp_path, "one"))
        many = ok(run(request(files, chunk_frames=64), tmp_path, "many"))
        assert np.load(one["arrays"])["rmsd_angstrom"] == pytest.approx(
            np.load(many["arrays"])["rmsd_angstrom"], abs=1e-9
        )

    def test_state_data_gives_the_time_of_each_frame(self, benzene, tmp_path):
        _, _, files = benzene
        path = tmp_path / "state.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "step",
                    "time_ps",
                    "potential_kj_mol",
                    "kinetic_kj_mol",
                    "total_kj_mol",
                    "temperature_kelvin",
                ]
            )
            for i in range(7):
                writer.writerow([1000 + 100 * i, 2.0 + 0.2 * i, -1.0, 1.0, 0.0, 300.0])
        out = ok(run(request(files, state_data=str(path)), tmp_path))
        assert out["first_time_ps"] == pytest.approx(2.2)
        assert out["last_time_ps"] == pytest.approx(3.2)
        assert np.load(out["arrays"])["time_ps"] == pytest.approx(
            [2.2, 2.4, 2.6, 2.8, 3.0, 3.2]
        )

    def test_hydrogens_of_the_ligand_do_not_enter_the_rmsd(self, tmp_path):
        system = System(ring(), hydrogens=True)
        frames = [system.transformed(), system.transformed(shift=[0, 0, 1.5])]
        files = system.write(tmp_path, frames)
        out = ok(run(request(files), tmp_path))
        assert out["heavy_atoms"] == 6
        assert np.load(out["arrays"])["rmsd_angstrom"] == pytest.approx(
            [0.0, 1.5], abs=2e-3
        )


class TestToluene:
    def test_the_rmsd_is_the_smaller_of_the_two_pairings(self, tmp_path):
        atoms = np.vstack([ring(), [[2.9, 0.0, 0.0]]])
        system = System(atoms, residue="TOL")
        frames = [system.transformed(), system.transformed(turn=([0, 0, 1], 60.0))]
        files = system.write(tmp_path, frames)
        selector = {"chain_id": "B", "residue_number": 1, "residue_name": "TOL"}
        out = ok(
            run(
                request(files, smiles=TOLUENE, ligand=selector, reference_ligand=selector),
                tmp_path,
            )
        )
        assert out["pairings"] == 2
        moved = frames[1][system.ligand]
        reference = system.xyz[system.ligand]
        identity = list(range(7))
        mirror = [0, 5, 4, 3, 2, 1, 6]
        errors = [
            math.sqrt(np.mean(np.sum((moved[i] - reference[p]) ** 2, axis=1)))
            for i, p in ((identity, identity), (identity, mirror))
        ]
        expected = min(
            math.sqrt(np.mean(np.sum((moved - reference[p]) ** 2, axis=1)))
            for p in (identity, mirror)
        )
        assert errors[0] > 1.0
        assert np.load(out["arrays"])["rmsd_angstrom"][1] == pytest.approx(
            expected, abs=2e-3
        )


class TestImaging:
    @pytest.fixture(scope="class")
    def periodic(self, tmp_path_factory):
        system = System(ring())
        frames = [system.transformed(), system.transformed(ligand_shift_cell=[1, 0, 0])]
        directory = tmp_path_factory.mktemp("periodic")
        return system.write(directory, frames, cell=CELL)

    def test_a_ligand_a_box_away_is_far_as_stored(self, periodic, tmp_path):
        out = ok(run(request(periodic, imaging="as-stored"), tmp_path))
        assert np.load(out["arrays"])["rmsd_angstrom"][1] == pytest.approx(CELL, abs=0.05)

    def test_whole_molecule_imaging_puts_it_back(self, periodic, tmp_path):
        out = ok(run(request(periodic, imaging="whole-molecules"), tmp_path))
        assert out["imaging"] == "whole-molecules"
        assert np.load(out["arrays"])["rmsd_angstrom"] == pytest.approx([0.0, 0.0], abs=2e-3)

    def test_whole_molecule_imaging_needs_a_cell(self, benzene, tmp_path):
        _, _, files = benzene
        outcome = run(request(files, imaging="whole-molecules"), tmp_path)
        assert not outcome.ok
        assert "positive-volume cell" in outcome.steps["pose"].error

    def test_the_nearest_image_search_matches_brute_force(self):
        rng = np.random.default_rng(1)
        boxes = np.tile(np.diag([30.0, 30.0, 30.0]), (5, 1, 1))
        anchor = rng.uniform(0, 30, size=(5, 3))
        ligand = anchor + rng.uniform(-12, 12, size=(5, 3)) + np.array([30.0, -30.0, 0.0])
        shift = ligand_pose.nearest_image_shift(ligand, anchor, boxes)
        moved = ligand + shift
        for f in range(5):
            best = min(
                np.linalg.norm(
                    ligand[f] + np.array([i, j, k]) * 30.0 - anchor[f]
                )
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                for k in (-1, 0, 1)
            )
            assert np.linalg.norm(moved[f] - anchor[f]) == pytest.approx(best)


class TestRefusals:
    def test_a_smiles_that_is_not_the_ligand_is_refused(self, benzene, tmp_path):
        _, _, files = benzene
        outcome = run(request(files, smiles=TOLUENE), tmp_path)
        assert not outcome.ok
        assert "heavy atoms" in outcome.steps["pose"].error

    def test_a_smiles_with_the_right_atoms_and_other_connectivity_is_refused(
        self, benzene, tmp_path
    ):
        _, _, files = benzene
        outcome = run(request(files, smiles="CCCCCC"), tmp_path)
        assert not outcome.ok
        assert "does not match" in outcome.steps["pose"].error

    def test_a_residue_that_is_not_in_the_topology_is_refused(self, benzene, tmp_path):
        _, _, files = benzene
        wrong = {"chain_id": "B", "residue_number": 7, "residue_name": "BNZ"}
        outcome = run(request(files, ligand=wrong), tmp_path)
        assert not outcome.ok
        assert "matches 0 residues" in outcome.steps["pose"].error

    def test_a_reference_residue_that_is_not_there_is_refused(self, benzene, tmp_path):
        _, _, files = benzene
        wrong = {"chain_id": "B", "residue_number": 1, "residue_name": "XYZ"}
        outcome = run(request(files, reference_ligand=wrong), tmp_path)
        assert not outcome.ok
        assert "reference_ligand" in outcome.steps["pose"].error

    def test_fewer_than_three_pocket_alpha_carbons_are_refused(self, benzene, tmp_path):
        _, _, files = benzene
        outcome = run(request(files, fit_cutoff_angstrom=5.0), tmp_path)
        assert not outcome.ok
        assert "at least 3" in outcome.steps["pose"].error

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"fit_cutoff_angstrom": 3.9}, "fit_cutoff_angstrom must be from 4 to 1000"),
            ({"fit_cutoff_angstrom": 1000.1}, "fit_cutoff_angstrom must be from 4 to 1000"),
            ({"threshold_angstrom": 0.05}, "threshold_angstrom must be from 0.1 to 20"),
            ({"threshold_angstrom": 20.5}, "threshold_angstrom must be from 0.1 to 20"),
            ({"chunk_frames": 0}, "chunk_frames must be an integer from 1 to 64"),
            ({"chunk_frames": 65}, "chunk_frames must be an integer from 1 to 64"),
            ({"smiles": "c1ccc cc1"}, "no whitespace"),
        ],
    )
    def test_settings_outside_their_limits_are_refused_by_name(
        self, benzene, tmp_path, change, message
    ):
        _, _, files = benzene
        outcome = run(request(files, **change), tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["pose"].error

    def test_an_unknown_imaging_is_refused(self):
        with pytest.raises(ValueError, match="`as-stored` or `whole-molecules`"):
            ligand_pose.check_pose(
                fit_cutoff_angstrom=10.0,
                threshold_angstrom=2.0,
                imaging="wrapped",
                chunk_frames=16,
            )

    def test_state_data_of_the_wrong_length_is_refused(self, benzene, tmp_path):
        _, _, files = benzene
        path = tmp_path / "short.csv"
        path.write_text(
            "step,time_ps,potential_kj_mol,kinetic_kj_mol,total_kj_mol,temperature_kelvin\n"
            "1,1.0,-1,1,0,300\n2,2.0,-1,1,0,300\n"
        )
        outcome = run(request(files, state_data=str(path)), tmp_path)
        assert not outcome.ok
        assert "state data has 2 samples" in outcome.steps["pose"].error
