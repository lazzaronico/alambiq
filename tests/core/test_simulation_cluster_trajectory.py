"""simulation.cluster_trajectory on trajectories whose clusters are known by construction.

Every frame is Trp-cage's first model, scaled about the centroid of the compared atoms by a
state factor and then moved by a random rigid rotation and translation. Scaling about the
centroid leaves the best superposition of two frames as the identity, so the RMSD between a
frame of scale s1 and one of scale s2 is |s1 - s2| times the radius of gyration of the compared
atoms, a value the test computes with NumPy alone. Frames of one state are at RMSD 0 from each
other whatever their rigid motion. With these distances the clusters that the Daura et al.
procedure forms follow by hand:

* a cutoff below the distance between states gives one cluster per state, ordered by size, and
  the center of each is its lowest-index frame (the neighbor counts of its frames are equal);
* a cutoff above the distance between neighboring states and below that between the outer ones
  makes the middle state's frames the most connected, so its first frame is the center of one
  cluster that holds every frame.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
md = pytest.importorskip("mdtraj")
gemmi = pytest.importorskip("gemmi")

STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"
STATES = {1.0: 30, 1.5: 10, 2.0: 5}
SELECTIONS = {
    "alpha-carbons": "protein and name CA",
    "backbone": "protein and backbone",
    "protein-heavy-atoms": "protein and not element H",
}
BOX = 40.0


def run_step(tmp_path: Path, name: str = "run", **params):
    from alambiq.api import run as execute

    return execute(
        {
            "id": "c",
            "steps": [
                {
                    "id": "cluster",
                    "capability": "simulation.cluster_trajectory",
                    "params": params,
                }
            ],
        },
        run_dir=tmp_path / name,
    )


def cluster(tmp_path: Path, name: str = "run", **params) -> dict:
    outcome = run_step(tmp_path, name, **params)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("cluster")


def error_of(tmp_path: Path, name: str = "run", **params) -> str:
    outcome = run_step(tmp_path, name, **params)
    assert not outcome.ok
    return " ".join(s.error for s in outcome.steps.values() if s.error)


def rotation(rng) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = rng.uniform(0, 2 * np.pi)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def radius_of_gyration(xyz, indices) -> float:
    points = xyz[indices]
    return float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))


def write_dcd(path: Path, frames, cell: float | None = None) -> None:
    xyz = np.asarray(frames, dtype=np.float32)
    with md.formats.DCDTrajectoryFile(str(path), "w") as handle:
        if cell:
            handle.write(
                xyz,
                cell_lengths=np.full((len(xyz), 3), cell, dtype=np.float32),
                cell_angles=np.full((len(xyz), 3), 90.0, dtype=np.float32),
            )
        else:
            handle.write(xyz)


@pytest.fixture(scope="module")
def trpcage(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cluster")
    structure = gemmi.read_structure(str(STRUCTURE))
    structure.setup_entities()
    del structure[1:]
    bare = directory / "bare.pdb"
    structure.write_pdb(str(bare))
    protein = md.load_pdb(str(bare))
    protein.save_pdb(str(directory / "topology.pdb"))
    return {
        "directory": directory,
        "topology": directory / "topology.pdb",
        "xyz": protein.xyz[0] * 10,
        "mdtop": protein.topology,
    }


@pytest.fixture(scope="module")
def states(trpcage):
    """45 frames in shuffled order, the scale of each frame, and the DCD that holds them."""
    rng = np.random.default_rng(7)
    scales = [s for s, count in STATES.items() for _ in range(count)]
    rng.shuffle(scales)
    ca = trpcage["mdtop"].select(SELECTIONS["alpha-carbons"])
    base = trpcage["xyz"]
    centre = base[ca].mean(axis=0)
    frames = []
    for scale in scales:
        moved = centre + scale * (base - centre)
        moved = (moved - centre) @ rotation(rng).T + centre + rng.uniform(-8, 8, 3)
        frames.append(moved)
    path = trpcage["directory"] / "states.dcd"
    write_dcd(path, frames)
    return {"scales": scales, "frames": frames, "dcd": path, **trpcage}


def request(states, **params) -> dict:
    return {
        "topology": str(states["topology"]),
        "trajectory": str(states["dcd"]),
        **params,
    }


def first_frame_of(scales, scale, start=0, stride=1) -> int:
    return next(i for i in range(start, len(scales), stride) if scales[i] == scale)


class TestOneClusterPerState:
    @pytest.fixture(scope="class")
    def out(self, states, tmp_path_factory):
        return cluster(
            tmp_path_factory.mktemp("separate"),
            **request(states, rmsd_cutoff_angstrom=1.0),
        )

    def test_there_is_one_cluster_per_state_ordered_by_size(self, out):
        assert [c["size"] for c in out["clusters"]] == [30, 10, 5]
        assert out["cluster_count"] == 3 and out["singleton_count"] == 0
        assert out["frames_read"] == out["frames_in_file"] == 45
        assert [c["index"] for c in out["clusters"]] == [1, 2, 3]
        assert [c["fraction"] for c in out["clusters"]] == pytest.approx(
            [30 / 45, 10 / 45, 5 / 45]
        )

    def test_the_center_of_each_cluster_is_the_lowest_index_frame_of_its_state(
        self, out, states
    ):
        expected = [first_frame_of(states["scales"], s) for s in (1.0, 1.5, 2.0)]
        assert [c["center_frame"] for c in out["clusters"]] == expected

    def test_members_are_at_zero_rmsd_from_the_center_whatever_the_rigid_motion(
        self, out
    ):
        for c in out["clusters"]:
            assert c["mean_rmsd_to_center_angstrom"] < 0.01
            assert c["max_rmsd_to_center_angstrom"] < 0.01

    def test_the_assignments_give_every_frame_the_rank_of_its_state(self, out, states):
        rows = [
            line.split(",")
            for line in Path(out["assignments"]).read_text().splitlines()
        ]
        assert rows[0] == ["frame", "cluster"]
        rank = {1.0: 1, 1.5: 2, 2.0: 3}
        assert [(int(f), int(c)) for f, c in rows[1:]] == [
            (i, rank[s]) for i, s in enumerate(states["scales"])
        ]


class TestDistancesAreExact:
    @pytest.mark.parametrize("atoms", list(SELECTIONS))
    def test_the_state_distance_is_the_scale_difference_times_the_radius_of_gyration(
        self, atoms, states, tmp_path
    ):
        selected = states["mdtop"].select(SELECTIONS[atoms])
        rg = radius_of_gyration(states["frames"][0], selected) / states["scales"][0]
        below = cluster(
            tmp_path,
            "below",
            **request(states, atoms=atoms, rmsd_cutoff_angstrom=0.45 * rg),
        )
        above = cluster(
            tmp_path,
            "above",
            **request(states, atoms=atoms, rmsd_cutoff_angstrom=0.55 * rg),
        )
        assert below["atoms_compared"] == len(selected)
        assert [c["size"] for c in below["clusters"]] == [30, 10, 5]
        assert [c["size"] for c in above["clusters"]] == [45]

    def test_a_cutoff_between_neighboring_and_outer_states_centers_the_cluster_on_the_middle_state(
        self, states, tmp_path
    ):
        ca = states["mdtop"].select(SELECTIONS["alpha-carbons"])
        rg = radius_of_gyration(states["frames"][0], ca) / states["scales"][0]
        out = cluster(tmp_path, **request(states, rmsd_cutoff_angstrom=0.5 * rg + 0.05))
        (only,) = out["clusters"]
        assert only["size"] == 45
        assert only["center_frame"] == first_frame_of(states["scales"], 1.5)
        assert only["max_rmsd_to_center_angstrom"] == pytest.approx(0.5 * rg, abs=0.01)
        assert only["mean_rmsd_to_center_angstrom"] == pytest.approx(
            35 / 45 * 0.5 * rg, abs=0.01
        )


class TestFramesRead:
    def test_first_frame_and_stride_select_the_frames_and_their_indices(
        self, states, tmp_path
    ):
        out = cluster(
            tmp_path,
            **request(states, rmsd_cutoff_angstrom=1.0, first_frame=5, stride=3),
        )
        read = list(range(5, 45, 3))
        assert out["frames_read"] == len(read) and out["frames_in_file"] == 45
        for c in out["clusters"]:
            assert (c["center_frame"] - 5) % 3 == 0
        sizes = {s: sum(states["scales"][i] == s for i in read) for s in STATES}
        assert sorted((c["size"] for c in out["clusters"]), reverse=True) == sorted(
            (n for n in sizes.values() if n), reverse=True
        )
        rows = Path(out["assignments"]).read_text().splitlines()[1:]
        assert [int(r.split(",")[0]) for r in rows] == read

    def test_a_trajectory_of_more_than_5000_frames_asks_for_a_larger_stride(
        self, states, tmp_path
    ):
        many = states["directory"] / "many.dcd"
        write_dcd(many, [states["frames"][0]] * 5001)
        message = error_of(tmp_path, **{**request(states), "trajectory": str(many)})
        assert "5001 frames" in message and "raise stride to at least 2" in message
        out = cluster(
            tmp_path,
            "strided",
            **{**request(states), "trajectory": str(many), "stride": 2},
        )
        assert out["frames_read"] == 2501 and out["cluster_count"] == 1

    def test_fewer_than_2_frames_are_refused(self, states, tmp_path):
        assert "leave 1" in error_of(tmp_path, **request(states, first_frame=44))


class TestRepresentatives:
    def test_the_largest_clusters_get_a_center_structure_and_the_others_none(
        self, states, tmp_path
    ):
        out = cluster(
            tmp_path, **request(states, rmsd_cutoff_angstrom=1.0, representatives=2)
        )
        assert [c["structure"] is not None for c in out["clusters"]] == [
            True,
            True,
            False,
        ]

    def test_a_center_structure_holds_the_coordinates_of_its_frame(
        self, states, tmp_path
    ):
        out = cluster(
            tmp_path, **request(states, rmsd_cutoff_angstrom=1.0, representatives=1)
        )
        center = out["clusters"][0]["center_frame"]
        written = md.load_pdb(out["clusters"][0]["structure"])
        assert written.n_atoms == states["mdtop"].n_atoms
        assert np.abs(written.xyz[0] * 10 - states["frames"][center]).max() < 0.005

    def test_no_representatives_writes_no_structure(self, states, tmp_path):
        out = cluster(
            tmp_path, **request(states, rmsd_cutoff_angstrom=1.0, representatives=0)
        )
        assert all(c["structure"] is None for c in out["clusters"])


class TestImaging:
    @pytest.fixture(scope="class")
    def wrapped(self, trpcage):
        """Six copies of one structure, 4 whole in the box and 2 straddling its edge, stored wrapped."""
        ca = trpcage["mdtop"].select(SELECTIONS["alpha-carbons"])
        base = trpcage["xyz"]
        frames = []
        for straddling in (False, False, False, False, True, True):
            x = 0.0 if straddling else BOX / 2
            moved = base - base[ca].mean(axis=0) + np.array([x, BOX / 2, BOX / 2])
            frames.append(moved % BOX)
        path = trpcage["directory"] / "wrapped.dcd"
        write_dcd(path, frames, cell=BOX)
        return {"trajectory": str(path), "topology": str(trpcage["topology"])}

    def test_a_structure_torn_across_the_box_forms_its_own_cluster_as_stored(
        self, wrapped, tmp_path
    ):
        out = cluster(tmp_path, **wrapped, rmsd_cutoff_angstrom=1.0)
        assert [c["size"] for c in out["clusters"]] == [4, 2]

    def test_whole_molecule_imaging_puts_it_back_in_one_cluster(
        self, wrapped, tmp_path
    ):
        out = cluster(
            tmp_path, **wrapped, rmsd_cutoff_angstrom=1.0, imaging="whole-molecules"
        )
        assert [c["size"] for c in out["clusters"]] == [6]

    def test_whole_molecule_imaging_needs_a_periodic_cell(self, states, tmp_path):
        assert "no periodic cell" in error_of(
            tmp_path, **request(states, imaging="whole-molecules")
        )


class TestRefusals:
    def test_a_selection_with_fewer_than_3_atoms_is_refused(self, trpcage, tmp_path):
        two = trpcage["directory"] / "two"
        two.mkdir(exist_ok=True)
        protein = md.load_pdb(str(trpcage["directory"] / "topology.pdb"))
        keep = [a.index for a in protein.topology.atoms if a.residue.index < 2]
        small = protein.atom_slice(keep)
        small.save_pdb(str(two / "small.pdb"))
        write_dcd(two / "small.dcd", [small.xyz[0] * 10] * 3)
        message = error_of(
            tmp_path, topology=str(two / "small.pdb"), trajectory=str(two / "small.dcd")
        )
        assert "matches 2 atom(s)" in message

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("rmsd_cutoff_angstrom", 0.01),
            ("stride", 0),
            ("first_frame", -1),
            ("representatives", 51),
        ],
    )
    def test_a_value_outside_its_range_is_refused_by_name(
        self, states, tmp_path, name, value
    ):
        assert name in error_of(tmp_path, **request(states, **{name: value}))
