"""simulation.inspect_trajectory against trajectories whose answers are known.

The synthetic cases build a DCD from Trp-cage's deposited coordinates by exact
transformations, so every number the capability returns has a value that can be
written down without running it:

* a rigid rotation and translation of every atom must give an alpha-carbon RMSD of
  zero after the fit;
* scaling every atom about the alpha-carbon centroid by s leaves the best fit as
  the identity, so the RMSD is (s - 1) times the radius of gyration of the alpha
  carbons and that radius is s times the original;
* an angle is invariant under both, a distance scales with s, and a contact turns
  on when s times the distance falls under the cutoff;
* the fluctuation of alpha carbon i about its mean is its distance from the
  centroid times the population standard deviation of the scale factors;
* a protein wrapped across a periodic box looks torn apart as stored and is put
  back together by whole-molecule imaging; its nearest approach to its own image
  is found by brute force over all 26 neighbouring cells and compared.

One more case runs the real OpenMM implicit-solvent dynamics (set
ALAMBIQ_ENV_OPENMM to the pinned openmm-ensemble Python) and inspects the
trajectory it wrote, to check that the two capabilities agree about the files.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
md = pytest.importorskip("mdtraj")
gemmi = pytest.importorskip("gemmi")

STRUCTURE = Path(__file__).resolve().parents[1] / "fixtures/structures/1L2Y.cif"
SCALES = [1.0, 1.0, 1.1, 0.9, 1.0, 1.0]
STEP0, INTERVAL, DT_PS = 1000, 100, 0.002
ENV = os.environ.get("ALAMBIQ_ENV_OPENMM", "")


def run(steps, tmp_path, name="run"):
    from alambiq.api import run as execute

    return execute({"id": "ins", "steps": steps}, run_dir=tmp_path / name)


def ok(outcome, step="ins"):
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs(step)


def rotation(axis=(1.0, 2.0, 3.0), degrees=37.0):
    axis = np.asarray(axis) / np.linalg.norm(axis)
    a = np.radians(degrees)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def write_pdb(path: Path, xyz_angstrom, topology, cell: float | None = None) -> None:
    traj = md.Trajectory(np.asarray(xyz_angstrom)[None] / 10, topology)
    if cell:
        traj.unitcell_lengths = np.full((1, 3), cell / 10)
        traj.unitcell_angles = np.full((1, 3), 90.0)
    traj.save_pdb(str(path))


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


def write_state(path: Path, frames: int, *, extra_first: bool, volume: bool = False):
    count = frames + int(extra_first)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["step", "time_ps", "potential_kj_mol", "kinetic_kj_mol", "total_kj_mol"]
            + ["temperature_kelvin"]
            + (["volume_nm3"] if volume else [])
        )
        for k in range(count):
            step = STEP0 + INTERVAL * (k + (0 if extra_first else 1))
            writer.writerow(
                [step, step * DT_PS, -1000.0 - 10 * k, 50.0, -950.0 - 10 * k, 300.0 + k]
                + ([30.0 + 0.1 * k] if volume else [])
            )


@pytest.fixture(scope="module")
def trpcage(tmp_path_factory):
    """Trp-cage's first model as a topology PDB, with a three-atom ligand beside it."""
    directory = tmp_path_factory.mktemp("trpcage")
    structure = gemmi.read_structure(str(STRUCTURE))
    structure.setup_entities()
    del structure[1:]
    bare = directory / "bare.pdb"
    structure.write_pdb(str(bare))
    protein = md.load_pdb(str(bare))
    top = protein.topology.copy()
    chain = top.add_chain()
    residue = top.add_residue("LIG", chain, resSeq=100)
    carbon, oxygen = md.element.carbon, md.element.oxygen
    atoms = [
        top.add_atom("C1", carbon, residue),
        top.add_atom("C2", carbon, residue),
        top.add_atom("O1", oxygen, residue),
    ]
    top.add_bond(atoms[0], atoms[1])
    top.add_bond(atoms[1], atoms[2])
    base = protein.xyz[0] * 10
    edge = base[:, 0].max() + 3.5
    ligand = np.array(
        [[edge, 0, 0], [edge + 1.5, 0, 0], [edge + 3.0, 0, 0]]
    ) + np.array([0.0, *base[:, 1:].mean(axis=0)])
    write_pdb(directory / "with_ligand.pdb", np.vstack([base, ligand]), top)
    # Read back what was written, so the expected values use the same rounding.
    written = md.load_pdb(str(directory / "with_ligand.pdb"))
    write_pdb(directory / "protein.pdb", base, protein.topology)
    return {
        "directory": directory,
        "protein": directory / "protein.pdb",
        "ligand_pdb": directory / "with_ligand.pdb",
        "xyz": written.xyz[0] * 10,
        "topology": written.topology,
        "protein_xyz": md.load_pdb(str(directory / "protein.pdb")).xyz[0] * 10,
        "protein_topology": md.load_pdb(str(directory / "protein.pdb")).topology,
    }


def transform(xyz, scale, ca, *, rotate=False):
    centre = xyz[ca].mean(axis=0)
    moved = centre + scale * (xyz - centre)
    if rotate:
        moved = (moved - centre) @ rotation().T + centre + np.array([5.0, -7.0, 3.0])
    return moved


def selector(topology, residue_number, name):
    atom = next(
        a
        for a in topology.atoms
        if a.residue.resSeq == residue_number and a.name == name
    )
    return atom.index, {
        "chain_id": atom.residue.chain.chain_id or str(atom.residue.chain.index),
        "residue_number": residue_number,
        "residue_name": atom.residue.name,
        "atom_name": name,
    }


@pytest.fixture(scope="module")
def scaled(trpcage):
    """Six frames: the structure, a rigid copy, 1.1x, 0.9x, and the structure twice."""
    top = trpcage["topology"]
    ca = top.select("protein and name CA")
    frames = [
        transform(trpcage["xyz"], s, ca, rotate=(i == 1)) for i, s in enumerate(SCALES)
    ]
    directory = trpcage["directory"]
    write_dcd(directory / "scaled.dcd", frames)
    write_state(directory / "scaled_plus.csv", len(frames), extra_first=True)
    write_state(directory / "scaled_equal.csv", len(frames), extra_first=False)
    return {"frames": frames, "ca": ca, **trpcage}


def request(scaled, **params):
    return {
        "id": "ins",
        "capability": "simulation.inspect_trajectory",
        "params": {
            "topology": str(scaled["ligand_pdb"]),
            "trajectory": str(scaled["directory"] / "scaled.dcd"),
            "state_data": str(scaled["directory"] / "scaled_plus.csv"),
            **params,
        },
    }


def arrays(out):
    return np.load(out["arrays"])


def radius_of_gyration(xyz, ca):
    points = xyz[ca]
    return float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))


class TestKnownGeometry:
    @pytest.fixture(scope="class")
    def outcome(self, scaled, tmp_path_factory):
        d1, a1 = selector(scaled["topology"], 3, "CA")
        d2, a2 = selector(scaled["topology"], 15, "CA")
        _, n = selector(scaled["topology"], 2, "N")
        _, c = selector(scaled["topology"], 2, "CA")
        _, r = selector(scaled["topology"], 2, "C")
        x = scaled["xyz"]
        scaled["d0"] = float(np.linalg.norm(x[d1] - x[d2]))
        scaled["indices"] = (d1, d2)
        observables = [
            {"id": "span", "kind": "distance", "atoms": [a1, a2]},
            {"id": "bend", "kind": "angle", "atoms": [n, c, r]},
            {
                "id": "closed",
                "kind": "contact",
                "atoms": [a1, a2],
                "cutoff_angstrom": 0.95 * scaled["d0"],
            },
        ]
        return run(
            [request(scaled, observables=observables)],
            tmp_path_factory.mktemp("known"),
        )

    def test_the_run_reports_its_frames_atoms_and_fit(self, outcome, scaled):
        out = ok(outcome)
        assert out["frames"] == 6
        assert out["atoms"] == len(scaled["xyz"])
        assert out["fit_atoms"] == 20
        assert out["reference"] == "first saved frame"

    def test_the_rmsd_is_zero_for_a_rigid_motion_and_s_minus_one_times_rg_for_scaling(
        self, outcome, scaled
    ):
        rg0 = radius_of_gyration(scaled["xyz"], scaled["ca"])
        expected = [abs(s - 1) * rg0 for s in SCALES]
        values = arrays(ok(outcome))["values"]
        assert values[:, 0] == pytest.approx(expected, abs=5e-3)
        assert values[1, 0] < 5e-3 and values[2, 0] == pytest.approx(
            0.1 * rg0, abs=5e-3
        )

    def test_the_radius_of_gyration_scales_with_the_factor(self, outcome, scaled):
        rg0 = radius_of_gyration(scaled["xyz"], scaled["ca"])
        values = arrays(ok(outcome))["values"]
        assert values[:, 1] == pytest.approx([s * rg0 for s in SCALES], abs=5e-3)

    def test_a_distance_scales_and_an_angle_does_not(self, outcome, scaled):
        d0 = scaled["d0"]
        values = arrays(ok(outcome))["values"]
        assert values[:, 2] == pytest.approx([s * d0 for s in SCALES], abs=5e-3)
        x = scaled["xyz"]
        n, c, r = (
            selector(scaled["topology"], 2, name)[0] for name in ("N", "CA", "C")
        )
        u, v = x[n] - x[c], x[r] - x[c]
        angle = np.degrees(np.arccos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))
        assert values[:, 3] == pytest.approx([angle] * 6, abs=0.05)

    def test_a_contact_is_on_only_where_the_scaled_distance_is_under_the_cutoff(
        self, outcome
    ):
        values = arrays(ok(outcome))["values"]
        assert values[:, 4].tolist() == [0, 0, 0, 1, 0, 0]

    def test_the_observable_summaries_match_the_arrays(self, outcome):
        out = ok(outcome)
        values = arrays(out)["values"]
        assert [o["id"] for o in out["observables"]] == [
            "ca_rmsd_angstrom",
            "ca_radius_gyration_angstrom",
            "span",
            "bend",
            "closed",
        ]
        span = next(o for o in out["observables"] if o["id"] == "span")
        assert span["unit"] == "angstrom" and span["mean"] == pytest.approx(
            values[:, 2].mean()
        )
        assert span["standard_deviation"] == pytest.approx(values[:, 2].std())
        assert span["minimum"] == pytest.approx(values[:, 2].min())
        assert span["maximum"] == pytest.approx(values[:, 2].max())
        assert (span["first"], span["last"]) == pytest.approx(
            (values[0, 2], values[-1, 2])
        )
        units = {o["id"]: o["unit"] for o in out["observables"]}
        assert units["bend"] == "degree" and units["closed"] == "fraction"

    def test_the_fluctuation_is_the_distance_from_the_centre_times_the_spread_of_scales(
        self, outcome, scaled
    ):
        x = scaled["xyz"][scaled["ca"]]
        radial = np.linalg.norm(x - x.mean(axis=0), axis=1)
        expected = radial * np.std(SCALES)
        out = ok(outcome)
        assert out["rmsf_mean_angstrom"] == pytest.approx(expected.mean(), abs=5e-3)
        assert out["rmsf_max_angstrom"] == pytest.approx(expected.max(), abs=5e-3)
        worst = int(np.argmax(expected))
        atom = scaled["topology"].atom(int(scaled["ca"][worst]))
        assert (
            out["rmsf_max_residue"]
            == f"{atom.residue.name}{atom.residue.resSeq} chain A"
        )

    def test_the_state_samples_pair_with_the_frames_after_the_extra_first_one(
        self, outcome
    ):
        out = ok(outcome)
        steps = np.load(out["arrays"])["steps"]
        assert steps.tolist() == [STEP0 + INTERVAL * k for k in range(1, 7)]
        assert out["first_time_ps"] == pytest.approx((STEP0 + INTERVAL) * DT_PS)
        assert out["last_time_ps"] == pytest.approx((STEP0 + 6 * INTERVAL) * DT_PS)
        assert out["mean_temperature_kelvin"] == pytest.approx(
            np.mean([301, 302, 303, 304, 305, 306])
        )
        assert out["mean_potential_kj_mol"] == pytest.approx(
            np.mean([-1010, -1020, -1030, -1040, -1050, -1060])
        )
        assert out["mean_volume_nm3"] is None

    def test_the_plotted_series_covers_every_frame_when_there_are_few(self, outcome):
        out = ok(outcome)
        assert len(out["plot_time_ps"]) == 6
        assert out["plot_rmsd_angstrom"][0] < 5e-3

    def test_the_details_file_holds_the_series_and_the_method(self, outcome):
        details = json.loads(Path(ok(outcome)["details"]).read_text())
        assert details["frame_count"] == 6
        assert [row["step"] for row in details["series"]] == [
            STEP0 + INTERVAL * k for k in range(1, 7)
        ]
        assert details["method"]["reference"].startswith("first saved frame")


class TestStatesAndPreviews:
    def test_state_data_with_one_sample_per_frame_is_used_whole(self, scaled, tmp_path):
        outcome = run(
            [
                request(
                    scaled,
                    state_data=str(scaled["directory"] / "scaled_equal.csv"),
                )
            ],
            tmp_path,
        )
        out = ok(outcome)
        assert out["first_time_ps"] == pytest.approx((STEP0 + INTERVAL) * DT_PS)
        assert out["mean_temperature_kelvin"] == pytest.approx(
            np.mean([300, 301, 302, 303, 304, 305])
        )

    def test_a_reference_structure_replaces_the_first_frame(self, scaled, tmp_path):
        ca = scaled["ca"]
        reference = transform(scaled["xyz"], 1.1, ca)
        path = scaled["directory"] / "reference.pdb"
        write_pdb(path, reference, scaled["topology"])
        out = ok(run([request(scaled, reference_structure=str(path))], tmp_path))
        rg0 = radius_of_gyration(scaled["xyz"], ca)
        values = arrays(out)["values"][:, 0]
        # The 1.1x frame is the reference; the original is 0.1 Rg away from it.
        assert values[2] < 5e-3
        assert values[0] == pytest.approx(0.1 * rg0, abs=5e-3)
        assert out["reference"] == "supplied structure"

    def test_reference_values_are_the_measures_taken_on_the_reference_structure(
        self, scaled, tmp_path
    ):
        ca = scaled["ca"]
        d1, a1 = selector(scaled["topology"], 3, "CA")
        d2, a2 = selector(scaled["topology"], 15, "CA")
        reference = transform(scaled["xyz"], 1.1, ca)
        path = scaled["directory"] / "reference_values.pdb"
        write_pdb(path, reference, scaled["topology"])
        out = ok(
            run(
                [
                    request(
                        scaled,
                        reference_structure=str(path),
                        observables=[
                            {"id": "span", "kind": "distance", "atoms": [a1, a2]}
                        ],
                    )
                ],
                tmp_path,
            )
        )
        found = json.loads(Path(out["details"]).read_text())["reference_values"]
        written = md.load_pdb(str(path)).xyz[0] * 10
        assert found["source"] == "supplied structure"
        assert found["values"]["ca_rmsd_angstrom"] == 0.0
        assert found["values"]["ca_radius_gyration_angstrom"] == pytest.approx(
            radius_of_gyration(written, ca), abs=5e-3
        )
        assert found["values"]["span"] == pytest.approx(
            np.linalg.norm(written[d1] - written[d2]), abs=5e-3
        )

    def test_without_a_reference_the_first_frame_supplies_the_reference_values(
        self, scaled, tmp_path
    ):
        out = ok(run([request(scaled)], tmp_path))
        found = json.loads(Path(out["details"]).read_text())["reference_values"]
        values = arrays(out)["values"]
        assert found["source"] == "first saved frame"
        assert found["values"]["ca_radius_gyration_angstrom"] == pytest.approx(
            values[0, 1]
        )

    def test_the_preview_keeps_the_chosen_frames_and_atoms_after_the_fit(
        self, scaled, tmp_path
    ):
        keep = [0, 5, 12, 40]
        out = ok(
            run(
                [
                    request(
                        scaled,
                        display_atom_indices=keep,
                        max_preview_frames=3,
                        max_plot_frames=100,
                        chunk_frames=2,
                    )
                ],
                tmp_path,
            )
        )
        details = json.loads(Path(out["details"]).read_text())
        assert details["preview_frame_indices"] == [0, 2, 5]
        coords = np.asarray(details["coordinates_angstrom"])
        assert coords.shape == (3, 4, 3)
        # Frame 0 is the reference; frame 5 is the same structure, fitted back onto it.
        assert coords[2] == pytest.approx(coords[0], abs=5e-3)
        assert coords[0] == pytest.approx(scaled["xyz"][keep], abs=5e-3)
        assert [a["source_atom_index"] for a in details["atoms"]] == keep
        assert details["streaming"]["chunk_frames"] == 2

    def test_a_long_series_is_thinned_for_the_plot_but_not_for_the_arrays(
        self, scaled, tmp_path
    ):
        frames = [scaled["frames"][i % 6] for i in range(600)]
        directory = scaled["directory"] / "long"
        directory.mkdir(exist_ok=True)
        write_dcd(directory / "long.dcd", frames)
        write_state(directory / "long.csv", 600, extra_first=True)
        out = ok(
            run(
                [
                    request(
                        scaled,
                        trajectory=str(directory / "long.dcd"),
                        state_data=str(directory / "long.csv"),
                        max_plot_frames=100,
                    )
                ],
                tmp_path,
            )
        )
        assert out["frames"] == 600
        assert len(arrays(out)["values"]) == 600
        assert len(out["plot_time_ps"]) <= 200
        details = json.loads(Path(out["details"]).read_text())
        assert len(details["series"]) < 600


class TestLigandContacts:
    @pytest.fixture(scope="class")
    def contacts(self, scaled, tmp_path_factory):
        top = scaled["topology"]
        ligand = top.n_residues - 1
        cutoff = 5.0
        outcome = run(
            [
                request(
                    scaled,
                    ligand_contacts={
                        "ligand_residue_index": ligand,
                        "cutoff_angstrom": cutoff,
                        "periodic": False,
                    },
                )
            ],
            tmp_path_factory.mktemp("contacts"),
        )
        return ok(outcome), ligand, cutoff

    def test_each_residues_contact_fraction_matches_a_brute_force_distance_search(
        self, contacts, scaled
    ):
        out, ligand, cutoff = contacts
        top = scaled["topology"]
        ligand_atoms = [a.index for a in top.residue(ligand).atoms]
        counts = {}
        closest = {}
        for residue in top.residues:
            if not residue.is_protein:
                continue
            heavy = [a.index for a in residue.atoms if a.element.symbol != "H"]
            minima = []
            for frame in scaled["frames"]:
                d = np.linalg.norm(
                    frame[heavy][:, None] - frame[ligand_atoms][None], axis=2
                )
                minima.append(d.min())
            counts[residue.index] = np.mean(np.asarray(minima) <= cutoff)
            closest[residue.index] = min(minima)
        expected = {k: v for k, v in counts.items() if v > 0}
        assert expected, "the ligand should touch something"
        found = {c["residue_index"]: c for c in out["ligand_contacts"]}
        assert set(found) == set(expected)
        for index, fraction in expected.items():
            assert found[index]["contact_fraction"] == pytest.approx(fraction)
            assert found[index]["minimum_distance_angstrom"] == pytest.approx(
                closest[index], abs=5e-3
            )

    def test_the_contacts_are_listed_most_frequent_first(self, contacts):
        fractions = [c["contact_fraction"] for c in contacts[0]["ligand_contacts"]]
        assert fractions == sorted(fractions, reverse=True)

    def test_a_ligand_that_is_a_protein_residue_is_refused(self, scaled, tmp_path):
        outcome = run(
            [
                request(
                    scaled,
                    ligand_contacts={
                        "ligand_residue_index": 0,
                        "cutoff_angstrom": 5.0,
                        "periodic": False,
                    },
                )
            ],
            tmp_path,
        )
        assert not outcome.ok
        assert "non-protein, non-water ligand" in outcome.steps["ins"].error

    def test_a_periodic_measurement_needs_a_cell(self, scaled, tmp_path):
        outcome = run(
            [
                request(
                    scaled,
                    ligand_contacts={
                        "ligand_residue_index": scaled["topology"].n_residues - 1,
                        "cutoff_angstrom": 5.0,
                        "periodic": True,
                    },
                )
            ],
            tmp_path,
        )
        assert not outcome.ok
        assert "unit cell" in outcome.steps["ins"].error


BOX = 40.0


@pytest.fixture(scope="module")
def wrapped(trpcage):
    """Trp-cage straddling a box edge, stored wrapped, with the whole copy as reference."""
    top = trpcage["protein_topology"]
    ca = top.select("protein and name CA")
    base = trpcage["protein_xyz"]
    unwrapped = []
    for scale in (1.0, 1.02, 0.98):
        moved = transform(base, scale, ca)
        # Put the alpha-carbon centroid on the plane x = 0 of the box.
        moved = moved - np.array([moved[ca].mean(axis=0)[0], -10.0, -10.0])
        unwrapped.append(moved)
    stored = [frame % BOX for frame in unwrapped]
    directory = trpcage["directory"]
    write_dcd(directory / "wrapped.dcd", stored, cell=BOX)
    write_state(directory / "wrapped.csv", 3, extra_first=True, volume=True)
    reference = directory / "whole.pdb"
    write_pdb(reference, unwrapped[0], top)
    return {
        "unwrapped": unwrapped,
        "stored": stored,
        "reference": reference,
        "ca": ca,
        **trpcage,
    }


def wrapped_request(wrapped, **params):
    return {
        "id": "ins",
        "capability": "simulation.inspect_trajectory",
        "params": {
            "topology": str(wrapped["protein"]),
            "trajectory": str(wrapped["directory"] / "wrapped.dcd"),
            "state_data": str(wrapped["directory"] / "wrapped.csv"),
            "reference_structure": str(wrapped["reference"]),
            **params,
        },
    }


class TestPeriodicBoundaries:
    def test_a_protein_wrapped_across_the_box_looks_torn_apart_as_stored(
        self, wrapped, tmp_path
    ):
        outcome = run([wrapped_request(wrapped)], tmp_path)
        out = ok(outcome)
        assert arrays(out)["values"][0, 0] > 5.0
        assert "used as stored" in " ".join(outcome.steps["ins"].notes)
        assert out["mean_volume_nm3"] == pytest.approx(np.mean([30.1, 30.2, 30.3]))

    def test_whole_molecule_imaging_restores_it(self, wrapped, tmp_path):
        out = ok(run([wrapped_request(wrapped, imaging="whole-molecules")], tmp_path))
        ca = wrapped["ca"]
        rg0 = radius_of_gyration(wrapped["unwrapped"][0], ca)
        values = arrays(out)["values"][:, 0]
        assert values == pytest.approx([0.0, 0.02 * rg0, 0.02 * rg0], abs=5e-3)

    def test_periodic_observables_use_the_minimum_image(self, wrapped, tmp_path):
        top = wrapped["protein_topology"]
        i, a = selector(top, 3, "CA")
        j, b = selector(top, 15, "CA")
        observables = [{"id": "span", "kind": "distance", "atoms": [a, b]}]
        plain = ok(
            run([wrapped_request(wrapped, observables=observables)], tmp_path, "plain")
        )
        image = ok(
            run(
                [
                    wrapped_request(
                        wrapped, observables=observables, periodic_observables=True
                    )
                ],
                tmp_path,
                "image",
            )
        )
        expected = [
            float(np.linalg.norm(frame[i] - frame[j])) for frame in wrapped["unwrapped"]
        ]
        # Periodic distances match the unwrapped geometry whether or not the pair was split.
        assert arrays(image)["values"][:, 2] == pytest.approx(expected, abs=5e-3)
        assert arrays(plain)["values"][:, 2] != pytest.approx(expected, abs=5e-3)

    def test_the_nearest_image_distance_matches_a_search_over_every_neighbouring_cell(
        self, wrapped, tmp_path
    ):
        nearest = min(
            np.linalg.norm(
                frame[:, None] - (frame + np.array(shift) * BOX)[None], axis=2
            ).min()
            for frame in wrapped["unwrapped"]
            for shift in itertools.product((-1, 0, 1), repeat=3)
            if any(shift)
        )
        cutoff_nm = nearest / 10
        checked = ok(
            run(
                [
                    wrapped_request(
                        wrapped,
                        imaging="whole-molecules",
                        protein_image_check={"cutoff_nm": cutoff_nm * 0.9, "frames": 3},
                    )
                ],
                tmp_path,
                "clear",
            )
        )
        check = checked["image_check"]
        assert check["checked_frames"] == 3 and check["all_frames_checked"]
        assert check["passed"] and check["within_cutoff_frames"] == 0
        assert check["minimum_distance_nm"] == pytest.approx(cutoff_nm, abs=2e-3)

    def test_a_protein_inside_the_cutoff_of_its_image_fails_the_check(
        self, wrapped, tmp_path
    ):
        outcome = run(
            [
                wrapped_request(
                    wrapped,
                    imaging="whole-molecules",
                    protein_image_check={"cutoff_nm": 2.9, "frames": 2},
                )
            ],
            tmp_path,
        )
        out = ok(outcome)
        assert out["image_check"]["passed"] is False
        assert out["image_check"]["within_cutoff_frames"] == 2
        assert out["image_check"]["checked_frames"] == 2
        assert not out["image_check"]["all_frames_checked"]
        assert "too small" in " ".join(outcome.steps["ins"].notes)

    def test_periodic_analysis_without_a_cell_is_refused(self, scaled, tmp_path):
        outcome = run([request(scaled, imaging="whole-molecules")], tmp_path)
        assert not outcome.ok
        assert "positive-volume cell" in outcome.steps["ins"].error


class TestRefusals:
    def test_state_data_with_the_wrong_number_of_samples_is_refused(
        self, scaled, tmp_path
    ):
        short = scaled["directory"] / "short.csv"
        write_state(short, 4, extra_first=True)
        outcome = run([request(scaled, state_data=str(short))], tmp_path)
        assert not outcome.ok
        assert "6 frames but the state data has 5 samples" in outcome.steps["ins"].error

    def test_state_data_with_other_columns_is_refused(self, scaled, tmp_path):
        bad = scaled["directory"] / "bad.csv"
        bad.write_text("step,energy\n1,2\n")
        outcome = run([request(scaled, state_data=str(bad))], tmp_path)
        assert not outcome.ok
        assert "State data needs the columns" in outcome.steps["ins"].error

    def test_state_steps_that_do_not_increase_are_refused(self, scaled, tmp_path):
        bad = scaled["directory"] / "backwards.csv"
        bad.write_text(
            "step,time_ps,potential_kj_mol,kinetic_kj_mol,total_kj_mol,temperature_kelvin\n"
            "200,0.4,1,1,2,300\n100,0.2,1,1,2,300\n"
        )
        outcome = run([request(scaled, state_data=str(bad))], tmp_path)
        assert not outcome.ok
        assert "must increase" in outcome.steps["ins"].error

    @pytest.mark.parametrize(
        ("changes", "message"),
        [
            ({"max_preview_frames": 1}, "max_preview_frames must be"),
            ({"max_plot_frames": 50}, "max_plot_frames must be"),
            ({"chunk_frames": 0}, "chunk_frames must be"),
            ({"display_atom_indices": []}, "display_atom_indices must select"),
            ({"display_atom_indices": [3, 3]}, "display_atom_indices must select"),
            (
                {"protein_image_check": {"cutoff_nm": 0.0, "frames": 1}},
                "cutoff_nm must be in",
            ),
            (
                {
                    "ligand_contacts": {
                        "ligand_residue_index": 20,
                        "cutoff_angstrom": 0.0,
                        "periodic": False,
                    }
                },
                "cutoff_angstrom in",
            ),
        ],
    )
    def test_settings_that_cannot_be_honoured_are_refused_in_plain_words(
        self, scaled, tmp_path, changes, message
    ):
        outcome = run([request(scaled, **changes)], tmp_path)
        assert not outcome.ok
        assert message in outcome.steps["ins"].error

    def test_an_observable_that_matches_no_atom_is_refused(self, scaled, tmp_path):
        ghost = {
            "chain_id": "A",
            "residue_number": 999,
            "residue_name": "ALA",
            "atom_name": "CA",
        }
        _, real = selector(scaled["topology"], 3, "CA")
        outcome = run(
            [
                request(
                    scaled,
                    observables=[
                        {"id": "bad", "kind": "distance", "atoms": [ghost, real]}
                    ],
                )
            ],
            tmp_path,
        )
        assert not outcome.ok
        assert "Exact atom selection has 0 matches" in outcome.steps["ins"].error

    def test_an_observable_that_reuses_a_built_in_name_is_refused(
        self, scaled, tmp_path
    ):
        _, a = selector(scaled["topology"], 3, "CA")
        _, b = selector(scaled["topology"], 15, "CA")
        outcome = run(
            [
                request(
                    scaled,
                    observables=[
                        {
                            "id": "ca_rmsd_angstrom",
                            "kind": "distance",
                            "atoms": [a, b],
                        }
                    ],
                )
            ],
            tmp_path,
        )
        assert not outcome.ok
        assert "collide with built-in" in outcome.steps["ins"].error

    def test_a_contact_without_a_cutoff_is_refused(self, scaled, tmp_path):
        _, a = selector(scaled["topology"], 3, "CA")
        _, b = selector(scaled["topology"], 15, "CA")
        outcome = run(
            [
                request(
                    scaled,
                    observables=[{"id": "c", "kind": "contact", "atoms": [a, b]}],
                )
            ],
            tmp_path,
        )
        assert not outcome.ok
        assert "cutoff_angstrom" in outcome.steps["ins"].error

    def test_a_topology_with_a_different_atom_count_is_refused(self, scaled, tmp_path):
        outcome = run(
            [
                request(
                    scaled,
                    topology=str(scaled["directory"] / "protein.pdb"),
                    trajectory=str(scaled["directory"] / "scaled.dcd"),
                )
            ],
            tmp_path,
        )
        # The ligand-free topology has three atoms fewer than the trajectory.
        assert not outcome.ok
        assert "disagree about the atoms" in outcome.steps["ins"].error


@pytest.mark.skipif(
    not (ENV and Path(ENV).is_file()),
    reason="set ALAMBIQ_ENV_OPENMM to run the OpenMM trajectory test",
)
class TestOnRealDynamics:
    @pytest.fixture(scope="class")
    def outcome(self, tmp_path_factory):
        os.environ["ALAMBIQ_ENV_OPENMM"] = ENV
        steps = [
            {
                "id": "prep",
                "capability": "structure.prepare_protein_system",
                "params": {
                    "structure": str(STRUCTURE),
                    "chain_id": "A",
                    "solvent": "gbn2",
                },
            },
            {
                "id": "md",
                "capability": "simulation.run_molecular_dynamics",
                "bind": {
                    "system": "prep.system_xml",
                    "state": "prep.state_xml",
                    "topology": "prep.topology_pdb",
                },
                "params": {
                    "ensemble": "implicit",
                    "thermalize_ps": 1.0,
                    "readiness_window_ps": 0,
                    "production_ps": 2.0,
                    "report_interval": 100,
                },
            },
            {
                "id": "ins",
                "capability": "simulation.inspect_trajectory",
                "bind": {
                    "topology": "prep.topology_pdb",
                    "trajectory": "md.trajectory",
                    "state_data": "md.state_data",
                },
            },
        ]
        return run(steps, tmp_path_factory.mktemp("real"))

    def test_the_frames_and_atoms_agree_with_the_run_that_wrote_them(self, outcome):
        md_out, out = ok(outcome, "md"), ok(outcome, "ins")
        assert out["frames"] == md_out["frames"] == 10
        assert out["atoms"] == md_out["particles"] == 304
        assert out["fit_atoms"] == 20

    def test_the_state_average_is_the_runs_own(self, outcome):
        md_out, out = ok(outcome, "md"), ok(outcome, "ins")
        # The inspection drops the sample at the start of production; the run keeps it.
        rows = list(csv.DictReader(open(md_out["state_data"])))
        assert len(rows) == 11
        assert out["mean_temperature_kelvin"] == pytest.approx(
            np.mean([float(r["temperature_kelvin"]) for r in rows[1:]])
        )
        assert out["first_time_ps"] == pytest.approx(float(rows[1]["time_ps"]))
        assert out["last_time_ps"] == pytest.approx(md_out["final_time_ps"])

    def test_two_picoseconds_of_a_small_protein_stay_near_the_start(self, outcome):
        out = ok(outcome, "ins")
        rmsd = next(o for o in out["observables"] if o["id"] == "ca_rmsd_angstrom")
        assert 0 < rmsd["maximum"] < 3.0
        assert out["rmsf_max_angstrom"] < 3.0
