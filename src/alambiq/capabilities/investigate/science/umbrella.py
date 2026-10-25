"""Joining recorded QM/MM umbrella windows into samples for the free-energy analysis.

Each window is one run of `simulation.run_qmmm_dynamics` under a harmonic bias on the
same distance-difference coordinate. Every window of every replica must come from one
Hamiltonian at one temperature along one coordinate, with centers increasing through the
windows and the same windows in every replica; the table of a window must be the one its
summary says it wrote.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from pathlib import Path
from typing import Any

MAX_FRAMES_PER_REPLICA = 100_000
MAX_FRAMES = 1_000_000


def _sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_layout(title: str, replicas: list[dict[str, Any]]) -> None:
    """The shape of the replica and window assignment, before any file is read."""
    if not title.strip():
        raise ValueError("title is empty; a non-empty title is required")
    if not 2 <= len(replicas) <= 20:
        raise ValueError(
            f"replicas holds {len(replicas)} entries; 2–20 independently sampled "
            "replicas are required"
        )
    ids: set[str] = set()
    order: list[str] | None = None
    for replica in replicas:
        if not replica["id"].strip() or replica["id"] in ids:
            raise ValueError(
                f"Replica id {replica['id']!r} is empty or repeated; replica ids must be "
                "distinct and non-empty"
            )
        ids.add(replica["id"])
        windows = [w["window"] for w in replica["windows"]]
        if not 2 <= len(windows) <= 100 or len(set(windows)) != len(windows):
            raise ValueError(
                f"Replica {replica['id']} has {len(windows)} windows, "
                f"{len(set(windows))} of them distinct; each replica needs 2–100 "
                "distinct, ordered windows"
            )
        if any(not w.strip() for w in windows):
            raise ValueError(
                f"Replica {replica['id']} has an empty window id; window ids must be "
                "non-empty"
            )
        if any(w["discard_frames"] < 0 for w in replica["windows"]):
            raise ValueError(
                f"Replica {replica['id']} has a negative discard_frames; the discarded "
                "prefix must be a non-negative number of frames"
            )
        if order is not None and order != windows:
            raise ValueError(
                f"Replica {replica['id']} does not list the same windows in the same "
                "order as the first replica; each replica must supply the same complete "
                "ordered windows"
            )
        order = windows


def assemble_samples(title: str, replicas: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the recorded windows against each other and join them.

    Each replica is `{"id", "windows": [{"window", "summary", "table", "discard_frames"}]}`
    with `summary` the parsed record of the run and `table` the path of its state table.
    """
    check_layout(title, replicas)
    windows: list[dict[str, Any]] | None = None
    common: tuple | None = None
    seeds: set[int] = set()
    physical: dict[str, Any] = {}
    total = 0
    out_replicas = []
    for replica in replicas:
        traces, definitions, count = [], [], 0
        for item in replica["windows"]:
            summary, table = item["summary"], Path(item["table"])
            if (
                summary.get("quantity") != "molecular-qmmm-trajectory"
                or summary.get("ensemble") != "NVT"
            ):
                raise ValueError(
                    f"{item['window']}: the record has quantity "
                    f"{summary.get('quantity')!r} and ensemble {summary.get('ensemble')!r}; "
                    "a window must be a recorded NVT QM/MM trajectory"
                )
            bias = summary["sampling"]["umbrella"]
            if bias is None:
                raise ValueError(
                    f"{item['window']}: this window was run without an umbrella bias"
                )
            sampling = summary["sampling"]
            identity = (
                summary["hamiltonian_sha256"],
                sampling["temperature_kelvin"],
                bias["pairs"],
            )
            if common is not None and identity != common:
                raise ValueError(
                    f"Window {item['window']} of replica {replica['id']} differs from the "
                    "first window in Hamiltonian, temperature or coordinate; all windows "
                    "and replicas require the same Hamiltonian, temperature and coordinate"
                )
            common = identity
            physical = {
                "system_sha256": summary["source_sha256"]["system_xml"],
            }
            if re.fullmatch(r"[0-9a-f]{64}", physical["system_sha256"]) is None:
                raise ValueError(
                    f"{item['window']}: the recorded system identity "
                    f"{physical['system_sha256']!r} is not a SHA-256 digest"
                )
            if sampling["seed"] in seeds:
                raise ValueError(
                    f"{item['window']}: seed {sampling['seed']} is used by another window; "
                    "each independently sampled window needs its own random seed"
                )
            seeds.add(sampling["seed"])
            interval = sampling["report_interval"] * sampling["timestep_fs"]
            if not math.isclose(summary["frame_interval_fs"], interval, rel_tol=1e-12):
                raise ValueError(
                    f"{item['window']}: the recorded frame interval "
                    f"{summary['frame_interval_fs']:g} fs differs from the interval "
                    f"{interval:g} fs of the sampling settings"
                )
            digest = _sha(table)
            if digest != summary["outputs_sha256"]["state.csv"]:
                raise ValueError(
                    f"{item['window']}: the state table is not the one its run wrote"
                )
            with table.open(newline="") as stream:
                coordinates = [
                    float(row["coordinate_angstrom"])
                    for row in csv.DictReader(stream)
                    if row["phase"] == "production"
                ]
            if len(coordinates) != summary["production_frames"]:
                raise ValueError(
                    f"{item['window']}: the table holds {len(coordinates)} production "
                    f"frames but the run recorded {summary['production_frames']}"
                )
            count += len(coordinates)
            total += len(coordinates)
            if count > MAX_FRAMES_PER_REPLICA or total > MAX_FRAMES:
                raise ValueError(
                    f"The windows hold {count} frames in replica {replica['id']} and "
                    f"{total} in all; at most {MAX_FRAMES_PER_REPLICA} frames per replica "
                    f"and {MAX_FRAMES} in all are allowed"
                )
            definitions.append(
                {
                    "id": item["window"],
                    "center_angstrom": bias["center_angstrom"],
                    "k_kj_mol_angstrom2": bias["coefficient_kj_mol_angstrom2"],
                }
            )
            traces.append(
                {
                    "window": item["window"],
                    "trajectory_sha256": digest,
                    "discard_frames": item["discard_frames"],
                    "frame_interval_fs": interval,
                    "coordinate_angstrom": coordinates,
                }
            )
        if windows is not None and definitions != windows:
            raise ValueError(
                f"Replica {replica['id']} differs from earlier replicas in the recorded "
                "centers or coefficients; corresponding windows need the same recorded "
                "centers and coefficients"
            )
        windows = definitions
        out_replicas.append({"id": replica["id"], "windows": traces})
    if common is None:
        raise ValueError("The replicas list no windows; at least one window is required")
    first, second = common[2]
    return {
        "schema_version": "0.1",
        "title": title,
        "system_sha256": physical["system_sha256"],
        "hamiltonian_sha256": common[0],
        "temperature_kelvin": common[1],
        "coordinate": {
            "unit": "angstrom",
            "definition": f"distance({first[0]},{first[1]})-distance({second[0]},{second[1]})",
            "positive_distance_atoms": first,
            "negative_distance_atoms": second,
        },
        "bias_convention": "K*(q-center)^2",
        "windows": windows,
        "replicas": out_replicas,
    }
