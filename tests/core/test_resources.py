"""Pinned resources: where they live, how they are checked, how they are fetched.

A tiny two-file resource stands in for the real ones; the checks are the same.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import pytest

from alambiq.core import resources

A, B = b"alpha bytes", b"beta bytes, a little longer"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(url: bool = False) -> dict:
    files = [
        {"path": "model/a.bin", "bytes": len(A), "sha256": sha(A)},
        {"path": "b.txt", "bytes": len(B), "sha256": sha(B)},
    ]
    if url:
        for f in files:
            f["url"] = "https://example.org/" + f["path"]
    return {"logical_id": "toy", "version": "1", "license": "CC0-1.0", "files": files}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ALAMBIQ_RESOURCE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ALAMBIQ_RESOURCE_TOY", raising=False)
    monkeypatch.setattr(resources, "spec", lambda name: record())
    return tmp_path / "home"


def install(home: Path) -> Path:
    root = resources.location("toy")
    (root / "model").mkdir(parents=True)
    (root / "model/a.bin").write_bytes(A)
    (root / "b.txt").write_bytes(B)
    return root


class TestLocation:
    def test_a_resource_lives_under_its_content_address(self, home):
        address = resources.content_address("toy")
        assert resources.location("toy") == home.resolve() / "sha256" / address

    def test_the_address_follows_the_files_not_the_version_label(
        self, home, monkeypatch
    ):
        first = resources.content_address("toy")
        changed = record()
        changed["files"][1]["sha256"] = sha(b"something else")
        monkeypatch.setattr(resources, "spec", lambda name: changed)
        assert resources.content_address("toy") != first

    def test_an_environment_variable_places_it_on_another_disk(
        self, home, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ALAMBIQ_RESOURCE_TOY", str(tmp_path / "elsewhere"))
        assert resources.location("toy") == (tmp_path / "elsewhere").resolve()

    def test_punctuation_in_a_name_becomes_underscores_in_the_variable(
        self, home, monkeypatch, tmp_path
    ):
        monkeypatch.setenv(
            "ALAMBIQ_RESOURCE_PFAM_38_2_PROFILES", str(tmp_path / "pfam")
        )
        assert resources.location("pfam-38.2-profiles") == (tmp_path / "pfam").resolve()

    def test_a_record_may_state_the_address_it_was_installed_under(
        self, home, monkeypatch
    ):
        stated = record()
        stated["content_address"] = "ab" * 32
        monkeypatch.setattr(resources, "spec", lambda name: stated)
        assert resources.content_address("toy") == "ab" * 32
        assert resources.location("toy") == home.resolve() / "sha256" / ("ab" * 32)

    def test_the_pinned_records_that_ship_have_distinct_addresses(self):
        names = [
            "esmc-300m",
            "esm3-small",
            "thermompnn",
            "pfam-38.2-profiles",
            "uniref50-2026_02",
            "p2rank-2.5.1",
        ]
        assert len({resources.content_address(n) for n in names}) == len(names)


class TestVerify:
    def test_a_missing_resource_says_how_to_provision_it(self, home):
        with pytest.raises(resources.ResourceMissing) as caught:
            resources.verify("toy")
        assert "alambiq resource provision toy" in str(caught.value)
        assert "ALAMBIQ_RESOURCE_TOY" in str(caught.value)

    def test_a_complete_resource_verifies_and_returns_its_root(self, home):
        root = install(home)
        assert resources.verify("toy") == root

    def test_a_file_of_the_wrong_size_is_named(self, home):
        root = install(home)
        (root / "b.txt").write_bytes(B + b"!")
        with pytest.raises(resources.ResourceMissing, match=r"b\.txt is \d+ bytes"):
            resources.verify("toy")

    def test_a_file_with_the_right_size_and_wrong_bytes_is_caught_by_its_digest(
        self, home
    ):
        root = install(home)
        (root / "b.txt").write_bytes(B[:-1] + b"?")
        with pytest.raises(resources.ResourceMissing, match="SHA-256"):
            resources.verify("toy")

    def test_a_missing_file_is_named(self, home):
        root = install(home)
        (root / "model/a.bin").unlink()
        with pytest.raises(resources.ResourceMissing, match=r"a\.bin is missing"):
            resources.verify("toy")

    def test_a_link_is_not_accepted_in_place_of_a_file(self, home, tmp_path):
        root = install(home)
        (root / "b.txt").unlink()
        outside = tmp_path / "outside.txt"
        outside.write_bytes(B)
        (root / "b.txt").symlink_to(outside)
        with pytest.raises(resources.ResourceMissing, match="not a regular file"):
            resources.verify("toy")

    def test_a_file_the_record_does_not_list_is_refused(self, home):
        root = install(home)
        (root / "extra.dat").write_bytes(b"x")
        with pytest.raises(resources.ResourceMissing, match=r"extra\.dat"):
            resources.verify("toy")

    def test_the_lock_file_a_provisioning_leaves_is_allowed(self, home):
        root = install(home)
        (root / "resource-lock.json").write_text("{}")
        assert resources.verify("toy") == root

    def test_a_second_check_does_not_hash_again(self, home, monkeypatch):
        install(home)
        resources.verify("toy")

        def refuse(path):
            raise AssertionError("hashed a file already verified")

        monkeypatch.setattr(resources, "_hash", refuse)
        resources.verify("toy")

    def test_a_file_changed_after_verification_is_hashed_and_caught(self, home):
        root = install(home)
        resources.verify("toy")
        # Same size, different bytes, and a later modification time.
        time.sleep(0.01)
        (root / "b.txt").write_bytes(B[:-1] + b"?")
        os.utime(root / "b.txt", ns=(time.time_ns(), time.time_ns() + 10**9))
        with pytest.raises(resources.ResourceMissing, match="SHA-256"):
            resources.verify("toy")

    def test_the_receipt_is_kept_beside_the_resources_not_inside_them(self, home):
        root = install(home)
        resources.verify("toy")
        receipts = list((home.resolve() / "verified").glob("*.json"))
        assert len(receipts) == 1
        assert sorted(p.name for p in root.iterdir()) == ["b.txt", "model"]
        assert json.loads(receipts[0].read_text())["root"] == str(root)


TREE = {
    "x/one.dat": b"first file",
    "x/sub/two.dat": b"second file, in a folder",
    "x/sub/three-4.dat": b"third",
}


def tree_record() -> dict:
    """Describe the tree the way a pinned record does: one digest over every file."""
    rows = [
        {"path": path.removeprefix("x/"), "bytes": len(data), "sha256": sha(data)}
        for path, data in sorted(TREE.items(), key=lambda kv: Path(kv[0]))
    ]
    return {
        "path": "x",
        "file_count": len(rows),
        "bytes": sum(r["bytes"] for r in rows),
        "sha256": hashlib.sha256(
            json.dumps(
                rows, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ).encode()
        ).hexdigest(),
    }


@pytest.fixture
def forest(tmp_path, monkeypatch):
    """A resource of one file and one tree of three files."""
    monkeypatch.setenv("ALAMBIQ_RESOURCE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ALAMBIQ_RESOURCE_TOY", raising=False)
    pinned = {**record(), "files": record()["files"][1:], "trees": [tree_record()]}
    monkeypatch.setattr(resources, "spec", lambda name: pinned)
    root = resources.location("toy")
    for path, data in TREE.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    (root / "b.txt").write_bytes(B)
    return root


class TestTrees:
    def test_a_resource_with_a_tree_verifies_and_returns_its_root(self, forest):
        assert resources.verify("toy") == forest

    def test_the_tree_digest_is_part_of_what_the_resource_is(self, forest, monkeypatch):
        first = resources.identity("toy")
        previous = resources.content_address("toy")
        changed = {
            **resources.spec("toy"),
            "trees": [{**tree_record(), "sha256": "0" * 64}],
        }
        monkeypatch.setattr(resources, "spec", lambda name: changed)
        assert resources.identity("toy") != first
        assert resources.content_address("toy") != previous

    def test_a_byte_changed_inside_the_tree_is_caught(self, forest):
        resources.verify("toy")
        target = forest / "x/sub/two.dat"
        time.sleep(0.01)
        target.write_bytes(TREE["x/sub/two.dat"][:-1] + b"!")
        os.utime(target, ns=(time.time_ns(), time.time_ns() + 10**9))
        with pytest.raises(resources.ResourceMissing, match="do not match the record"):
            resources.verify("toy")

    def test_a_file_added_to_the_tree_is_caught_by_the_count(self, forest):
        (forest / "x/extra.dat").write_bytes(b"?")
        with pytest.raises(resources.ResourceMissing, match="found 4 files"):
            resources.verify("toy")

    def test_a_file_removed_from_the_tree_is_caught(self, forest):
        (forest / "x/one.dat").unlink()
        with pytest.raises(resources.ResourceMissing, match="found 2 files"):
            resources.verify("toy")

    def test_a_missing_tree_is_named(self, forest):
        import shutil

        shutil.rmtree(forest / "x")
        with pytest.raises(resources.ResourceMissing, match="x is not a directory"):
            resources.verify("toy")

    def test_a_link_inside_the_tree_is_refused(self, forest, tmp_path):
        (tmp_path / "outside").write_bytes(b"outside")
        (forest / "x/sub/link.dat").symlink_to(tmp_path / "outside")
        with pytest.raises(resources.ResourceMissing, match="is a link"):
            resources.verify("toy")

    def test_a_file_outside_the_tree_and_the_record_is_still_refused(self, forest):
        (forest / "stray.txt").write_bytes(b"?")
        with pytest.raises(resources.ResourceMissing, match="stray.txt"):
            resources.verify("toy")

    def test_a_second_check_does_not_hash_the_tree_again(self, forest, monkeypatch):
        resources.verify("toy")

        def refuse(path):
            raise AssertionError("hashed a file already verified")

        monkeypatch.setattr(resources, "_hash", refuse)
        resources.verify("toy")

    def test_a_tree_cannot_be_downloaded_by_provisioning(self, forest):
        with pytest.raises(resources.ResourceMissing, match="built locally"):
            resources.provision("toy")


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, url: str):
        super().__init__(data)
        self.url = url

    def geturl(self) -> str:
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestProvision:
    @pytest.fixture
    def web(self, home, monkeypatch):
        monkeypatch.setattr(resources, "spec", lambda name: record(url=True))
        served = {"https://example.org/model/a.bin": A, "https://example.org/b.txt": B}
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda request, timeout=0: FakeResponse(
                served[request.full_url], request.full_url
            ),
        )
        return served

    def test_a_download_is_checked_and_moved_into_place(self, home, web):
        root = resources.provision("toy")
        assert (root / "model/a.bin").read_bytes() == A
        assert (root / "resource-lock.json").is_file()
        assert not list(root.parent.glob(".staging-*"))

    def test_an_installed_resource_is_verified_not_downloaded_again(self, home, web):
        resources.provision("toy")
        web.clear()  # any further request would raise KeyError
        assert resources.provision("toy") == resources.location("toy")

    def test_bytes_that_differ_from_the_record_install_nothing(self, home, web):
        web["https://example.org/b.txt"] = b"tampered content!!!!!!!!!!!!!"[: len(B)]
        with pytest.raises(
            resources.ResourceMissing, match="does not match its record"
        ):
            resources.provision("toy")
        assert not resources.location("toy").exists()
        assert not list(resources.location("toy").parent.glob(".staging-*"))

    def test_a_response_longer_than_recorded_is_stopped(self, home, web):
        web["https://example.org/b.txt"] = B + b"more"
        with pytest.raises(resources.ResourceMissing, match="larger than the recorded"):
            resources.provision("toy")

    def test_a_resource_built_locally_cannot_be_downloaded(self, home):
        with pytest.raises(resources.ResourceMissing, match="built locally"):
            resources.provision("toy")


class TestFetchedByTheToolsOwnDownloader:
    """A resource whose record names an engine that fetches it, then verifies the result."""

    @pytest.fixture
    def fetched(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_RESOURCE_HOME", str(tmp_path / "home"))
        monkeypatch.delenv("ALAMBIQ_RESOURCE_TOY", raising=False)
        pinned = {
            **record(),
            "fetch": {"environment": "boltz", "entry": "boltz_download:run"},
        }
        monkeypatch.setattr(resources, "spec", lambda name: pinned)
        return pinned

    def fake_engine(self, monkeypatch, files):
        calls = []

        def run(environment, entry, request, **kwargs):
            calls.append((environment, entry, request["target"]))
            for path, data in files.items():
                destination = Path(request["target"]) / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            return {}

        from alambiq.core import envs

        monkeypatch.setattr(envs, "run_in_environment", run)
        return calls

    def test_the_named_engine_fetches_and_the_result_is_verified(
        self, fetched, monkeypatch
    ):
        calls = self.fake_engine(monkeypatch, {"model/a.bin": A, "b.txt": B})
        root = resources.provision("toy")
        assert root == resources.location("toy")
        assert (root / "model/a.bin").read_bytes() == A
        assert calls[0][:2] == ("boltz", "boltz_download:run")
        assert not [p for p in root.parent.iterdir() if p.name.startswith(".staging")]

    def test_a_second_provision_does_not_fetch_again(self, fetched, monkeypatch):
        calls = self.fake_engine(monkeypatch, {"model/a.bin": A, "b.txt": B})
        resources.provision("toy")
        resources.provision("toy")
        assert len(calls) == 1

    def test_files_that_do_not_match_the_record_are_removed(self, fetched, monkeypatch):
        self.fake_engine(monkeypatch, {"model/a.bin": b"wrong", "b.txt": B})
        with pytest.raises(resources.ResourceMissing, match="a.bin"):
            resources.provision("toy")
        assert not resources.location("toy").exists()

    def test_an_engine_that_fails_leaves_nothing_behind(self, fetched, monkeypatch):
        from alambiq.core import envs

        def refuse(*args, **kwargs):
            raise RuntimeError("no network")

        monkeypatch.setattr(envs, "run_in_environment", refuse)
        with pytest.raises(RuntimeError, match="no network"):
            resources.provision("toy")
        assert not resources.location("toy").exists()
        assert not [
            p for p in resources.location("toy").parent.iterdir() if p.exists()
        ]


def test_the_boltz_downloader_engine_calls_boltzs_own_function(tmp_path, monkeypatch):
    """With a stand-in `boltz.main`, the engine fetches into the target and checks the set."""
    import importlib.util
    import sys
    import types

    def download_boltz2(cache):
        (cache / "mols").mkdir()
        for name in ("mols.tar", "boltz2_conf.ckpt", "boltz2_aff.ckpt"):
            (cache / name).write_bytes(b"x")

    package = types.ModuleType("boltz")
    main = types.ModuleType("boltz.main")
    main.download_boltz2 = download_boltz2
    package.main = main
    monkeypatch.setitem(sys.modules, "boltz", package)
    monkeypatch.setitem(sys.modules, "boltz.main", main)
    source = Path(resources.__file__).resolve().parents[1] / "engines/boltz_download.py"
    spec = importlib.util.spec_from_file_location("boltz_download_under_test", source)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    out = engine.run({"target": str(tmp_path / "cache")})
    assert out["fetched"] == ["boltz2_aff.ckpt", "boltz2_conf.ckpt", "mols", "mols.tar"]

    main.download_boltz2 = lambda cache: (cache / "mols.tar").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="boltz2_conf.ckpt"):
        engine.run({"target": str(tmp_path / "partial")})


class TestReadOnlyStore:
    def test_a_store_that_cannot_hold_a_receipt_still_verifies(self, home, monkeypatch):
        root = install(home)
        monkeypatch.setenv("ALAMBIQ_RESOURCE_RECEIPTS", "/proc/no/such/place")
        assert resources.verify("toy") == root

    def test_receipts_can_be_kept_in_another_directory(
        self, home, tmp_path, monkeypatch
    ):
        install(home)
        elsewhere = tmp_path / "receipts"
        monkeypatch.setenv("ALAMBIQ_RESOURCE_RECEIPTS", str(elsewhere))
        resources.verify("toy")
        assert list(elsewhere.glob("*.json"))
        assert not (home / "verified").exists()
