"""The Swiss-Prot resource is fetched and built on request, from files checked against the record."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

from alambiq import cli
from alambiq.core import resources
from alambiq.engines import swissprot_build as build

FASTA = ">sp|P00001|KINA_TEST Test kinase\nMAFSAEDVLKEYDRRRRMEALLLSLYYPNDRK\n>sp|P00002|NOEC_TEST Other\nMKKKKKKKKKKKKKKKK\n"
FLAT_FILE = """\
ID   KINA_TEST               Reviewed;          32 AA.
AC   P00001;
DE   RecName: Full=Test kinase;
DE            EC=2.7.11.1;
OS   Homo sapiens (Human).
DR   GO; GO:0004672; F:protein kinase activity; IDA:UniProtKB.
SQ   SEQUENCE   32 AA;  3000 MW;  0000000000000000 CRC64;
     MAFSAEDVLK EYDRRRRMEA LLLSLYYPND RK
//
ID   NOEC_TEST               Reviewed;          17 AA.
AC   P00002;
DE   RecName: Full=Other;
OS   Escherichia coli.
SQ   SEQUENCE   17 AA;  2000 MW;  0000000000000000 CRC64;
     MKKKKKKKKKKKKKKKK
//
"""


def packed(text: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.encode())
    return buffer.getvalue()


class Response:
    def __init__(self, data: bytes, url: str):
        self.stream, self.url = io.BytesIO(data), url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, count: int = -1) -> bytes:
        return self.stream.read(count)

    def geturl(self) -> str:
        return self.url


def serving(files: dict[str, bytes], redirect: str | None = None):
    def opener(request, timeout=0):
        name = request.full_url.rsplit("/", 1)[1]
        return Response(files[name], redirect or request.full_url)

    return opener


def sources(files: dict[str, bytes]) -> list[dict]:
    return [
        {
            "name": name,
            "url": f"https://example.org/{name}",
            "bytes": len(data),
            "md5": hashlib.md5(data).hexdigest(),  # noqa: S324
        }
        for name, data in files.items()
    ]


@pytest.fixture
def files() -> dict[str, bytes]:
    return {
        "uniprot_sprot.fasta.gz": packed(FASTA),
        "uniprot_sprot.dat.gz": packed(FLAT_FILE),
    }


class TestDownload:
    def test_a_file_that_matches_its_record_is_written(self, tmp_path, files):
        name = "uniprot_sprot.fasta.gz"
        source = sources(files)[0]
        build.download(
            source["url"],
            tmp_path / name,
            source["bytes"],
            source["md5"],
            serving(files),
        )
        assert (tmp_path / name).read_bytes() == files[name]

    def test_an_address_that_is_not_https_is_refused(self, tmp_path, files):
        with pytest.raises(RuntimeError, match="not an HTTPS address"):
            build.download("http://example.org/a.gz", tmp_path / "a.gz", 1, "0" * 32)

    def test_a_redirect_to_a_plain_address_is_refused(self, tmp_path, files):
        source = sources(files)[0]
        opener = serving(files, redirect="http://elsewhere.org/x.gz")
        with pytest.raises(
            RuntimeError, match="redirected to an address that is not HTTPS"
        ):
            build.download(
                source["url"], tmp_path / "a.gz", source["bytes"], source["md5"], opener
            )

    def test_a_different_digest_is_refused_and_names_the_release_cause(
        self, tmp_path, files
    ):
        source = sources(files)[0]
        with pytest.raises(
            RuntimeError, match="replaces the files at `current_release`"
        ):
            build.download(
                source["url"],
                tmp_path / "a.gz",
                source["bytes"],
                "0" * 32,
                serving(files),
            )

    def test_a_file_larger_than_recorded_is_refused_before_it_is_stored_whole(
        self, tmp_path, files
    ):
        source = sources(files)[0]
        with pytest.raises(RuntimeError, match="larger than the recorded"):
            build.download(
                source["url"],
                tmp_path / "a.gz",
                source["bytes"] - 1,
                source["md5"],
                serving(files),
            )

    def test_a_file_shorter_than_recorded_is_refused(self, tmp_path, files):
        source = sources(files)[0]
        with pytest.raises(RuntimeError, match="does not match its record"):
            build.download(
                source["url"],
                tmp_path / "a.gz",
                source["bytes"] + 5,
                source["md5"],
                serving(files),
            )


class TestBuild:
    @pytest.fixture(autouse=True)
    def mmseqs(self, monkeypatch):
        binary = shutil.which("mmseqs")
        if binary is None:
            pytest.skip("mmseqs is not available to build a database")

    def test_the_target_holds_the_database_the_index_and_the_table_and_nothing_else(
        self, tmp_path, files
    ):
        target = tmp_path / "staging"
        result = build.run(
            {"target": str(target), "sources": sources(files), "threads": 2},
            opener=serving(files),
        )
        names = {p.name for p in target.iterdir()}
        assert {
            "swissprot-db",
            "swissprot-db.index",
            "swissprot-db.idx",
            "annotations.tsv",
        } <= names
        assert not any(n.endswith(".gz") or n.endswith(".fasta") for n in names)
        assert result["entries"] == 2
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".work-")]

    def test_the_table_carries_the_annotations_of_the_flat_file(self, tmp_path, files):
        target = tmp_path / "staging"
        build.run(
            {"target": str(target), "sources": sources(files), "threads": 2},
            opener=serving(files),
        )
        from alambiq.engines.swissprot_annotations import read

        rows = read(target / "annotations.tsv", {"P00001", "P00002"})
        assert rows["P00001"]["ec"] == ["2.7.11.1"]
        assert rows["P00001"]["go_function"] == ["GO:0004672|protein kinase activity"]

    def test_two_builds_of_the_same_files_are_byte_identical(self, tmp_path, files):
        digests = []
        for number in (1, 2):
            target = tmp_path / f"staging{number}"
            build.run(
                {"target": str(target), "sources": sources(files), "threads": number},
                opener=serving(files),
            )
            digests.append(
                {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(target.iterdir())
                }
            )
        assert digests[0] == digests[1]

    def test_a_download_that_fails_its_check_leaves_nothing_in_the_target(
        self, tmp_path, files
    ):
        broken = {**files, "uniprot_sprot.dat.gz": packed("tampered")}
        listed = sources(files)
        target = tmp_path / "staging"
        with pytest.raises(RuntimeError, match="does not match its record"):
            build.run(
                {"target": str(target), "sources": listed, "threads": 2},
                opener=serving(broken),
            )
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".work-")]
        assert not list(target.iterdir())

    def test_a_record_without_both_source_files_is_refused(self, tmp_path, files):
        with pytest.raises(RuntimeError, match="lists no source file"):
            build.run({"target": str(tmp_path / "t"), "sources": sources(files)[:1]})


class TestProvisioning:
    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALAMBIQ_RESOURCE_HOME", str(tmp_path / "home"))
        monkeypatch.delenv("ALAMBIQ_RESOURCE_TOY", raising=False)
        data = b"table"
        self.pinned = {
            "logical_id": "toy",
            "version": "1",
            "license": "CC-BY-4.0",
            "files": [
                {
                    "path": "t.tsv",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            ],
            "fetch": {"environment": "mmseqs", "entry": "swissprot_build:run"},
            "source_files": [
                {
                    "name": "uniprot_sprot.fasta.gz",
                    "url": "https://ftp.example.org/a.gz",
                    "bytes": 5_000_000,
                    "md5": "0" * 32,
                },
                {
                    "name": "uniprot_sprot.dat.gz",
                    "url": "https://ftp.example.org/b.gz",
                    "bytes": 795_000_000,
                    "md5": "1" * 32,
                },
            ],
        }
        monkeypatch.setattr(resources, "spec", lambda name: self.pinned)
        monkeypatch.setattr(resources, "names", lambda: ["toy"])
        return data

    def test_the_engine_receives_the_source_files_of_the_record(
        self, home, monkeypatch
    ):
        from alambiq.core import envs

        seen = {}

        def run(environment, entry, request, **kwargs):
            seen.update(
                environment=environment, entry=entry, sources=request["sources"]
            )
            (Path(request["target"]) / "t.tsv").write_bytes(home)
            return {}

        monkeypatch.setattr(envs, "run_in_environment", run)
        resources.provision("toy")
        assert (seen["environment"], seen["entry"]) == ("mmseqs", "swissprot_build:run")
        assert [s["name"] for s in seen["sources"]] == [
            "uniprot_sprot.fasta.gz",
            "uniprot_sprot.dat.gz",
        ]

    def test_the_command_states_the_download_and_the_build_and_downloads_nothing(
        self, home, monkeypatch, capsys
    ):
        from alambiq.core import envs

        monkeypatch.setattr(
            envs,
            "run_in_environment",
            lambda *a, **k: pytest.fail("a plan must not run the engine"),
        )
        assert cli.main(["resource", "provision", "toy"]) == 2
        out = capsys.readouterr().out
        assert "Would download 800.0 MB from ftp.example.org and build" in out
        assert "Run again with --yes" in out
        assert not resources.location("toy").exists()


def test_the_swissprot_record_is_fetched_by_the_build_entry():
    record = json.loads(
        (
            Path(sys.modules["alambiq"].__file__).parent
            / "resources"
            / "swissprot-2026_03.json"
        ).read_text()
    )
    assert record["fetch"] == {"environment": "mmseqs", "entry": "swissprot_build:run"}
    assert {s["name"] for s in record["source_files"]} == {
        "uniprot_sprot.fasta.gz",
        "uniprot_sprot.dat.gz",
    }
    assert all(
        s["url"].startswith("https://ftp.uniprot.org/") for s in record["source_files"]
    )
