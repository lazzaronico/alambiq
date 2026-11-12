"""structure.fetch against a server on the loopback address that stands in for the archives.

No archive is contacted. A small HTTP server serves files of the repository's test fixtures at
the paths of the RCSB download service and of the AlphaFold DB prediction listing, so each
answer is known: the stored file equals the served bytes, its SHA-256 is that of the fixture,
the URL is the one the server was asked for, and the server's request count shows whether a
connection was made at all.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/structures"
CIF = (FIXTURES / "1L2Y.cif").read_bytes()
PDB = b"ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\nEND\n"


class Archive:
    """A loopback server with routes added by the tests, and a log of the paths requested."""

    def __init__(self):
        self.routes: dict[str, tuple[int, bytes, dict[str, str], float]] = {}
        self.requests: list[str] = []
        archive = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                archive.requests.append(self.path)
                status, body, headers, delay = archive.routes.get(
                    self.path, (404, b"missing", {}, 0.0)
                )
                time.sleep(delay)
                self.send_response(status)
                for key, value in headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def serve(self, path, body, status=200, headers=None, delay=0.0):
        self.routes[path] = (status, body, headers or {}, delay)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def archive():
    server = Archive()
    yield server
    server.close()


def fetch(tmp_path: Path, name: str = "run", **params):
    from alambiq.api import run as execute

    return execute(
        {
            "id": "f",
            "steps": [
                {"id": "fetch", "capability": "structure.fetch", "params": params}
            ],
        },
        run_dir=tmp_path / name,
    )


def stored(tmp_path: Path, name: str = "run", **params) -> dict:
    outcome = fetch(tmp_path, name, **params)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome.outputs("fetch")


def refused(tmp_path: Path, name: str = "run", **params) -> str:
    outcome = fetch(tmp_path, name, **params)
    assert not outcome.ok
    return " ".join(s.error for s in outcome.steps.values() if s.error)


class TestRcsb:
    def test_the_stored_file_is_the_served_file_with_its_provenance(
        self, archive, tmp_path
    ):
        archive.serve(
            "/download/1L2Y.cif",
            CIF,
            headers={"Last-Modified": "Wed, 01 Jan 2025 00:00:00 GMT"},
        )
        out = stored(
            tmp_path,
            source="rcsb",
            identifier="1l2y",
            allow_network=True,
            base_url=archive.url + "/download",
        )
        assert Path(out["structure"]).read_bytes() == CIF
        assert out["sha256"] == hashlib.sha256(CIF).hexdigest() and out["bytes"] == len(
            CIF
        )
        assert out["url"] == f"{archive.url}/download/1L2Y.cif"
        assert out["identifier"] == "1L2Y" and out["format"] == "mmcif"
        assert out["last_modified"] == "Wed, 01 Jan 2025 00:00:00 GMT"
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", out["retrieved_utc"])
        assert (
            abs(
                time.mktime(time.strptime(out["retrieved_utc"], "%Y-%m-%dT%H:%M:%SZ"))
                - time.mktime(time.gmtime())
            )
            < 60
        )

    def test_the_pdb_format_reads_the_pdb_path(self, archive, tmp_path):
        archive.serve("/download/1L2Y.pdb", PDB)
        out = stored(
            tmp_path,
            source="rcsb",
            identifier="1L2Y",
            format="pdb",
            allow_network=True,
            base_url=archive.url + "/download",
        )
        assert Path(out["structure"]).read_bytes() == PDB and out["url"].endswith(
            "/1L2Y.pdb"
        )

    def test_a_repeated_step_in_the_same_run_directory_reuses_the_stored_file(
        self, archive, tmp_path
    ):
        archive.serve("/download/1L2Y.cif", CIF)
        params = {
            "source": "rcsb",
            "identifier": "1L2Y",
            "allow_network": True,
            "base_url": archive.url + "/download",
        }
        first = stored(tmp_path, **params)
        second = stored(tmp_path, **params)
        assert len(archive.requests) == 1
        assert (
            second["sha256"] == first["sha256"]
            and second["retrieved_utc"] == first["retrieved_utc"]
        )


class TestPermissionAndRefusals:
    def test_without_allow_network_no_connection_is_made(self, archive, tmp_path):
        archive.serve("/download/1L2Y.cif", CIF)
        message = refused(
            tmp_path,
            source="rcsb",
            identifier="1L2Y",
            base_url=archive.url + "/download",
        )
        assert "allow_network: true" in message and archive.requests == []

    def test_a_missing_entry_names_the_url_and_the_status(self, archive, tmp_path):
        message = refused(
            tmp_path,
            source="rcsb",
            identifier="9ZZZ",
            allow_network=True,
            base_url=archive.url + "/download",
        )
        assert "9ZZZ.cif" in message and "404" in message

    def test_a_page_that_is_not_the_format_is_refused(self, archive, tmp_path):
        archive.serve("/download/1L2Y.cif", b"<html><body>maintenance</body></html>")
        message = refused(
            tmp_path,
            source="rcsb",
            identifier="1L2Y",
            allow_network=True,
            base_url=archive.url + "/download",
        )
        assert "does not hold mmCIF" in message
        archive.serve("/download/1L2Y.pdb", b"<html></html>")
        again = refused(
            tmp_path,
            "pdb",
            source="rcsb",
            identifier="1L2Y",
            format="pdb",
            allow_network=True,
            base_url=archive.url + "/download",
        )
        assert "does not hold PDB atom records" in again

    def test_a_response_above_max_megabytes_is_refused(self, archive, tmp_path):
        archive.serve(
            "/download/1L2Y.cif", b"data_x\n_atom_site.id\n" + b"x" * 1_200_000
        )
        message = refused(
            tmp_path,
            source="rcsb",
            identifier="1L2Y",
            allow_network=True,
            max_megabytes=1,
            base_url=archive.url + "/download",
        )
        assert "max_megabytes" in message

    def test_a_server_slower_than_the_timeout_is_reported_as_a_timeout(
        self, archive, tmp_path
    ):
        archive.serve("/download/1L2Y.cif", CIF, delay=3.0)
        message = refused(
            tmp_path,
            source="rcsb",
            identifier="1L2Y",
            allow_network=True,
            timeout_seconds=1,
            base_url=archive.url + "/download",
        )
        assert "within 1 s" in message

    @pytest.mark.parametrize(
        ("source", "identifier"),
        [
            ("rcsb", "12"),
            ("rcsb", "ABCD"),
            ("rcsb", "1L2Y/../x"),
            ("alphafold", "1L2Y"),
            ("alphafold", "p69905!"),
        ],
    )
    def test_an_identifier_of_the_wrong_form_is_refused(
        self, tmp_path, source, identifier
    ):
        assert "identifier" in refused(
            tmp_path, source=source, identifier=identifier, allow_network=True
        )

    @pytest.mark.parametrize(
        "url", ["http://example.org/x", "ftp://example.org/x", "file:///etc/passwd"]
    )
    def test_a_mirror_must_be_https_or_loopback(self, tmp_path, url):
        assert "base_url" in refused(
            tmp_path, source="rcsb", identifier="1L2Y", allow_network=True, base_url=url
        )

    @pytest.mark.parametrize(
        ("name", "value"), [("timeout_seconds", 0), ("max_megabytes", 2000)]
    )
    def test_a_value_outside_its_range_is_refused_by_name(self, tmp_path, name, value):
        assert name in refused(
            tmp_path,
            source="rcsb",
            identifier="1L2Y",
            allow_network=True,
            **{name: value},
        )


class TestAlphaFold:
    def listing(self, archive, **entry):
        base = {
            "uniprotAccession": "P69905",
            "cifUrl": f"{archive.url}/files/AF-P69905-F1-model_v9.cif",
            "pdbUrl": f"{archive.url}/files/AF-P69905-F1-model_v9.pdb",
        }
        return json.dumps([{**base, **entry}]).encode()

    def test_the_file_named_by_the_listing_is_downloaded_and_its_url_recorded(
        self, archive, tmp_path
    ):
        archive.serve("/api/prediction/P69905", self.listing(archive))
        archive.serve("/files/AF-P69905-F1-model_v9.cif", CIF)
        out = stored(
            tmp_path,
            source="alphafold",
            identifier="p69905",
            allow_network=True,
            base_url=archive.url + "/api/prediction",
        )
        assert out["url"] == f"{archive.url}/files/AF-P69905-F1-model_v9.cif"
        assert out["sha256"] == hashlib.sha256(CIF).hexdigest()
        assert archive.requests == [
            "/api/prediction/P69905",
            "/files/AF-P69905-F1-model_v9.cif",
        ]

    def test_the_pdb_format_follows_the_pdb_url(self, archive, tmp_path):
        archive.serve("/api/prediction/P69905", self.listing(archive))
        archive.serve("/files/AF-P69905-F1-model_v9.pdb", PDB)
        out = stored(
            tmp_path,
            source="alphafold",
            identifier="P69905",
            format="pdb",
            allow_network=True,
            base_url=archive.url + "/api/prediction",
        )
        assert Path(out["structure"]).read_bytes() == PDB

    def test_an_empty_listing_means_no_prediction(self, archive, tmp_path):
        archive.serve("/api/prediction/P69905", b"[]")
        message = refused(
            tmp_path,
            source="alphafold",
            identifier="P69905",
            allow_network=True,
            base_url=archive.url + "/api/prediction",
        )
        assert "no prediction for P69905" in message

    def test_a_listing_without_the_file_field_names_the_field(self, archive, tmp_path):
        archive.serve(
            "/api/prediction/P69905",
            json.dumps([{"uniprotAccession": "P69905"}]).encode(),
        )
        message = refused(
            tmp_path,
            source="alphafold",
            identifier="P69905",
            allow_network=True,
            base_url=archive.url + "/api/prediction",
        )
        assert "no cifUrl field" in message

    def test_a_file_url_that_is_not_https_or_loopback_is_refused(
        self, archive, tmp_path
    ):
        archive.serve(
            "/api/prediction/P69905",
            self.listing(archive, cifUrl="http://example.org/x.cif"),
        )
        message = refused(
            tmp_path,
            source="alphafold",
            identifier="P69905",
            allow_network=True,
            base_url=archive.url + "/api/prediction",
        )
        assert "an https URL is required" in message
