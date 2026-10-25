"""The `alambiq resource` command, against a two-file resource kept in a temporary home."""

from __future__ import annotations

import hashlib

import pytest

from alambiq import cli
from alambiq.core import resources

A, B = b"alpha bytes", b"beta bytes, a little longer"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(url: bool = False) -> dict:
    files = [
        {"path": "a.bin", "bytes": len(A), "sha256": sha(A)},
        {"path": "b.txt", "bytes": len(B), "sha256": sha(B)},
    ]
    if url:
        for f in files:
            f["url"] = "https://example.org/" + f["path"]
    return {
        "logical_id": "toy",
        "version": "1",
        "license": "CC0-1.0",
        "provider": "Nobody",
        "files": files,
    }


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ALAMBIQ_RESOURCE_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ALAMBIQ_RESOURCE_TOY", raising=False)
    monkeypatch.setattr(
        resources,
        "spec",
        lambda name: (
            record(url=True) if name == "toy" else (_ for _ in ()).throw(KeyError(name))
        ),
    )
    monkeypatch.setattr(resources, "names", lambda: ["toy"])
    return tmp_path / "home"


def install() -> None:
    root = resources.location("toy")
    root.mkdir(parents=True)
    (root / "a.bin").write_bytes(A)
    (root / "b.txt").write_bytes(B)


def test_list_shows_size_licence_and_whether_it_is_installed(home, capsys):
    assert cli.main(["resource", "list"]) == 0
    first = capsys.readouterr().out
    assert "toy" in first and "absent" in first
    install()
    cli.main(["resource", "list"])
    line = capsys.readouterr().out.strip()
    assert "CC0-1.0" in line and "installed" in line and "absent" not in line


def test_show_gives_identity_location_and_how_it_was_obtained(home, capsys):
    assert cli.main(["resource", "show", "toy"]) == 0
    out = capsys.readouterr().out
    assert "license:  CC0-1.0" in out and "absent" in out
    assert resources.identity("toy") in out


def test_verify_passes_for_an_installed_resource_and_fails_naming_the_file(
    home, capsys
):
    install()
    assert cli.main(["resource", "verify", "toy"]) == 0
    assert "verified:" in capsys.readouterr().out
    (resources.location("toy") / "b.txt").write_bytes(B[:-1] + b"?")
    assert cli.main(["resource", "verify", "toy"]) == 1
    assert "b.txt" in capsys.readouterr().err


def test_verify_of_an_absent_resource_says_how_to_get_it(home, capsys):
    assert cli.main(["resource", "verify", "toy"]) == 1
    assert "alambiq resource provision toy" in capsys.readouterr().err


def test_provision_without_confirmation_states_what_it_would_do_and_downloads_nothing(
    home, capsys
):
    assert cli.main(["resource", "provision", "toy"]) == 2
    out = capsys.readouterr().out
    assert "Would download" in out and "example.org" in out and "--yes" in out
    assert not resources.location("toy").exists()


def test_provision_with_confirmation_downloads_and_verifies(home, monkeypatch, capsys):
    served = {"https://example.org/a.bin": A, "https://example.org/b.txt": B}

    class Response:
        def __init__(self, data, url):
            import io

            self.stream, self.url = io.BytesIO(data), url

        def read(self, n=-1):
            return self.stream.read(n)

        def geturl(self):
            return self.url

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=0: Response(served[request.full_url], request.full_url),
    )
    assert cli.main(["resource", "provision", "toy", "--yes"]) == 0
    assert "provisioned:" in capsys.readouterr().out
    assert (resources.location("toy") / "a.bin").read_bytes() == A


def test_an_unknown_resource_is_named(home, capsys):
    assert cli.main(["resource", "show", "nothing"]) == 2
    assert "nothing" in capsys.readouterr().err


def test_an_action_that_needs_a_name_says_so(home, capsys):
    assert cli.main(["resource", "verify"]) == 2
    assert "needs a resource name" in capsys.readouterr().err


def test_a_resource_built_locally_is_not_downloadable(home, monkeypatch, capsys):
    built = record()
    built["build"] = "Built with the tool."
    monkeypatch.setattr(resources, "spec", lambda name: built)
    assert cli.main(["resource", "provision", "toy", "--yes"]) == 1
    assert "built locally" in capsys.readouterr().err


def test_a_fetched_resource_asks_for_confirmation_naming_its_sources(
    home, monkeypatch, capsys
):
    fetched = record()
    fetched["sources"] = ["https://example.org/archive.tar"]
    fetched["fetch"] = {"environment": "boltz", "entry": "boltz_download:run"}
    monkeypatch.setattr(resources, "spec", lambda name: fetched)
    assert cli.main(["resource", "provision", "toy"]) == 2
    out = capsys.readouterr().out
    assert "example.org" in out and "--yes" in out
    assert not resources.location("toy").exists()


def test_provisioning_an_installed_resource_verifies_it_without_asking(home, capsys):
    root = resources.location("toy")
    root.mkdir(parents=True)
    (root / "a.bin").write_bytes(A)
    (root / "b.txt").write_bytes(B)
    assert cli.main(["resource", "provision", "toy"]) == 0
    assert "already installed and verified" in capsys.readouterr().out
