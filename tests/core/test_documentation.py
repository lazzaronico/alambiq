"""The documentation agrees with the code and with itself.

* every relative link in a Markdown file points at something that exists;
* every capability is described on a methods page that the methods index lists;
* every file path a shell block names exists, and every `alambiq` subcommand it uses is one
  the command line has;
* every example has a README with the sections the others have, and appears in the index
  of examples.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"```(?:bash|sh)\n(.*?)```", re.DOTALL)


def markdown_files() -> list[Path]:
    skipped = {"build", "dist", ".venv", ".alambiq", "archive"}
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if not skipped & set(p.relative_to(ROOT).parts)
        and ".egg-info" not in str(p)
        and "node_modules" not in str(p)
        and "tests/fixtures" not in p.as_posix()
        and "src/alambiq/report/assets" not in p.as_posix()
    )


def test_relative_links_resolve():
    broken = []
    for path in markdown_files():
        text = FENCE.sub("", path.read_text())
        text = re.sub(r"`[^`\n]*`", "", text)
        for target in LINK.findall(text):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            location = target.split("#")[0]
            if location and not (path.parent / location).exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert broken == []


def test_every_capability_has_a_methods_page_the_index_lists():
    from alambiq import api
    from alambiq.core import registry

    api._ready()
    ids = sorted(c.id for c in registry.registered())
    index = (ROOT / "docs/methods/README.md").read_text()
    pages = {
        name: (ROOT / "docs/methods" / name).read_text()
        for name in re.findall(r"\]\(([\w-]+\.md)\)", index)
    }
    assert pages
    missing = [i for i in ids if not any(f"`{i}`" in text for text in pages.values())]
    assert missing == []


def test_shell_blocks_name_files_that_exist_and_commands_that_do_exist():
    from alambiq import cli

    commands = set(
        cli.build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    )
    problems = []
    for path in markdown_files():
        for block in FENCE.findall(path.read_text()):
            for line in block.splitlines():
                tokens = line.replace("\\", " ").split()
                if "alambiq" in tokens or any(t.endswith("/alambiq") for t in tokens):
                    after = tokens[
                        next(
                            i
                            for i, t in enumerate(tokens)
                            if t == "alambiq" or t.endswith("/alambiq")
                        )
                        + 1 :
                    ]
                    if after and not after[0].startswith("-") and after[0] not in commands:
                        problems.append(f"{path.name}: unknown command in: {line.strip()}")
                for token in tokens:
                    if token.startswith(("examples/", "docs/", "tests/", "tools/")):
                        if not (ROOT / token).exists():
                            problems.append(f"{path.name}: {token} does not exist")
    assert problems == []


def test_examples_are_indexed_and_described_alike():
    index = (ROOT / "examples/README.md").read_text()
    for directory in sorted(p for p in (ROOT / "examples").iterdir() if p.is_dir()):
        assert f"({directory.name}/README.md)" in index, directory.name
        assert (directory / "README.md").is_file()


@pytest.mark.parametrize("name", ["README.md", "docs/getting-started.md"])
def test_the_first_run_command_names_an_example_that_exists(name):
    text = (ROOT / name).read_text()
    paths = re.findall(r"alambiq run (examples/[\w./-]+)", text)
    assert paths
    for path in paths:
        assert (ROOT / path).is_file(), path


def test_the_extras_named_in_the_documentation_and_in_ci_are_defined_in_pyproject():
    import tomllib

    extras = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
    )
    assert "profiles" not in extras
    table = re.findall(
        r"^\| `([a-z-]+)`(?:, `([a-z-]+)`)? \|",
        (ROOT / "docs/install.md").read_text(),
        re.MULTILINE,
    )
    named = {name for pair in table for name in pair if name} - {"Extra"}
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    install = re.search(r'"\$wheel\[([^\]]+)\]"', ci)
    build = re.search(r"ALAMBIQ_EXTRAS=([^']+)'", ci)
    named |= set(install.group(1).split(",")) | set(build.group(1).split())
    assert named - extras == set()
