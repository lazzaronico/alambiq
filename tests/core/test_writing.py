"""The repository's text follows the rules in the "Writing" section of AGENTS.md.

The checks are mechanical and run on the Markdown files outside `tests/` and `src/`, with code
blocks, inline code, link targets and URLs removed first:

* headings are not questions and do not begin with a question word; outside the procedural
  guides they do not begin with an imperative verb;
* no contractions, no second or first person ("you", "we"), no exclamation marks;
* none of a short list of evaluative, vague or colloquial words;
* no spelling that the American convention replaces;
* no "workbench" for the software, and no release labels;
* no sentence aside set off with an em dash;
* none of the disclaimer phrases the "Scope" rule of AGENTS.md excludes ("does not establish", "makes no claim", "not a substitute").

The same rules apply to the fixed text that Alambiq writes at run time (report paragraphs,
labels, notes and error messages): `test_the_run_time_text_follows_the_writing_rules` reads
every string constant of 12 or more characters with a space in it under `src/alambiq`.

Everything else in the rules (definitions, precision, vocabulary) is judged in review.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROCEDURAL = ("docs/getting-started.md", "docs/install.md", "environments/")
IMPERATIVES = (
    "Look|Read|Run|Use|Keep|Check|Choose|Write|Make|Find|Build|Fetch|Provision|Open|"
    "Install|Create|Start|Stop|Add|Set|Select|Get|Try|See|Note"
)
# Headings whose first word is a noun here, not an imperative verb.
NOUN_HEADINGS = {"Run records", "Run directories", "Run history"}
QUESTION_WORDS = r"(What|How|Why|When|Where|Who|Which|Is|Are|Does|Do|Can|Should)\b"

CONTRACTION = re.compile(
    r"\b\w+n't\b|\b(?:it|that|there|here|let|what|who)'s\b|\b\w+'(?:re|ve|ll|d)\b",
    re.IGNORECASE,
)
PERSON = re.compile(r"\b(?:you|your|yours|yourself|we|our|ours|us)\b", re.IGNORECASE)
BANNED = re.compile(
    r"\b(?:honest|honestly|simply|just|easily|powerful|robust|seamless|seamlessly|carefully|"
    r"obviously|clearly|of course|silently|crucial|basically|essentially|very|quite|really|"
    r"truly|a bit|a few|a lot|a couple|usually|typically|generally|roughly|somewhat|nice|great)\b",
    re.IGNORECASE,
)
BRITISH = re.compile(
    r"\b(?:modelled|modelling|labelled|labelling|favour\w*|behaviour\w*|colour\w*|centre\w*|"
    r"licence\w*|catalogue\w*|normalis\w*|minimis\w*|organis(?:e|ed|es|ing|ation|ations)\b|analyse\b|analysed|analysing|initialis\w*|"
    r"optimis\w*|summaris\w*|recognis\w*|characteris(?!tic)\w*|utilis\w*|visualis\w*|"
    r"serialis\w*|neutralis\w*|standardis\w*|generalis\w*|randomis\w*|stabilis\w*|"
    r"parametris\w*|whilst|amongst|grey)\b",
    re.IGNORECASE,
)
DISCLAIMER = re.compile(
    r"\b(?:does|do|did) not (?:establish|prove|demonstrate|guarantee|imply|claim)\b|"
    r"\bmakes? no claims?\b|\bnot a substitute\b|\bnot legal advice\b|"
    r"\bnot (?:cleared|established|evidence of)\b",
    re.IGNORECASE,
)
VOCABULARY = re.compile(r"\b(?:workbench|R1)\b")
ASIDE = re.compile(r"\s—\s|\w—\w")


def markdown_files() -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    skipped = ("tests/", "src/")
    return [f for f in listed if not f.startswith(skipped) and (ROOT / f).is_file()]


def prose(text: str) -> list[tuple[int, str]]:
    """The lines of a Markdown file as prose: code, links' targets and URLs removed."""
    lines, in_code = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        line = re.sub(r"!\[", "[", line)  # Markdown image syntax
        line = re.sub(r"`[^`]*`", " ", line)
        line = re.sub(r'"[^"]{1,40}"', " ", line)  # a quoted word or phrase is a mention
        line = re.sub(r"\]\([^)]*\)", "]", line)
        line = re.sub(r"https?://\S+", " ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        lines.append((number, line))
    return lines


def violations() -> list[str]:
    found = []
    for name in markdown_files():
        procedural = name.startswith(PROCEDURAL)
        for number, line in prose((ROOT / name).read_text()):
            where = f"{name}:{number}"
            heading = re.match(r"#{1,6} (.*)", line)
            if heading:
                title = heading.group(1).strip()
                if title.endswith("?") or re.match(QUESTION_WORDS, title):
                    found.append(f"{where}: heading is a question: {title}")
                elif (
                    not procedural
                    and title not in NOUN_HEADINGS
                    and re.match(rf"({IMPERATIVES})\b", title)
                ):
                    found.append(f"{where}: heading begins with a verb: {title}")
            cells = line.strip().strip("|").split("|")
            if not (len(cells) > 1 and all(c.strip() in {"—", ""} for c in cells[1:])):
                for rule, pattern in (
                    ("contraction", CONTRACTION),
                    ("first or second person", PERSON),
                    ("evaluative or vague word", BANNED),
                    ("British spelling", BRITISH),
                    ("vocabulary or release label", VOCABULARY),
                    ("dash aside", ASIDE),
                    ("disclaimer", DISCLAIMER),
                ):
                    for match in pattern.finditer(line):
                        found.append(f"{where}: {rule}: {match.group(0).strip()}")
            if "!" in line:
                found.append(f"{where}: exclamation mark")
    return found


def capability_texts() -> list[tuple[str, str]]:
    """Every docstring a reader of `capability show` or a report meets, by capability."""
    import dataclasses
    import inspect
    import sys

    from alambiq import api
    from alambiq.core import registry

    api._ready()
    texts = []
    for capability in registry.registered():
        module = sys.modules[capability.run.__module__]
        texts.append((f"{capability.id} (module)", module.__doc__ or ""))
        texts.append((f"{capability.id} (function)", capability.run.__doc__ or ""))
        texts.append((f"{capability.id} (summary)", capability.summary))
        for kind, type_ in (
            ("request", capability.request_type),
            ("result", capability.result_type),
        ):
            texts.append((f"{capability.id} ({kind})", inspect.getdoc(type_) or ""))
            for field in dataclasses.fields(type_):
                nested = inspect.getdoc(field.type) if inspect.isclass(field.type) else ""
                if nested and dataclasses.is_dataclass(field.type):
                    texts.append((f"{capability.id} ({kind}.{field.name})", nested))
    return texts


def docstring_violations() -> list[str]:
    found = []
    for where, text in capability_texts():
        for number, line in prose(text):
            for rule, pattern in (
                ("contraction", CONTRACTION),
                ("first or second person", PERSON),
                ("evaluative or vague word", BANNED),
                ("British spelling", BRITISH),
                ("vocabulary or release label", VOCABULARY),
                ("dash aside", ASIDE),
                ("disclaimer", DISCLAIMER),
            ):
                for match in pattern.finditer(line):
                    found.append(f"{where}:{number}: {rule}: {match.group(0).strip()}")
            if "!" in line:
                found.append(f"{where}:{number}: exclamation mark")
    return found


def test_the_capability_documentation_follows_the_writing_rules():
    problems = docstring_violations()
    assert not problems, f"{len(problems)} problems:\n" + "\n".join(problems[:80])


def test_the_text_follows_the_writing_rules():
    problems = violations()
    assert not problems, f"{len(problems)} problems:\n" + "\n".join(problems[:80])


def run_time_strings() -> list[tuple[str, int, str]]:
    """The prose string constants in the package, outside docstrings."""
    import ast

    found = []
    for path in sorted((ROOT / "src" / "alambiq").rglob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and len(node.value) >= 12
                and " " in node.value
            ):
                found.append((str(path.relative_to(ROOT)), node.lineno, node.value))
    return found


def test_the_run_time_text_follows_the_writing_rules():
    found = []
    for name, number, text in run_time_strings():
        text = re.sub(r"<[^>]+>", " ", text)
        for rule, pattern in (
            ("contraction", CONTRACTION),
            ("first or second person", PERSON),
            ("evaluative or vague word", BANNED),
            ("British spelling", BRITISH),
            ("dash aside", ASIDE),
            ("disclaimer", DISCLAIMER),
        ):
            for match in pattern.finditer(text):
                found.append(f"{name}:{number}: {rule}: {match.group(0).strip()}")
        if "!" in text:
            found.append(f"{name}:{number}: exclamation mark")
    assert not found, f"{len(found)} problems:\n" + "\n".join(found[:80])
