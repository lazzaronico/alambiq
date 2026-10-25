"""What a researcher sees for the mistakes made first.

Each case starts from the kinetics example and breaks one thing the way a first-time
user does, then runs `alambiq` as the command line does. The message has to name what is
wrong (the field, column, unit, file, step or variable), the exit status has to say the run
did not succeed, and no Python traceback may reach the terminal. These tests check that the
information is present; whether it reads well is for the usability test.
"""

from __future__ import annotations

import shutil
from importlib.util import find_spec
from pathlib import Path

import pytest
import yaml

from alambiq import cli

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)

SOURCE = Path(__file__).resolve().parents[2] / "examples/kinetics-from-measurements"


def start(tmp_path: Path):
    """A copy of the kinetics example, returned as (campaign dict, table text)."""
    for name in ("campaign.yaml", "rates.csv"):
        shutil.copy(SOURCE / name, tmp_path / name)
    return (
        yaml.safe_load((tmp_path / "campaign.yaml").read_text()),
        (tmp_path / "rates.csv").read_text(),
    )


def command_line(tmp_path, campaign, table=None, subcommand="run", capsys=None):
    (tmp_path / "campaign.yaml").write_text(yaml.safe_dump(campaign, sort_keys=False))
    if table is not None:
        (tmp_path / "rates.csv").write_text(table)
    argv = [subcommand, str(tmp_path / "campaign.yaml")]
    if subcommand == "run":
        argv += ["--run-dir", str(tmp_path / "run")]
    status = cli.main(argv)
    seen = capsys.readouterr()
    return status, seen.out, seen.err


def step(campaign, step_id):
    return next(s for s in campaign["steps"] if s["id"] == step_id)


def replace_row(table: str, row: int, **fields: str) -> str:
    lines = table.splitlines()
    header = lines[0].split(",")
    cells = lines[row].split(",")
    for name, value in fields.items():
        cells[header.index(name)] = value
    lines[row] = ",".join(cells)
    return "\n".join(lines) + "\n"


def refuses(status, err, *needles):
    assert status != 0
    assert "Traceback" not in err
    for needle in needles:
        assert needle in err, (needle, err)


# ----------------------------------------------------------------- what the table says


def test_a_reading_marked_measured_without_a_value_names_the_reading(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = replace_row(table, 3, value="", status="measured")
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "m2", "measured")


def test_a_value_of_ND_names_the_reading(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = replace_row(table, 4, value="ND")
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "m3", "finite number")


def test_a_required_column_that_is_absent_is_named(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = "\n".join(
        ",".join(c for i, c in enumerate(line.split(",")) if i != 3)
        for line in table.splitlines()
    )
    status, _, err = command_line(tmp_path, campaign, table + "\n", capsys=capsys)
    refuses(status, err, "metric")


def test_a_misnamed_column_is_named_with_the_way_to_map_it(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = table.replace("sample_id", "sample", 1)
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "sample", "sample_id")


def test_a_unit_that_disagrees_with_the_data_says_what_the_data_has(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["params"]["y_unit"] = "mM/min"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "mM/min", "uM/min")


def test_a_substrate_unit_that_disagrees_says_what_the_conditions_have(
    tmp_path, capsys
):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["params"]["x_unit"] = "uM"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "uM", "mM")


def test_a_fit_across_conditions_that_differ_in_an_unmodelled_quantity(
    tmp_path, capsys
):
    campaign, table = start(tmp_path)
    conditions = step(campaign, "intake")["params"]["condition_values"]
    conditions["c1"]["temperature"] = {"value": 30, "unit": "degC"}
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "temperature")


def test_a_reading_marked_failed_runs_and_is_counted(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = replace_row(table, 5, value="", status="failed")
    status, out, err = command_line(tmp_path, campaign, table, capsys=capsys)
    assert status == 0, err
    assert "1 reading failed" in out
    assert "1 readings failed" not in out


def test_the_count_of_failed_readings_agrees_in_number(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = replace_row(table, 5, value="", status="failed")
    table = replace_row(table, 6, value="", status="failed")
    status, out, err = command_line(tmp_path, campaign, table, capsys=capsys)
    assert status == 0, err
    assert "2 readings failed" in out


def test_a_zero_marked_measured_is_noted(tmp_path, capsys):
    campaign, table = start(tmp_path)
    table = replace_row(table, 5, value="0")
    status, out, err = command_line(tmp_path, campaign, table, capsys=capsys)
    assert status == 0, err
    assert "exactly zero" in out
    assert "failed or missing" in out


# ----------------------------------------------------------------- what the request says


def test_a_misspelled_field_names_it_and_the_step(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["params"]["modle"] = "michaelis_menten"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "step fit", "modle", "accepts", "model")


def test_a_missing_required_field_names_it(tmp_path, capsys):
    campaign, table = start(tmp_path)
    del step(campaign, "fit")["params"]["y_unit"]
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "step fit", "y_unit is required")


def test_a_value_of_the_wrong_kind_names_the_field(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["params"]["seed"] = "lucky"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "fit.seed", "whole number")


def test_an_unsupported_choice_lists_the_choices(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["params"]["model"] = "hill"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "fit.model", "michaelis_menten", "linear")


# ----------------------------------------------------------------- what the files say


def test_a_missing_file_is_named_once_and_its_dependents_are_not_repeated_errors(
    tmp_path, capsys
):
    campaign, table = start(tmp_path)
    step(campaign, "intake")["params"]["table"] = "rattes.csv"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "rattes.csv", "intake")
    assert err.count("rattes.csv") == 1
    assert "did not produce a result" not in err


def test_a_binding_to_an_unknown_step_is_named_with_the_steps_there_are(
    tmp_path, capsys
):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["bind"]["measurements"] = "intak.record"
    status, _, err = command_line(
        tmp_path, campaign, table, subcommand="validate", capsys=capsys
    )
    refuses(status, err, "intak", "intake")


def test_a_binding_to_an_output_the_step_lacks_lists_what_it_has(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["bind"]["measurements"] = "intake.recrod"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "intake.recrod", "record")
    assert "measurements" in err


def test_an_unknown_capability_suggests_the_listing(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "fit")["capability"] = "assay.fitt"
    status, _, err = command_line(tmp_path, campaign, table, capsys=capsys)
    refuses(status, err, "assay.fitt", "capability list")


def test_a_cycle_names_the_steps_in_it(tmp_path, capsys):
    campaign, table = start(tmp_path)
    step(campaign, "intake")["bind"] = {"table": "fit.record"}
    status, _, err = command_line(
        tmp_path, campaign, table, subcommand="validate", capsys=capsys
    )
    refuses(status, err, "intake", "fit")


# ----------------------------------------------------------------- what the environment says


def engine_campaign(tmp_path):
    fasta = tmp_path / "q.fasta"
    fasta.write_text(">q\nMKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDG\n")
    panel = tmp_path / "p.fasta"
    panel.write_text(">a\nMKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDG\n")
    return {
        "id": "search",
        "steps": [
            {
                "id": "s",
                "capability": "protein.search",
                "params": {"query": str(fasta), "panel": str(panel)},
            }
        ],
    }


def test_an_engine_variable_that_is_unset_names_the_variable(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv("ALAMBIQ_ENV_MMSEQS", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    status, _, err = command_line(tmp_path, engine_campaign(tmp_path), capsys=capsys)
    refuses(status, err, "ALAMBIQ_ENV_MMSEQS", "install.md")


def test_an_engine_variable_that_points_nowhere_names_the_path(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", "/no/such/python")
    status, _, err = command_line(tmp_path, engine_campaign(tmp_path), capsys=capsys)
    refuses(status, err, "ALAMBIQ_ENV_MMSEQS", "/no/such/python", "not a file")


def test_an_environment_without_the_tool_names_the_tool_and_the_variable(
    tmp_path, capsys, monkeypatch
):
    import sys

    monkeypatch.setenv("ALAMBIQ_ENV_MMSEQS", sys.executable)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    status, _, err = command_line(tmp_path, engine_campaign(tmp_path), capsys=capsys)
    refuses(status, err, "mmseqs", "ALAMBIQ_ENV_MMSEQS", "install.md")


# ----------------------------------------------------------------- validate before running


def test_validate_finds_the_mistakes_in_a_request_before_anything_runs(tmp_path, capsys):
    campaign, table = start(tmp_path)
    fit = step(campaign, "fit")
    fit["params"]["modle"] = "linear"
    del fit["params"]["y_unit"]
    fit["params"]["seed"] = "lucky"
    status, _, err = command_line(
        tmp_path, campaign, table, subcommand="validate", capsys=capsys
    )
    refuses(status, err, "step fit")
    assert "modle" in err
    assert not (tmp_path / "run").exists()


def test_validate_names_a_missing_required_field(tmp_path, capsys):
    campaign, table = start(tmp_path)
    del step(campaign, "fit")["params"]["y_unit"]
    status, _, err = command_line(
        tmp_path, campaign, table, subcommand="validate", capsys=capsys
    )
    refuses(status, err, "step fit", "y_unit is required")


def test_validate_leaves_to_the_run_what_a_binding_will_supply(tmp_path, capsys):
    campaign, table = start(tmp_path)
    status, out, err = command_line(
        tmp_path, campaign, table, subcommand="validate", capsys=capsys
    )
    assert status == 0, err
    assert "valid" in out


def test_every_example_validates():
    from alambiq.api import validate

    for path in sorted(SOURCE.parent.glob("*/campaign.yaml")):
        assert validate(path) == [], path.parent.name
