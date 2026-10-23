import importlib

import pytest

from evals.cli import COMMANDS, main, usage


def test_every_subcommand_points_at_a_module_with_main():
    for module, _ in COMMANDS.values():
        assert callable(importlib.import_module(module).main)


def test_help_lists_every_command(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS) and out == usage()


def test_unknown_command_is_an_error(capsys):
    assert main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_dispatch_passes_the_remaining_arguments(monkeypatch):
    seen = {}
    monkeypatch.setattr("validation.kappa.main", lambda: seen.update(argv=__import__("sys").argv))
    assert main(["kappa", "labels.json", "key.csv"]) == 0
    assert seen["argv"] == ["reliance kappa", "labels.json", "key.csv"]


def test_help_flag_of_a_tool_reaches_that_tool():
    with pytest.raises(SystemExit) as exc:
        main(["kappa", "--help"])
    assert exc.value.code == 0
