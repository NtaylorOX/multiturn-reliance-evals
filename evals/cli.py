"""One `reliance` command in front of the repo's tools.

    uv run reliance run --models llama31-8b --n 1 --log-dir logs/my-run
    uv run reliance monitor logs/my-run
    uv run reliance plots logs/my-run --out report

Each subcommand is the existing module's own command line, so `reliance <name> --help` shows that tool's options and
`python -m <module>` keeps working.
"""
from __future__ import annotations

import importlib
import sys

# subcommand -> (module with a `main()`, one-line summary)
COMMANDS: dict[str, tuple[str, str]] = {
    "run": ("evals.run", "run the models x personas x pressures grid"),
    "progress": ("evals.progress", "terminal progress view for a running grid"),
    "rejudge": ("evals.rejudge", "re-score existing logs with another rubric or judge"),
    "plots": ("observability.plots", "plots, CSVs and a static HTML report from logs"),
    "monitor": ("monitor.server", "live monitor with Run, Label and Tune tabs"),
    "sample": ("validation.sample", "draw a blind sample of turns to label"),
    "kappa": ("validation.kappa", "agreement between the judge and a labeller"),
    "tune": ("validation.tune", "tune the judge rubric against labelled turns"),
}


def usage() -> str:
    """The help text: one line per subcommand."""
    width = max(map(len, COMMANDS))
    lines = [f"  {name:<{width}}  {summary}" for name, (_, summary) in COMMANDS.items()]
    return "usage: reliance <command> [options]\n\ncommands:\n" + "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the chosen tool. Returns a process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(usage())
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        sys.stderr.write(f"reliance: unknown command {name!r}\n\n{usage()}")
        return 2
    module = importlib.import_module(COMMANDS[name][0])
    sys.argv = [f"reliance {name}", *rest]   # the tool's argparse reads sys.argv
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
