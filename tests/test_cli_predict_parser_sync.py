"""`src/cli.py` defines the predict options twice; the two copies must agree.

`_build_predict_parser` exists only to render usage text for `_require_file`'s
error messages, so a flag that drifts there does not break dispatch: it silently
advertises an interface different from the one `main()` actually accepts. The
comment at the head of that block says to keep the two definitions in sync, and
the block has now drifted twice - first losing `--virus` entirely, then shipping
`--conformal-calibrator` without `type=str` and with narrower help text than
`main()`'s copy.

Scope, stated so this file is not read as a wider guarantee than it gives: it
pins the three options the conformal/freeze-mode work owns. `--virus` and
`--per-virus-calibration-dir` still carry help-text drift that predates this
test and is deliberately not asserted here.

`main()`'s `p_predict` is authoritative: it is the parser that parses real
invocations, while `_build_predict_parser` only prints.
"""

from __future__ import annotations

import argparse

import pytest

# Attributes that change what the parser accepts or what it tells the user.
_COMPARED = ("type", "default", "help", "nargs", "const", "choices", "required")

_SYNCED_OPTIONS = ("--conformal", "--conformal-calibrator", "--freeze-mode")


def _capture_main_parser(monkeypatch) -> argparse.ArgumentParser:
    """Build `main()`'s top-level parser without dispatching a subcommand."""
    from src import cli

    captured: dict[str, argparse.ArgumentParser] = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured.setdefault("parser", self)
        return real_parse_args(self, args, namespace)

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    # No subcommand: main() prints help and returns 0, touching nothing else.
    assert cli.main([]) == 0
    assert "parser" in captured, "main() did not call parse_args"
    return captured["parser"]


def _predict_subparser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices["predict"]
    raise AssertionError("no subparsers on the top-level parser")


def _action_for(parser: argparse.ArgumentParser, option: str) -> argparse.Action:
    for action in parser._actions:
        if option in action.option_strings:
            return action
    raise AssertionError(f"{option} is not defined on {parser}")


@pytest.mark.parametrize("option", _SYNCED_OPTIONS)
def test_usage_parser_mirrors_the_real_predict_parser(monkeypatch, option):
    from src import cli

    real = _action_for(_predict_subparser(_capture_main_parser(monkeypatch)), option)
    usage_only = _action_for(cli._build_predict_parser(), option)

    assert type(usage_only) is type(real), f"{option}: action class differs"
    for attr in _COMPARED:
        assert getattr(usage_only, attr) == getattr(real, attr), (
            f"{option}: _build_predict_parser has {attr}={getattr(usage_only, attr)!r} "
            f"but main()'s p_predict has {attr}={getattr(real, attr)!r}"
        )
