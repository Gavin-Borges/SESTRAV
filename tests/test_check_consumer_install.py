"""Tests for the release-time consumer-install gate.

Every case but one builds a synthetic source tree under ``tmp_path``, and that
one passes an inline string, so none of these touch the real ``src/`` or
``functions/`` and none of them import a first-party module.
``reachable_modules`` is pure AST work, which is what makes that possible: the
import half of the gate is exercised by the release workflow against a real
installed distribution, not here.
"""

from __future__ import annotations

from scripts.check_consumer_install import (
    declared_subcommands,
    reachable_modules,
    subcommands,
)


def build_tree(root, cli_body: str, modules: dict[str, str] | None = None) -> None:
    """Write a minimal src/ + functions/ tree with the given cli.py body."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "functions").mkdir(parents=True, exist_ok=True)
    (root / "src" / "cli.py").write_text(cli_body, encoding="utf-8")
    for name, source in (modules or {}).items():
        path = root / name.replace(".", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".py").write_text(source, encoding="utf-8")


def test_lazy_import_inside_a_command_function_is_reached(tmp_path):
    """src/cli.py imports its command implementations lazily, so a module-scope
    walk alone would miss every one of them."""
    build_tree(
        tmp_path,
        cli_body=(
            "import argparse\n"
            "def cmd_predict(args):\n"
            "    from src import scorer\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('predict')\n"
        ),
        modules={"src.scorer": "value = 1\n"},
    )
    reached, errors = reachable_modules(tmp_path)
    assert errors == []
    assert "src.scorer" in reached


def test_module_scope_imports_are_followed_transitively(tmp_path):
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_validate(args):\n"
            "    from src import first\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('validate')\n"
        ),
        modules={
            "src.first": "from src import second\n",
            "src.second": "from functions import third\n",
            "functions.third": "value = 3\n",
        },
    )
    reached, errors = reachable_modules(tmp_path)
    assert errors == []
    assert {"src.first", "src.second", "functions.third"} <= reached


def test_module_no_command_reaches_is_excluded(tmp_path):
    """The pulp/optimizer case. A module imported only by tests must not drag
    its third-party dependency into [project].dependencies."""
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_info(args):\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('info')\n"
        ),
        modules={"src.optimizer": "import pulp\n"},
    )
    reached, errors = reachable_modules(tmp_path)
    assert errors == []
    assert "src.optimizer" not in reached


def test_subcommand_with_no_command_function_is_reported(tmp_path):
    """A parser the gate cannot seed from is silent under-coverage, so it is an
    error rather than an intersection quietly taken."""
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_info(args):\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('info')\n"
            "    subparsers.add_parser('benchmark')\n"
        ),
    )
    _reached, errors = reachable_modules(tmp_path)
    assert any("benchmark" in message and "no cmd_benchmark" in message for message in errors)


def test_command_function_no_parser_advertises_is_reported(tmp_path):
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_info(args):\n"
            "    return 0\n"
            "def cmd_ghost(args):\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('info')\n"
        ),
    )
    _reached, errors = reachable_modules(tmp_path)
    assert any("cmd_ghost" in message for message in errors)


def test_nested_function_imports_do_not_leak_into_module_scope(tmp_path):
    """import_nodes skips nested defs, so a helper's private import must not be
    treated as a module-scope edge of the module that defines the helper."""
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_info(args):\n"
            "    from src import entry\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('info')\n"
        ),
        modules={
            "src.entry": "def helper():\n    from src import buried\n    return 1\n",
            "src.buried": "value = 1\n",
        },
    )
    reached, errors = reachable_modules(tmp_path)
    assert errors == []
    assert "src.entry" in reached
    assert "src.buried" not in reached


def test_declared_subcommands_reads_both_quote_styles():
    source = "subparsers.add_parser('predict')\nsubparsers.add_parser(\"validate\")\n"
    assert declared_subcommands(source) == {"predict", "validate"}


def test_missing_cli_module_is_an_error_not_an_empty_pass(tmp_path):
    """An empty result must never read as a clean run."""
    (tmp_path / "src").mkdir()
    reached, errors = reachable_modules(tmp_path)
    assert reached == set()
    assert errors and "src.cli" in errors[0]


def test_subcommands_returns_names_and_flags_mismatch(tmp_path):
    """The release workflow derives its --help loop from this, so a mismatch
    here must be an error rather than a silently shorter list."""
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_info(args):\n"
            "    return 0\n"
            "def cmd_orphan(args):\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('info')\n"
        ),
    )
    names, errors = subcommands(tmp_path)
    assert names == {"info"}
    assert any("cmd_orphan" in message for message in errors)


def test_subcommands_flags_a_parser_with_no_implementation(tmp_path):
    """The other direction. Without this case the declared-minus-implemented
    branch of subcommands() is untested, and a mutation removing it survives -
    which is exactly what happened before this test was added."""
    build_tree(
        tmp_path,
        cli_body=(
            "def cmd_info(args):\n"
            "    return 0\n"
            "def build():\n"
            "    subparsers.add_parser('info')\n"
            "    subparsers.add_parser('benchmark')\n"
        ),
    )
    names, errors = subcommands(tmp_path)
    assert names == {"info", "benchmark"}
    assert any("benchmark" in message and "no cmd_benchmark" in message for message in errors)
