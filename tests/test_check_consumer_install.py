"""Tests for the release-time consumer-install gate.

Every case but one builds a synthetic source tree under ``tmp_path``, and that
one passes an inline string, so none of these touch the real ``src/`` or
``functions/`` and none of them import a first-party module.
``reachable_modules`` is pure AST work, which is what makes that possible.
The ``main()`` cases replace the gate's ``importlib`` with a recorder, so they
prove WHICH modules each mode asks to import (every reachable one for the
default run, none for ``--list`` and ``--list-commands``) without importing
anything real. Whether those imports succeed against a real installed
distribution is exercised by the release workflow, not here.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts.check_consumer_install import (
    declared_subcommands,
    main,
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


# ---------------------------------------------------------------------------
# main(): the modes release.yml runs
# ---------------------------------------------------------------------------

ONE_COMMAND_CLI = (
    "def cmd_predict(args):\n"
    "    from src import scorer\n"
    "    return 0\n"
    "def build():\n"
    "    subparsers.add_parser('predict')\n"
)
REACHABLE = ["src.cli", "src.scorer"]


def record_imports(monkeypatch, origin_root: Path, fail: dict | None = None) -> list[str]:
    """Swap the gate's importlib for a recorder; nothing real is imported.

    Each recorded module reports a ``__file__`` under ``origin_root``, so a
    caller chooses whether the imports look installed or in-tree. A name in
    ``fail`` raises that exception instead.
    """
    calls: list[str] = []

    def import_module(name: str) -> types.ModuleType:
        calls.append(name)
        if fail and name in fail:
            raise fail[name]
        module = types.ModuleType(name)
        module.__file__ = str(origin_root / (name.replace(".", "/") + ".py"))
        return module

    monkeypatch.setattr(
        "scripts.check_consumer_install.importlib",
        types.SimpleNamespace(import_module=import_module),
    )
    return calls


def run_main(monkeypatch, *args: str) -> int:
    monkeypatch.setattr(sys, "argv", ["check_consumer_install.py", *args])
    return main()


def test_main_release_import_step_imports_every_reachable_module(tmp_path, monkeypatch, capsys):
    """release.yml's import step passes --root and --require-installed and no
    list flag. If either list branch fired here, the step would print a list
    and pass having imported nothing, so the recorded calls are the point."""
    repo = tmp_path / "repo"
    build_tree(repo, ONE_COMMAND_CLI, modules={"src.scorer": "value = 1\n"})
    calls = record_imports(monkeypatch, tmp_path / "site-packages")

    rc = run_main(monkeypatch, "--root", str(repo), "--require-installed")
    out = capsys.readouterr().out

    assert sorted(calls) == REACHABLE
    assert rc == 0
    assert "reachable first-party modules: 2" in out
    assert "import failures: 0" in out
    assert "IN-TREE" not in out
    assert "OK every module reachable from a shipped command imports" in out


def test_main_fails_when_a_reachable_module_does_not_import(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    build_tree(repo, ONE_COMMAND_CLI, modules={"src.scorer": "value = 1\n"})
    missing = ModuleNotFoundError("No module named 'pulp'", name="pulp")
    calls = record_imports(monkeypatch, tmp_path / "site-packages", fail={"src.scorer": missing})

    rc = run_main(monkeypatch, "--root", str(repo), "--require-installed")
    out = capsys.readouterr().out

    assert sorted(calls) == REACHABLE
    assert rc == 1
    assert "import failures: 1" in out
    assert "FAIL module=src.scorer error=ModuleNotFoundError missing=pulp" in out
    assert "OK every module" not in out


def test_main_require_installed_refuses_imports_from_the_tree(tmp_path, monkeypatch, capsys):
    """The vacuous case --require-installed exists to catch: every module
    resolved inside --root, so the run certified the checkout, not an install."""
    repo = tmp_path / "repo"
    build_tree(repo, ONE_COMMAND_CLI, modules={"src.scorer": "value = 1\n"})
    calls = record_imports(monkeypatch, repo.resolve())

    rc = run_main(monkeypatch, "--root", str(repo), "--require-installed")
    out = capsys.readouterr().out

    assert sorted(calls) == REACHABLE
    assert rc == 1
    assert "IN-TREE 2 module(s) imported from" in out
    assert "OK every module" not in out


def test_main_list_prints_the_modules_without_importing(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    build_tree(repo, ONE_COMMAND_CLI, modules={"src.scorer": "value = 1\n"})
    calls = record_imports(monkeypatch, tmp_path / "site-packages")

    rc = run_main(monkeypatch, "--root", str(repo), "--list")
    out = capsys.readouterr().out

    assert calls == []
    assert rc == 0
    assert out.splitlines() == REACHABLE


def test_main_list_commands_prints_the_names_without_importing(tmp_path, monkeypatch, capsys):
    """release.yml's --help loop is driven by exactly this output."""
    repo = tmp_path / "repo"
    build_tree(repo, ONE_COMMAND_CLI, modules={"src.scorer": "value = 1\n"})
    calls = record_imports(monkeypatch, tmp_path / "site-packages")

    rc = run_main(monkeypatch, "--root", str(repo), "--list-commands")
    out = capsys.readouterr().out

    assert calls == []
    assert rc == 0
    assert out.splitlines() == ["predict"]
