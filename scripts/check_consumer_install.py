#!/usr/bin/env python3
"""Verify that a PyPI consumer's install can actually run the shipped CLI.

`[project].dependencies` is what a consumer gets from `pip install sestrav`.
Anything reachable from a shipped command must import under exactly that set,
with no extras and no lock file.  A missing entry is invisible on a developer
machine, where the dev extra and the conda environment supply the package
anyway, and it only surfaces after the release is irreversible.

The check is deliberately scoped to REACHABILITY FROM A SHIPPED COMMAND rather
than to every module in the wheel.  `src/optimizer.py` imports `pulp`, which
sits in the `dev` extra; that module is wired into neither `src/cli.py` nor
`pipeline.smk`, so no command can reach it and `dev` is the correct home.
Importing every shipped module instead would fail here and push `pulp` into
every consumer's install for nothing.

Run it with the CONSUMER's interpreter, from outside the repository, so the
imports resolve to the installed distribution rather than to the working tree:

    cd "$(mktemp -d)"
    /path/to/venv/bin/python /path/to/scripts/check_consumer_install.py \
        --root /path/to/repo --require-installed

`--root` points at the source tree only so the reachability graph can be built
by AST; no module is imported from it.  `--require-installed` fails if the
imports resolved inside `--root`, which is the failure mode that makes this
whole check vacuous.

Stdlib only, so it needs no dependency of its own to check dependencies.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import sys
from pathlib import Path

PACKAGED_ROOTS = ("src", "functions")
CLI_MODULE = "src.cli"
COMMAND_PREFIX = "cmd_"


def module_name(root: Path, path: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def local_modules(root: Path) -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for package in PACKAGED_ROOTS:
        base = root / package
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            modules[module_name(root, path)] = path
    return modules


def import_nodes(statements: list[ast.stmt]) -> list[ast.Import | ast.ImportFrom]:
    """Collect imports executed in this scope, excluding nested functions and classes."""
    found: list[ast.Import | ast.ImportFrom] = []
    for statement in statements:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            found.append(statement)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        else:
            children: list[ast.stmt] = []
            for _field, value in ast.iter_fields(statement):
                if isinstance(value, list):
                    children.extend(item for item in value if isinstance(item, ast.stmt))
                elif isinstance(value, ast.stmt):
                    children.append(value)
            found.extend(import_nodes(children))
    return found


def resolve_imports(
    current: str, node: ast.Import | ast.ImportFrom, modules: dict[str, Path]
) -> set[str]:
    resolved: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name in modules:
                resolved.add(alias.name)
        return resolved

    package = current.split(".")[:-1]
    if node.level:
        keep = len(package) - (node.level - 1)
        prefix = package[:keep]
    else:
        prefix = []
    base_parts = prefix + (node.module.split(".") if node.module else [])
    base = ".".join(base_parts)
    if base in modules:
        resolved.add(base)
    for alias in node.names:
        candidate = ".".join([*base_parts, alias.name])
        if candidate in modules:
            resolved.add(candidate)
    return resolved


def module_scope_edges(modules: dict[str, Path]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        edges[name] = set().union(
            set(), *(resolve_imports(name, node, modules) for node in import_nodes(tree.body))
        )
    return edges


def closure(seeds: set[str], edges: dict[str, set[str]]) -> set[str]:
    reached: set[str] = set()
    pending = list(seeds)
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        reached.add(module)
        pending.extend(edges.get(module, set()) - reached)
    return reached


def declared_subcommands(cli_source: str) -> set[str]:
    """Subcommand names as argparse advertises them."""
    return set(re.findall(r"""add_parser\(\s*["']([A-Za-z0-9_-]+)["']""", cli_source))


def command_functions(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Command implementations, keyed by the subcommand name they serve."""
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            COMMAND_PREFIX
        ):
            found[node.name[len(COMMAND_PREFIX) :]] = node
    return found


def subcommands(root: Path) -> tuple[set[str], list[str]]:
    """Subcommand names, with any parser/implementation mismatch as errors."""
    modules = local_modules(root)
    if CLI_MODULE not in modules:
        return set(), [f"{CLI_MODULE} not found under {root}"]
    cli_source = modules[CLI_MODULE].read_text(encoding="utf-8")
    tree = ast.parse(cli_source, filename=str(modules[CLI_MODULE]))
    declared = declared_subcommands(cli_source)
    implemented = set(command_functions(tree))
    errors = [
        f"subcommand {name!r} has no {COMMAND_PREFIX}{name} function to seed from"
        for name in sorted(declared - implemented)
    ] + [
        f"{COMMAND_PREFIX}{name} exists but no parser advertises {name!r}"
        for name in sorted(implemented - declared)
    ]
    return declared, errors


def reachable_modules(root: Path) -> tuple[set[str], list[str]]:
    """Return every first-party module a shipped command can reach, plus any errors."""
    modules = local_modules(root)
    if CLI_MODULE not in modules:
        return set(), [f"{CLI_MODULE} not found under {root}"]

    cli_path = modules[CLI_MODULE]
    cli_source = cli_path.read_text(encoding="utf-8")
    tree = ast.parse(cli_source, filename=str(cli_path))

    declared = declared_subcommands(cli_source)
    implemented = command_functions(tree)
    errors: list[str] = []
    # A subcommand with no cmd_ function, or a cmd_ function no parser exposes,
    # means this gate is silently not covering something. Say so rather than
    # quietly checking the intersection.
    for name in sorted(declared - set(implemented)):
        errors.append(f"subcommand {name!r} has no {COMMAND_PREFIX}{name} function to seed from")
    for name in sorted(set(implemented) - declared):
        errors.append(f"{COMMAND_PREFIX}{name} exists but no parser advertises {name!r}")

    edges = module_scope_edges(modules)
    seeds: set[str] = {CLI_MODULE}
    for function in implemented.values():
        for node in ast.walk(function):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                seeds |= resolve_imports(CLI_MODULE, node, modules)
    return closure(seeds, edges), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="repository source root")
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="fail if imports resolve inside --root instead of an installed distribution",
    )
    parser.add_argument("--list", action="store_true", help="print the module list and exit")
    parser.add_argument(
        "--list-commands",
        action="store_true",
        help="print the shipped subcommand names, one per line, and exit",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"--root is not a directory: {root}", file=sys.stderr)
        return 2

    if args.list_commands:
        names, errors = subcommands(root)
        for error in errors:
            print(f"WIRING {error}", file=sys.stderr)
        for name in sorted(names):
            print(name)
        return 1 if errors else 0

    reachable, errors = reachable_modules(root)
    for error in errors:
        print(f"WIRING {error}")

    if args.list:
        for name in sorted(reachable):
            print(name)
        return 1 if errors else 0

    failures: list[tuple[str, str, str]] = []
    in_tree: list[str] = []
    for name in sorted(reachable):
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any import failure is a finding
            missing = getattr(exc, "name", None) or type(exc).__name__
            failures.append((name, type(exc).__name__, str(missing)))
            continue
        origin = getattr(module, "__file__", None)
        if origin and root in Path(origin).resolve().parents:
            in_tree.append(name)

    print(f"reachable first-party modules: {len(reachable)}")
    print(f"import failures: {len(failures)}")
    for name, kind, missing in failures:
        print(f"FAIL module={name} error={kind} missing={missing}")

    if in_tree:
        print(f"IN-TREE {len(in_tree)} module(s) imported from {root}, not from an install")
        if args.require_installed:
            print(
                "This run certifies the working tree, not a consumer install. "
                "Run it from outside the repository with the consumer interpreter."
            )
            return 1

    if failures or errors:
        return 1
    print("OK every module reachable from a shipped command imports under [project].dependencies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
