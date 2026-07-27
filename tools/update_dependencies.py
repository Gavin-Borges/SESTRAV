#!/usr/bin/env python
"""Recompile SESTRAV's hash-pinned lockfiles with `uv pip compile`.

Every lockfile in this repo is installed with `pip install --require-hashes`,
so a recompile must be deterministic and must produce the same resolution the
Ubuntu CI runners will install. This wrapper encodes the per-lockfile
conventions (interpreter version, unsafe-package handling, hash generation)
that are otherwise only recorded in the generated file headers.

Usage:
    python tools/update_dependencies.py --target pillow
    python tools/update_dependencies.py --ci-env mypy
    python tools/update_dependencies.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON_PLATFORM = "linux"

# pip-compile's "unsafe" set. `uv pip compile` emits these by default, which is
# equivalent to pip-compile --allow-unsafe; specs compiled without that flag
# must opt out explicitly or the recompile adds packages that were never there.
UNSAFE_PACKAGES = ("pip", "setuptools", "wheel")


@dataclass(frozen=True)
class LockSpec:
    """One source `.in` file and the hash-pinned artifact compiled from it."""

    name: str
    source: str
    output: str
    python_version: str
    allow_unsafe: bool


LOCK_SPECS: tuple[LockSpec, ...] = (
    LockSpec("runtime", "requirements.in", "requirements.txt", "3.11", True),
    LockSpec(
        "lock",
        "environments/requirements-lock.in",
        "environments/requirements.lock",
        "3.11",
        True,
    ),
    LockSpec(
        "ci",
        "environments/requirements-ci.in",
        "environments/requirements-ci.txt",
        "3.11",
        True,
    ),
    LockSpec(
        "ci-build",
        "environments/requirements-ci-build.in",
        "environments/requirements-ci-build.txt",
        "3.11",
        False,
    ),
    LockSpec(
        "ci-mypy",
        "environments/requirements-ci-mypy.in",
        "environments/requirements-ci-mypy.txt",
        "3.11",
        False,
    ),
    LockSpec(
        "ci-pytest-cov",
        "environments/requirements-ci-pytest-cov.in",
        "environments/requirements-ci-pytest-cov.txt",
        "3.11",
        False,
    ),
    LockSpec(
        "ci-ruff",
        "environments/requirements-ci-ruff.in",
        "environments/requirements-ci-ruff.txt",
        "3.11",
        False,
    ),
    LockSpec(
        "pip-audit",
        "environments/requirements-pip-audit.in",
        "environments/requirements-pip-audit.txt",
        "3.12",
        False,
    ),
    LockSpec(
        "security",
        "environments/requirements-security.in",
        "environments/requirements-security.txt",
        "3.13",
        False,
    ),
    LockSpec(
        "semgrep",
        "environments/requirements-semgrep.in",
        "environments/requirements-semgrep.txt",
        "3.12",
        False,
    ),
)

CI_ENV_PREFIX = "ci-"

UV_MISSING_MESSAGE = (
    "ERROR: `uv` is not installed or is not on PATH.\n"
    "Install it, then re-run this command:\n"
    "    pip install uv\n"
    "This script does not install `uv` for you: the lockfiles it writes are the\n"
    "supply-chain root of trust, so the compiler has to be installed deliberately."
)

EPILOG = """\
Notes:
  * --python-platform defaults to 'linux' so lockfiles resolve for the Ubuntu CI
    runners even when compiled from a Windows or macOS workstation. Without it,
    uv resolves for the host platform and the resulting file installs cleanly
    locally but fails --require-hashes on CI (missing or extra marker-gated
    wheels). Override only if you know why.
  * --generate-hashes and the pinned --python-version reproduce the
    pip-compile invocations recorded in each lockfile header.
  * --no-emit-index-url matches the repo convention and keeps a local
    [tool.uv.pip] emit-index-url setting from leaking an index into the file.
  * Files without a `.in` source (requirements-ci-render.txt,
    requirements-ci-torch-cpu.txt, requirements-sbom.txt) are curated by hand
    with `pip download` + `pip hash` and are deliberately not managed here.

Known blocker: the 'runtime' and 'lock' specs do not resolve today. They floor
setuptools>=83 for GHSA-h35f-9h28-mq5c while torch 2.12.0 declares
setuptools<82, so any resolver returns ResolutionImpossible. Those two files are
currently hand-maintained. Resolve the conflict (see
environments/requirements-lock.in) before relying on --all.
"""


def ci_env_choices() -> list[str]:
    """Return the --ci-env names, i.e. the <name> in requirements-ci-<name>.in."""
    return [spec.name[len(CI_ENV_PREFIX) :] for spec in LOCK_SPECS if spec.name.startswith(CI_ENV_PREFIX)]


def build_command(
    spec: LockSpec,
    *,
    python_platform: str = DEFAULT_PYTHON_PLATFORM,
    upgrade_package: str | None = None,
    upgrade_all: bool = False,
) -> list[str]:
    """Return the argv for the `uv pip compile` run that regenerates `spec`."""
    command = [
        "uv",
        "pip",
        "compile",
        spec.source,
        "--output-file",
        spec.output,
        "--generate-hashes",
        "--no-emit-index-url",
        "--python-version",
        spec.python_version,
        "--python-platform",
        python_platform,
    ]
    if not spec.allow_unsafe:
        for package in UNSAFE_PACKAGES:
            command += ["--unsafe-package", package]
    if upgrade_all:
        command.append("--upgrade")
    elif upgrade_package:
        command += ["--upgrade-package", upgrade_package]
    return command


def uv_version() -> str | None:
    """Return the installed uv version string, or None if uv is unavailable."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def select_specs(ci_env: str | None = None) -> list[LockSpec]:
    """Resolve the CLI selection to the lockfiles that should be recompiled."""
    if ci_env is None:
        return list(LOCK_SPECS)
    wanted = CI_ENV_PREFIX + ci_env
    return [spec for spec in LOCK_SPECS if spec.name == wanted]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="update_dependencies.py",
        description=__doc__,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        metavar="PACKAGE",
        help="bump a single package and nothing else (uv pip compile --upgrade-package)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="upgrade_all",
        help="full re-lock of every managed lockfile (uv pip compile --upgrade)",
    )
    parser.add_argument(
        "--ci-env",
        metavar="NAME",
        choices=ci_env_choices(),
        help="recompile environments/requirements-ci-<NAME>.in only (choices: %(choices)s)",
    )
    parser.add_argument(
        "--python-platform",
        default=DEFAULT_PYTHON_PLATFORM,
        help="resolution target platform (default: %(default)s, matching the CI runners)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the uv commands without running them",
    )
    args = parser.parse_args(argv)

    if args.upgrade_all and (args.target or args.ci_env):
        parser.error("--all cannot be combined with --target or --ci-env")
    if not (args.upgrade_all or args.target or args.ci_env):
        parser.error("choose one of --target, --all or --ci-env")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    specs = select_specs(args.ci_env)

    commands = [
        build_command(
            spec,
            python_platform=args.python_platform,
            upgrade_package=args.target,
            upgrade_all=args.upgrade_all,
        )
        for spec in specs
    ]

    if args.dry_run:
        for command in commands:
            print(" ".join(command))
        return 0

    version = uv_version()
    if version is None:
        print(UV_MISSING_MESSAGE, file=sys.stderr)
        return 1
    print(f"Using {version}")

    for spec, command in zip(specs, commands):
        print(f"[{spec.name}] {' '.join(command)}")
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            print(f"ERROR: uv pip compile failed for {spec.source}", file=sys.stderr)
            return result.returncode
    print(f"Recompiled {len(specs)} lockfile(s). Review the diff before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
