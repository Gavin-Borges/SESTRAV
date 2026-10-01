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
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYTHON_PLATFORM = "linux"

# pip-compile's "unsafe" set. `uv pip compile` emits these by default, which is
# equivalent to pip-compile --allow-unsafe; specs compiled without that flag
# must opt out explicitly or the recompile adds packages that were never there.
UNSAFE_PACKAGES = ("pip", "setuptools", "wheel")

# The application lockfiles, as opposed to the CI tool environments: the runtime
# closure, the production lock, and the API and demo images' closures. --ci-env
# selects among the tool environments only.
RUNTIME_SPEC_NAMES = ("runtime", "lock", "api", "demo")

# Image lockfiles compiled with another lockfile's pins as uv's version
# preferences, so the transitive dependencies they share with it keep the
# versions CI tests. uv prefers whatever its --output-file already pins, so each
# of these is compiled in a scratch copy whose output starts as a preference
# list: the seed's version for every package the seed pins, the previous lock's
# version for the rest (so an unrelated recompile does not churn them). See
# _compile_seeded. Their specs must be self-contained (no -r or -c lines).
SEEDED_SPECS = {"api": "requirements.txt", "demo": "requirements.txt"}

_PIN_LINE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s\\;]+)")


@dataclass(frozen=True)
class LockSpec:
    """One source `.in` file and the hash-pinned artifact compiled from it."""

    name: str
    source: str
    output: str
    python_version: str
    allow_unsafe: bool
    # uv override file this spec needs to reach a patched release, or None.
    # An override makes the lock disagree with a package's own metadata, so a
    # resolving `pip install` of it fails and every install must pass --no-deps.
    # See environments/semgrep-overrides.txt for the one current case.
    overrides: str | None = None


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
        "api",
        "environments/requirements-api.in",
        "environments/requirements-api.txt",
        "3.13",
        True,
    ),
    LockSpec(
        "demo",
        "environments/requirements-demo.in",
        "environments/requirements-demo.txt",
        "3.13",
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
        "3.13",
        False,
    ),
    LockSpec(
        "ci-mypy",
        "environments/requirements-ci-mypy.in",
        "environments/requirements-ci-mypy.txt",
        "3.13",
        False,
    ),
    LockSpec(
        "ci-pytest-cov",
        "environments/requirements-ci-pytest-cov.in",
        "environments/requirements-ci-pytest-cov.txt",
        "3.13",
        False,
    ),
    LockSpec(
        "ci-ruff",
        "environments/requirements-ci-ruff.in",
        "environments/requirements-ci-ruff.txt",
        "3.13",
        False,
    ),
    LockSpec(
        "pip-audit",
        "environments/requirements-pip-audit.in",
        "environments/requirements-pip-audit.txt",
        "3.11",
        False,
    ),
    LockSpec(
        "security",
        "environments/requirements-security.in",
        "environments/requirements-security.txt",
        "3.11",
        False,
    ),
    LockSpec(
        "semgrep",
        "environments/requirements-semgrep.in",
        "environments/requirements-semgrep.txt",
        "3.11",
        False,
        overrides="environments/semgrep-overrides.txt",
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
    uv pip compile invocations recorded in each lockfile header.
  * --no-emit-index-url matches the repo convention and keeps a local
    [tool.uv.pip] emit-index-url setting from leaking an index into the file.
  * Files without a `.in` source (requirements-ci-render.txt,
    requirements-ci-torch-cpu.txt, requirements-sbom.txt,
    requirements-pip-bootstrap.txt) are curated by hand
    with `pip download` + `pip hash` and are deliberately not managed here.

The 'runtime' and 'lock' specs constrain setuptools to 84.0.0. The advisory,
GHSA-h35f-9h28-mq5c, is patched at 83.0.0, so both specs sit one release above
the advisory minimum. That floor used to collide with torch 2.12.0's declared
`setuptools<82` build-metadata cap, making both specs unsatisfiable for any
resolver, so each compiled through a `--overrides overrides.txt` file. torch
2.13.0 raised the cap to `setuptools>=77.0.3`, so the override was retired and
both specs now compile with no special handling.

The 'semgrep' spec compiles with `--overrides environments/semgrep-overrides.txt`.
semgrep 1.178.0 (pinned, the latest on 2026-09-30) declares
`pyjwt[crypto]~=2.13.0`, and the override lifts pyjwt to a patched release. The
resulting lock no longer satisfies semgrep's own metadata, so it must be
installed with --no-deps; that file records the measurement behind it and its
exit condition.
"""


def ci_env_choices() -> list[str]:
    """Return the --ci-env names: every spec except the application lockfiles.

    Specs named `ci-<name>` are offered as `<name>`, the shorthand this flag has
    always used. The remaining tool environments (ci, pip-audit, security,
    semgrep) are not `ci-` prefixed and used to have no individual selector at
    all, leaving 4 of the 8 reachable only via --all.
    """
    return [
        spec.name[len(CI_ENV_PREFIX) :] if spec.name.startswith(CI_ENV_PREFIX) else spec.name
        for spec in LOCK_SPECS
        if spec.name not in RUNTIME_SPEC_NAMES
    ]


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
    if spec.overrides:
        command += ["--overrides", spec.overrides]
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
    # `--ci-env mypy` means spec `ci-mypy`; `--ci-env semgrep` means spec
    # `semgrep`, which carries no prefix. Accept both spellings.
    wanted = {CI_ENV_PREFIX + ci_env, ci_env}
    return [spec for spec in LOCK_SPECS if spec.name in wanted and spec.name not in RUNTIME_SPEC_NAMES]


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


def _pins(path: Path) -> dict[str, str]:
    """Map each `name==version` line's canonical name to its version."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _PIN_LINE.match(line)
        if match:
            pins[re.sub(r"[-_.]+", "-", match.group(1)).lower()] = match.group(2)
    return pins


def preference_text(
    spec: LockSpec,
    root: Path = REPO_ROOT,
    *,
    keep_previous: bool = True,
    drop: str | None = None,
) -> str:
    """uv's version preferences for a seeded spec, as `name==version` lines.

    The seed's version for every package the seed pins; for the rest, the
    spec's current lock's version, unless keep_previous is False (a full
    re-lock) or the package is `drop` (the one being upgraded). Pure: reads
    files, writes nothing.
    """
    output = root / spec.output
    preferences = _pins(output) if keep_previous and output.is_file() else {}
    if drop is not None:
        preferences.pop(re.sub(r"[-_.]+", "-", drop).lower(), None)
    preferences.update(_pins(root / SEEDED_SPECS[spec.name]))
    return "".join(f"{name}=={version}\n" for name, version in sorted(preferences.items()))


def _without_upgrade_flags(command: list[str]) -> tuple[list[str], bool, str | None]:
    """Split --upgrade / --upgrade-package X out of an argv."""
    kept: list[str] = []
    upgrade_all, upgrade_package = False, None
    arguments = iter(command)
    for argument in arguments:
        if argument == "--upgrade":
            upgrade_all = True
        elif argument == "--upgrade-package":
            upgrade_package = next(arguments)
        else:
            kept.append(argument)
    return kept, upgrade_all, upgrade_package


def _compile_seeded(spec: LockSpec, command: list[str], root: Path = REPO_ROOT) -> int:
    """Compile a seeded spec in a scratch copy and install the result only on success.

    uv reads its preferences from the --output-file, so the preference list has
    to occupy that path while uv runs. Doing that in the working tree would leave
    an unhashed list behind whenever the compile fails or never runs (a test that
    stubs subprocess, say). The scratch directory holds the spec and the output
    at the same relative paths, so the argv, and the header uv writes from it,
    are unchanged; the real output is replaced only by a lock uv actually wrote,
    byte for byte.

    uv ignores every preference under --upgrade and ignores the named package's
    under --upgrade-package, which would let a re-lock float the image locks off
    the seed. So neither flag reaches uv here; each is applied to the preference
    list instead (a full re-lock keeps only the seed's pins, a single-package
    bump drops that package's previous pin). uv never records either flag in the
    lock header.
    """
    command, upgrade_all, upgrade_package = _without_upgrade_flags(command)
    preferences = preference_text(spec, root, keep_previous=not upgrade_all, drop=upgrade_package)
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir)
        source = scratch / spec.source
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / spec.source, source)
        output = scratch / spec.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(preferences.encode("utf-8"))
        result = subprocess.run(command, cwd=scratch, check=False)
        compiled = output.read_bytes()
        if result.returncode == 0 and b"--hash=" in compiled:
            (root / spec.output).write_bytes(compiled)
        return result.returncode


def _display(spec: LockSpec, command: list[str]) -> str:
    """The argv uv actually receives, so a printed line can be pasted and rerun."""
    if spec.name not in SEEDED_SPECS:
        return " ".join(command)
    kept, upgrade_all, upgrade_package = _without_upgrade_flags(command)
    note = f"  # seeded from {SEEDED_SPECS[spec.name]}"
    if upgrade_all or upgrade_package:
        note += "; the upgrade is applied to the preference list, not passed to uv"
    return " ".join(kept) + note


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
        for spec, command in zip(specs, commands):
            print(_display(spec, command))
        return 0

    version = uv_version()
    if version is None:
        print(UV_MISSING_MESSAGE, file=sys.stderr)
        return 1
    print(f"Using {version}")

    for spec, command in zip(specs, commands):
        print(f"[{spec.name}] {_display(spec, command)}")
        if spec.name in SEEDED_SPECS:
            returncode = _compile_seeded(spec, command)
        else:
            returncode = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        if returncode != 0:
            print(f"ERROR: uv pip compile failed for {spec.source}", file=sys.stderr)
            return returncode
    print(f"Recompiled {len(specs)} lockfile(s). Review the diff before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
