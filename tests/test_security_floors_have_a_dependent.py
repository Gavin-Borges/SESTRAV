"""A security floor in a lockfile spec must name a package something else requires.

A ``name>=version`` line in a ``requirements*.in`` spec is a requirement, not a
constraint: ``uv pip compile`` installs whatever it names. A floor added to hold a
transitive dependency at a patched release therefore keeps installing that package
after the dependency that needed it has gone, and nothing notices, because every
lock gate compares versions and none asks why a package is present.

That is how ``environments/requirements.lock`` came to carry pyjwt, cryptography,
msgpack and pydantic-settings (plus cffi, pycparser and python-dotenv beneath
them) into the production image with nothing requiring any of them. Each lock
entry was annotated ``# via -r environments/requirements-lock.in`` and nothing
else. Then pyjwt 2.14.0, the version its own floor selected, was itself the
subject of GHSA-42vr-xj54-vc7v.

The check reads the lock's ``# via`` annotations. A floored package must be
required by at least one thing other than the spec that floors it: another
package, or an ``-r`` include. An ``--override`` or ``-c`` entry does not count,
because it constrains a package without requiring it. uv run from the repo root
writes the spec's self-reference relative to the working directory
(``-r environments/requirements-lock.in``); a lock last compiled by another tool
can carry it relative to the spec's own directory (``-r requirements-semgrep.in``,
as the semgrep lock did). Both forms are handled.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path

import pytest

from tools.check_lockfile_freshness import LOCKFILE_PAIRS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_FLOOR_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*>=")
_PIN_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _floors(spec_text: str) -> list[str]:
    """Canonical names floored by the spec's own lines, not by its -r includes."""
    names = []
    for line in spec_text.splitlines():
        match = _FLOOR_RE.match(line.strip())
        if match:
            names.append(_canonical(match.group(1)))
    return names


def _via(lock_text: str) -> dict[str, list[str]]:
    """Map each pinned package to the entries of its `# via` annotation."""
    via: dict[str, list[str]] = {}
    current: str | None = None
    for raw in lock_text.splitlines():
        pin = _PIN_RE.match(raw)
        if pin:
            current = _canonical(pin.group(1))
            via.setdefault(current, [])
            continue
        if current is None:
            continue
        stripped = raw.strip()
        if stripped.startswith("# via"):
            rest = stripped[len("# via") :].strip()
            if rest:
                via[current].append(rest)
        elif stripped.startswith("#   "):
            via[current].append(stripped[1:].strip())
    return via


def _is_self_reference(entry: str, spec: str) -> bool:
    if not entry.startswith("-r "):
        return False
    target = entry[len("-r ") :].strip()
    spec_norm = os.path.normpath(spec)
    return spec_norm in {
        os.path.normpath(target),
        os.path.normpath(os.path.join(os.path.dirname(spec), target)),
    }


def _requires(entry: str, spec: str) -> bool:
    """Whether a `# via` entry is something that requires the package."""
    if entry.startswith(("--override", "--constraint", "-c ")):
        return False
    return not _is_self_reference(entry, spec)


def orphaned_floors(spec: str, spec_text: str, lock_text: str) -> list[str]:
    """Return a problem line for every floor whose package only the spec requires."""
    via = _via(lock_text)
    problems = []
    for name in _floors(spec_text):
        dependents = [e for e in via.get(name, []) if _requires(e, spec)]
        if name not in via:
            problems.append(f"{spec}: floor {name!r} is absent from the lock")
        elif not dependents:
            problems.append(
                f"{spec}: floor {name!r} is required by nothing but the spec itself, "
                "so the floor is what installs it"
            )
    return problems


@pytest.mark.parametrize(("spec", "lock"), LOCKFILE_PAIRS)
def test_every_security_floor_has_a_dependent(spec: str, lock: str) -> None:
    spec_path = PROJECT_ROOT / spec
    lock_path = PROJECT_ROOT / lock
    assert spec_path.is_file(), f"{spec} is listed in LOCKFILE_PAIRS but missing"
    assert lock_path.is_file(), f"{lock} is listed in LOCKFILE_PAIRS but missing"
    problems = orphaned_floors(
        spec,
        spec_path.read_text(encoding="utf-8"),
        lock_path.read_text(encoding="utf-8"),
    )
    assert not problems, "\n".join(problems)


def test_the_live_specs_carry_floors_to_check() -> None:
    """Guard against the parametrized test passing because it checked nothing."""
    total = sum(
        len(_floors((PROJECT_ROOT / spec).read_text(encoding="utf-8")))
        for spec, _ in LOCKFILE_PAIRS
    )
    assert total >= 1, "no floor lines found in any lockfile spec; the parser has drifted"


_SPEC = "environments/requirements-lock.in"

_LOCK = textwrap.dedent(
    """\
    aiohttp==3.14.3 \\
        --hash=sha256:aa
        # via
        #   -r environments/requirements-lock.in
        #   snakemake
    pyjwt==2.14.0 \\
        --hash=sha256:bb
        # via -r environments/requirements-lock.in
    snakemake==9.27.0 \\
        --hash=sha256:cc
        # via -r environments/../requirements.in
    """
)


def test_a_floor_with_another_dependent_passes() -> None:
    assert orphaned_floors(_SPEC, "aiohttp>=3.14.3\n", _LOCK) == []


def test_a_floor_that_only_the_spec_requires_is_reported() -> None:
    problems = orphaned_floors(_SPEC, "aiohttp>=3.14.3\npyjwt>=2.14.0\n", _LOCK)
    assert len(problems) == 1
    assert "'pyjwt'" in problems[0]


def test_a_floor_with_extras_is_matched_by_its_base_name() -> None:
    problems = orphaned_floors(_SPEC, "PyJWT[crypto]>=2.14.0\n", _LOCK)
    assert len(problems) == 1
    assert "'pyjwt'" in problems[0]


def test_a_self_reference_relative_to_the_spec_directory_is_recognised() -> None:
    lock = textwrap.dedent(
        """\
        cryptography==50.0.1 \\
            --hash=sha256:dd
            # via -r requirements-semgrep.in
        """
    )
    problems = orphaned_floors(
        "environments/requirements-semgrep.in", "cryptography>=50.0.1\n", lock
    )
    assert len(problems) == 1


def test_an_include_of_another_spec_counts_as_a_dependent() -> None:
    lock = textwrap.dedent(
        """\
        urllib3==2.8.0 \\
            --hash=sha256:ee
            # via
            #   -r environments/../requirements.in
            #   -r environments/requirements-lock.in
        """
    )
    assert orphaned_floors(_SPEC, "urllib3>=2.8.0\n", lock) == []


def test_a_floor_absent_from_the_lock_is_reported() -> None:
    problems = orphaned_floors(_SPEC, "msgpack>=1.2.2\n", _LOCK)
    assert problems == [f"{_SPEC}: floor 'msgpack' is absent from the lock"]


def test_an_override_or_constraint_entry_is_not_a_dependent() -> None:
    lock = textwrap.dedent(
        """\
        pyjwt==2.15.1 \\
            --hash=sha256:ff
            # via
            #   --override environments/semgrep-overrides.txt
            #   -c environments/constraints.txt
            #   -r environments/requirements-lock.in
        """
    )
    problems = orphaned_floors(_SPEC, "pyjwt>=2.15.0\n", lock)
    assert len(problems) == 1
    assert "'pyjwt'" in problems[0]
