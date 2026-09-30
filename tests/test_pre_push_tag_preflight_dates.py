"""pre-push Check 1b, checked against the Release workflow's own verdicts.

Why this test exists
--------------------
A pushed `v*` tag starts the Release workflow, which creates a public GitHub Release and,
when the `PYPI_PUBLISH` repository variable is true, queues a PyPI upload that can never
be replaced. Nothing on the server gates a tag push. Check 1b re-runs the workflow's
version and date assertions LOCALLY, so that a bad carrier costs a corrected commit
instead of a public tag that fails its own release.

A local copy of those assertions can disagree with the workflow in more than one way.
Three are pinned below:

- A SHAPE check on `date-released`, `^[0-9]{4}-[0-9]{2}-[0-9]{2}$`, would pass
  `2026-02-30`, which the workflow's `datetime.date.fromisoformat` rejects.
- Reading the field differently from the workflow's `scalar()`, which accepts an optional
  quote and stops at a quote, whitespace or `#`. Keeping quotes or trailing comments
  blocks a date the workflow accepts; skipping `[[:space:]]*` after the colon, or
  splitting lines on LF only, passes files the workflow rejects.
- Reading the WORKING TREE. The workflow checks out the commit the tag points at, and
  the working tree at push time can hold something else.

The oracle
----------
The expected verdict for every date below is computed by `datetime.date.fromisoformat`,
the exact call the workflow makes, instead of being written by hand. The cases are all
in `YYYY-MM-DD` shape, the form the hook accepts. So the test fails the moment the hook
and the workflow disagree about any of them.

`test_no_crafted_file_passes_the_hook_that_the_workflow_rejects` extends that to whole
files, against a verbatim copy of the workflow's `scalar()`: non-blank whitespace right
after the colon, and a lone CR used as a line separator. A reader that skipped
`[[:space:]]*` after the colon and split lines on LF only would pass 15 of those cases
under Git for Windows and 14 under Linux. It runs under `LC_ALL=C` and `LC_ALL=C.UTF-8`,
because bash's `[[:space:]]` covers different characters in each: such a reader skips the
Unicode spaces only under the UTF-8 locale, and the no-break space only on Git for Windows.

The carriers are committed in a throwaway repository and the hook is handed that commit,
as git hands it the pushed sha. The tagged-commit cases then change the working tree
after the commit, so a hook that reads the working tree fails them.

Scope: the hook accepts only the `YYYY-MM-DD` form. On Python 3.11+ `fromisoformat` also
accepts other ISO forms, such as `20260617`; the hook blocks those. It is a copy of the
workflow's checks, not the workflow: it also takes pyproject.toml's first
`version = "..."` line instead of parsing TOML.

Anti-vacuity
------------
`test_the_premise_shape_alone_admits_impossible_dates` asserts the premise directly:
every impossible date in the list matches the shape pattern. If that ever stopped
holding, the blocking cases would no longer exercise the calendar check at all.
`test_a_real_date_still_passes`, the extraction cases and
`test_a_bad_working_tree_does_not_block_a_good_tagged_commit` guard the other direction,
so the fix cannot be the degenerate "block every tag".
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-push"

TAG = "v2.0.3"
VERSION = "2.0.3"
SUCCESS_BANNER = "version carriers agree, date-released is not future"
NOT_ISO = "is not an ISO date"
FUTURE = "is in the future"
MISMATCH = "does not match tag"

# A shape-only check of date-released. Every impossible date below matches it.
OLD_SHAPE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")

IMPOSSIBLE_DATES = [
    "2026-02-30",
    "2026-04-31",
    "2026-06-31",
    "2026-09-31",
    "2026-11-31",
    "2026-00-10",
    "2026-13-01",
    "2026-01-00",
    "2026-01-32",
    "2025-02-29",  # not a leap year
    "1900-02-29",  # divisible by 100 and not by 400
    "2100-02-29",
    "0000-01-01",  # datetime.MINYEAR is 1
]

REAL_PAST_DATES = [
    "2026-06-17",
    "2024-02-29",  # leap year
    "2000-02-29",  # divisible by 400
    "2026-01-31",
    "1999-12-31",
    "0001-01-01",
]

# git reads the developer's GLOBAL and SYSTEM config even for a throwaway directory,
# and both the fixture and the hook make git calls. Pinned so the result depends on the
# hook. The repository-discovery variables are dropped so every git call resolves to the
# fixture repository, never to the repository running this test.
BASE_ENV = {
    key: value
    for key, value in os.environ.items()
    if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX"}
}
BASE_ENV.update(
    {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "fixture",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "fixture",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    }
)

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="pre-push is a bash script; without bash it cannot run at all",
)


def _release_workflow_accepts(date_text: str) -> bool:
    """The Release workflow's own verdict: `datetime.date.fromisoformat`."""
    try:
        datetime.date.fromisoformat(date_text)
    except ValueError:
        return False
    return True


def _workflow_scalar(text: str, field: str) -> str | None:
    """release.yml's `scalar()`, regex copied verbatim, over text as `read_text` returns it."""
    m = re.search(rf"^{re.escape(field)}:[ \t]*['\"]?([^'\"\s#]+)", text, re.M)
    return m.group(1) if m else None


def _release_workflow_accepts_file(cff_text: str) -> bool:
    """The workflow's whole CITATION.cff verdict: version, then date-released and not future."""
    text = cff_text.replace("\r\n", "\n").replace("\r", "\n")  # universal newlines
    if _workflow_scalar(text, "version") != VERSION:
        return False
    date_text = _workflow_scalar(text, "date-released")
    if date_text is None or not _release_workflow_accepts(date_text):
        return False
    today = datetime.datetime.now(datetime.timezone.utc).date()
    return datetime.date.fromisoformat(date_text) <= today


def _git(root: Path, *args: str) -> str:
    """Run git in the fixture repository. autocrlf off, so every file is stored byte for byte."""
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env=BASE_ENV,
        timeout=60,
    ).stdout.strip()


def _fixture(
    tmp_path: Path, date_line: str, *, crlf: bool = False, cff_text: str | None = None
) -> Path:
    """A git repository whose one commit holds the two release carriers.

    `cff_text`, when given, is written as the whole CITATION.cff, byte for byte. The hook
    and the python shim sit beside the carriers but are never committed.
    """
    root = tmp_path / "carriers"
    root.mkdir()
    newline = "\r\n" if crlf else "\n"
    (root / "hook").write_text(
        HOOK_SOURCE.read_text(encoding="utf-8"), encoding="utf-8", newline=""
    )
    (root / "pyproject.toml").write_text(
        f'[project]{newline}name = "fixture"{newline}version = "{VERSION}"{newline}',
        encoding="utf-8",
        newline="",
    )
    if cff_text is None:
        cff_text = (
            f"cff-version: 1.2.0{newline}title: fixture{newline}"
            f"version: {VERSION}{newline}{date_line}{newline}"
        )
    (root / "CITATION.cff").write_text(cff_text, encoding="utf-8", newline="")
    # Everything after Check 1b needs a Python with pytest. A `python` that exits 1
    # makes the hook stop at its pytest pre-flight instead of running a real suite,
    # so no case runs the real suite, and none touches this repository.
    shim_dir = root / "shim"
    shim_dir.mkdir()
    shim = shim_dir / "python"
    shim.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8", newline="\n")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    _git(root, "init", "-q")
    _git(root, "add", "pyproject.toml", "CITATION.cff")
    _git(root, "commit", "-q", "-m", "carriers")
    return root


def _push_tag(
    root: Path, *, local_sha: str | None = None, **extra_env: str
) -> subprocess.CompletedProcess:
    """Push `TAG` through the hook. `local_sha` defaults to the fixture's commit."""
    if local_sha is None:
        local_sha = _git(root, "rev-parse", "HEAD")
    env = dict(BASE_ENV)
    env.update(extra_env)
    env["PATH"] = str(root / "shim") + os.pathsep + env.get("PATH", "")
    stdin = f"refs/tags/{TAG} {local_sha} refs/tags/{TAG} {'0' * 40}\n"
    return subprocess.run(
        ["bash", "hook", "origin", "https://example.invalid/fixture.git"],
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=120,
    )


def _passed_check_1b(result: subprocess.CompletedProcess) -> bool:
    return SUCCESS_BANNER in result.stdout


def test_the_premise_shape_alone_admits_impossible_dates() -> None:
    for date_text in IMPOSSIBLE_DATES:
        assert OLD_SHAPE.fullmatch(date_text), date_text
        assert not _release_workflow_accepts(date_text), date_text


@pytest.mark.parametrize("date_text", IMPOSSIBLE_DATES)
def test_an_impossible_date_is_blocked(tmp_path: Path, date_text: str) -> None:
    result = _push_tag(_fixture(tmp_path, f"date-released: {date_text}"))
    assert result.returncode != 0
    assert NOT_ISO in result.stderr, result.stderr
    assert not _passed_check_1b(result), result.stdout


@pytest.mark.parametrize("date_text", REAL_PAST_DATES)
def test_a_real_date_still_passes(tmp_path: Path, date_text: str) -> None:
    assert _release_workflow_accepts(date_text)
    result = _push_tag(_fixture(tmp_path, f"date-released: {date_text}"))
    assert _passed_check_1b(result), result.stdout + result.stderr
    for message in (NOT_ISO, FUTURE, MISMATCH):
        assert message not in result.stderr, result.stderr


@pytest.mark.parametrize(
    "date_line",
    [
        'date-released: "2026-06-17"',
        "date-released: '2026-06-17'",
        "date-released: 2026-06-17 # first release",
        "date-released:\t2026-06-17",
    ],
)
def test_the_field_is_read_the_way_the_workflow_reads_it(tmp_path: Path, date_line: str) -> None:
    result = _push_tag(_fixture(tmp_path, date_line))
    assert _passed_check_1b(result), result.stdout + result.stderr


def test_crlf_line_endings_read_the_same_values(tmp_path: Path) -> None:
    result = _push_tag(_fixture(tmp_path, "date-released: 2026-06-17", crlf=True))
    assert _passed_check_1b(result), result.stdout + result.stderr


def test_a_future_date_is_still_blocked(tmp_path: Path) -> None:
    result = _push_tag(_fixture(tmp_path, "date-released: 9999-12-31"))
    assert result.returncode != 0
    assert FUTURE in result.stderr, result.stderr
    assert not _passed_check_1b(result), result.stdout


def test_a_missing_date_is_blocked(tmp_path: Path) -> None:
    result = _push_tag(_fixture(tmp_path, "abstract: no release date here"))
    assert result.returncode != 0
    assert NOT_ISO in result.stderr, result.stderr
    assert not _passed_check_1b(result), result.stdout


# Whole-file inputs the workflow rejects that a looser reader would pass: a non-blank
# whitespace character right after the colon (which the workflow's `[ \t]*` does not
# skip), and a lone CR used as a line separator (which Python's universal newlines
# split on and grep does not). Built with chr() so the source stays ASCII.
_HEAD = "cff-version: 1.2.0\ntitle: fixture\n"
CRAFTED_FILES = {
    "vt-after-date-colon": _HEAD + f"version: {VERSION}\ndate-released:{chr(0x0B)}2026-06-17\n",
    "ff-after-date-colon": _HEAD + f"version: {VERSION}\ndate-released:{chr(0x0C)}2026-06-17\n",
    "cr-after-date-colon": _HEAD + f"version: {VERSION}\ndate-released:{chr(0x0D)}2026-06-17\n",
    "nbsp-after-date-colon": _HEAD + f"version: {VERSION}\ndate-released:{chr(0xA0)}2026-06-17\n",
    "en-space-after-date-colon": _HEAD
    + f"version: {VERSION}\ndate-released:{chr(0x2002)}2026-06-17\n",
    "ideographic-space-after-date-colon": _HEAD
    + f"version: {VERSION}\ndate-released:{chr(0x3000)}2026-06-17\n",
    "vt-after-version-colon": _HEAD + f"version:{chr(0x0B)}{VERSION}\ndate-released: 2026-06-17\n",
    "lone-cr-separates-a-future-date": _HEAD
    + f"version: {VERSION}\rdate-released: 2099-01-01\ndate-released: 2024-01-01\n",
    "lone-cr-separates-an-impossible-date": _HEAD
    + f"version: {VERSION}\rdate-released: 2024-02-30\ndate-released: 2024-01-01\n",
}


@pytest.mark.parametrize("locale", ["C", "C.UTF-8"])
@pytest.mark.parametrize("name", sorted(CRAFTED_FILES))
def test_no_crafted_file_passes_the_hook_that_the_workflow_rejects(
    tmp_path: Path, name: str, locale: str
) -> None:
    cff_text = CRAFTED_FILES[name]
    assert not _release_workflow_accepts_file(cff_text), "premise: the workflow rejects it"
    result = _push_tag(_fixture(tmp_path, "", cff_text=cff_text), LC_ALL=locale)
    assert not _passed_check_1b(result), result.stdout + result.stderr


def test_the_crafted_harness_is_not_vacuous(tmp_path: Path) -> None:
    """A clean file must pass BOTH readers, or the test above proves nothing."""
    cff_text = _HEAD + f"version: {VERSION}\ndate-released: 2024-01-01\n"
    assert _release_workflow_accepts_file(cff_text)
    result = _push_tag(_fixture(tmp_path, "", cff_text=cff_text))
    assert _passed_check_1b(result), result.stdout + result.stderr


# The Release workflow checks out the commit the tag points at. The working tree at push
# time can hold something else: uncommitted edits, or a different branch entirely.
_GOOD_CFF = _HEAD + f"version: {VERSION}\ndate-released: 2024-01-01\n"


def test_the_tagged_commit_is_read_not_the_working_tree(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "", cff_text=_HEAD + "version: 2.0.2\ndate-released: 2024-01-01\n")
    (root / "CITATION.cff").write_text(_GOOD_CFF, encoding="utf-8", newline="")
    assert _release_workflow_accepts_file(_GOOD_CFF), "premise: the working tree alone passes"
    result = _push_tag(root)
    assert result.returncode != 0
    assert MISMATCH in result.stderr, result.stderr
    assert not _passed_check_1b(result), result.stdout


def test_a_bad_working_tree_does_not_block_a_good_tagged_commit(tmp_path: Path) -> None:
    """The other direction, so the fix cannot be "block whenever the two differ"."""
    root = _fixture(tmp_path, "", cff_text=_GOOD_CFF)
    (root / "CITATION.cff").write_text(
        _HEAD + "version: 2.0.2\ndate-released: 2026-02-30\n", encoding="utf-8", newline=""
    )
    (root / "pyproject.toml").write_text('[project]\nversion = "2.0.2"\n', encoding="utf-8")
    result = _push_tag(root)
    assert _passed_check_1b(result), result.stdout + result.stderr


def test_an_annotated_tag_is_peeled_to_its_commit(tmp_path: Path) -> None:
    root = _fixture(tmp_path, "", cff_text=_GOOD_CFF)
    _git(root, "tag", "-a", TAG, "-m", "fixture release")
    tag_object = _git(root, "rev-parse", TAG)
    assert tag_object != _git(root, "rev-parse", "HEAD"), "premise: a tag object of its own"
    result = _push_tag(root, local_sha=tag_object)
    assert _passed_check_1b(result), result.stdout + result.stderr


def test_a_sha_that_is_not_a_commit_here_is_blocked(tmp_path: Path) -> None:
    result = _push_tag(_fixture(tmp_path, "", cff_text=_GOOD_CFF), local_sha="1" * 40)
    assert result.returncode != 0
    assert "does not point at a commit" in result.stderr, result.stderr
    assert not _passed_check_1b(result), result.stdout


def test_a_tag_deletion_is_not_checked(tmp_path: Path) -> None:
    """A deletion pushes no commit. The hook must go on to its later checks untouched."""
    root = _fixture(tmp_path, "abstract: no release date here")
    result = _push_tag(root, local_sha="0" * 40)
    assert "Running fast test gate" in result.stdout, result.stdout + result.stderr
    for message in (NOT_ISO, FUTURE, MISMATCH, "does not point at a commit"):
        assert message not in result.stderr, result.stderr
    assert not _passed_check_1b(result), result.stdout
