"""`_version_key` orders versions the way PEP 440 does, pre-releases included.

Why this file exists
--------------------
`tools/check_lockfile_freshness.py` backs the CI gate
`.github/workflows/dependency-lockfile-check.yml`, and one of that gate's jobs is
enforcing the ``>=`` CVE floors declared in `environments/requirements-lock.in`.
It does that at a single site: `check_pair` reports a "below floor" problem when
`_version_key(compiled) < _version_key(floor)`.

Until 2026-10-25 `_version_key` was `re.findall(r"\\d+", version)`, which read a
pre-release number as an extra RELEASE segment. ``0.66b1`` became ``(0, 66, 1)``
and ``0.66`` became ``(0, 66)``, so the gate ranked a pre-release ABOVE its own
final release and a floor of ``>=0.66`` was satisfied, in the gate's view, by a
lock pinning ``0.66b1``. pip and PEP 440 both read that pin as below the floor.
**That is fail-open in a security gate**, and `_version_key` had no direct test
of any kind before this file: `git grep -l _version_key -- tests/` returned
nothing, which is why it survived.

A first PEP 440 rewrite of the key marked an absent dev segment with the
sentinel ``(1 << 20,)``. Date-stamped dev numbers such as ``.dev20261024``
exceed that bound, so ``1.0rc1.dev20261024`` sorted ABOVE ``1.0rc1`` - the same
fail-open shape again, for a floor of ``>=1.0rc1`` - and at exactly
``.dev1048576`` a dev release compared EQUAL to its parent. The tests below that
name date-stamped dev numbers exist to catch that sentinel, which every other
test here passed.

The repo is not hypothetically exposed to pre-release versions. Measured on
2026-10-25 at `5e72008d` across the 28 tracked lockfiles and `.in` sources,
232 distinct version strings appear; all but two are plain dotted-numeric, and
those two are ``0.58b0``, a pre-release, and ``2.9.0.post0``.

How these tests avoid being vacuous
-----------------------------------
`packaging.version.Version` is the oracle, and it is a genuinely independent
one: the gate itself is stdlib-only by design, because its CI job installs
nothing before running it, so the implementation under test cannot and does not
call `packaging`. Two of the tests below compare the two over every ordered pair
of a corpus rather than over hand-picked examples, and each asserts a floor on
the corpus size and on the shapes it contains, so a corpus that silently stopped
being collected would fail instead of passing trivially.
"""

from __future__ import annotations

import glob
import itertools
import re
import sys
import textwrap

import pytest
from packaging.version import Version

from tools import check_lockfile_freshness as freshness
from tools.check_lockfile_freshness import REPO_ROOT, _version_key, check_pair

_HASH = "--hash=sha256:a53b324a331d807ab68d442bc777a9de1894d82e6fff082cdeae5b5e4355bd68"

# The four pairs the old `re.findall` key got WRONG, each a pre-release ranked
# above its own release, plus three it got right. Kept as an explicit list so a
# future rewrite has to face the exact regression rather than a generated sample
# that might not contain it.
_PRERELEASE_REGRESSIONS = [
    ("0.66b1", "0.66"),
    ("1.0rc1", "1.0"),
    ("2.0a1", "2.0"),
    ("1.0.dev1", "1.0"),
]
_ALREADY_CORRECT = [
    ("1.2.3.post1", "1.2.3"),
    ("1.2.3", "1.2.3.post1"),
    ("0.66b2", "0.66b10"),
]

# (lower, higher): a dev release sorts below the version it develops towards,
# however large its dev number. The first three pairs, and the boundary pair,
# are the ones the `(1 << 20,)` dev sentinel got wrong; 1,048,575 sits just
# under that bound and the bare-release pair is decided by the pre-release
# sentinel instead, so both are controls that the old sentinel also passed.
_DATE_STAMPED_DEV = [
    ("1.0rc1.dev20261024", "1.0rc1"),
    ("2.9.0.post0.dev20261024", "2.9.0.post0"),
    ("1.0.post1.dev20261024", "1.0.post1"),
    ("1.0a1.dev1048576", "1.0a1"),
    ("1.0a1.dev1048575", "1.0a1"),
    ("1.0.dev20261024", "1.0a1"),
]

# (lower, higher) and (equal, equal) local versions, per PEP 440: segments split
# on ".", "-" and "_", text segments sort below numeric ones, text compares
# case-insensitively, numbers numerically, and a prefix sorts first. The first
# rewrite ordered two local versions by their digit runs alone and got each of
# the ordered pairs below wrong.
_LOCAL_ORDER = [
    ("1.0+a2", "1.0+abc"),
    ("1.0+abc.5", "1.0+5"),
    ("1.0+abc", "1.0+cpu"),
    ("1.0+b1", "1.0+1.a"),
    ("1.0+1", "1.0+1.a"),
    ("1.0+cu118", "1.0+5"),
]
_LOCAL_EQUAL = [
    ("1.0+ABC", "1.0+abc"),
    ("1.0+1.0", "1.0+1-0"),
    ("1.0+1_0", "1.0+1.0"),
    ("1.0+01", "1.0+1"),
]

# Deliberately wider than anything in the repo: epochs, local versions, dev and
# post releases, every PEP 440 pre-release alias, a `v` prefix, the implicit
# post form `1.0-1`, surrounding whitespace, and date-stamped dev numbers on
# both sides of the old 1,048,576 sentinel. The repo has none of these shapes
# today, so without them a regression in that handling would be invisible here.
# The second block is the 80-version probe set an independent review of the
# first rewrite measured it with; it is kept whole so that measurement stays
# reproducible from this file.
_CORPUS = sorted(
    set(
        [
            "0.58b0", "0.66", "0.66b1", "0.66b2", "0.66.1", "0.67",
            "1.0", "1.0.0", "1.0a1", "1.0b1", "1.0rc1", "1.0.preview3", "1.0c1",
            "1.0.post1", "1.0rev2", "1.0-1", "1.0.dev1", "1.0a1.dev1",
            "1!1.0", "1!0.9", "2.0", "2.0a1", "v1.0", "1.0+local1", "1.0+local2",
            "1.29.0", "1.30.0", "1.37.0", "1.45.1", "2.9.0.post0",
            "3.2.0", "3.2.1", "84.0.0", "84.1.0", "2026.10.19",
            "2.9.0.post0.dev20261024", "1.0+ABC", "1.0+1-0", "1.0+1_0", "1.0+01",
        ]
        + [
            "0", "00", "0.0", "1", "1.0", "1.0.0", "1.0.0.0.0", "01.02.03", "1.00",
            "1.0.1", "1.0.0.0.1", "1.2.3.4.5.6.7.8.9.10", "2026.10.19",
            "99999999999999999999.0",
            "0!1.0", "1!0.0.1", "2!1.0", "1!1.0a1", "1!2.0rc1.post2.dev3+local.7",
            "1.0+abc", "1.0+abc.5", "1.0+5", "1.0+1.a", "1.0+b1", "1.0+a2",
            "1.0+ubuntu-1", "1.0+cu118", "1.0+cpu", "1.0+0", "1.0+00", "1.0+1.0",
            "1.0+1",
            "1.0.dev0", "1.0.dev", "1.0dev1", "1.0-dev2", "1.0_dev3",
            "1.0.dev20261024", "1.0a1.dev20261024", "1.0rc1.dev1",
            "1.0rc1.dev20261024",
            "1.0.post1.dev1", "1.0.post1.dev20261024", "1.0a1.post1.dev1",
            "1.0.dev1048576", "1.0a1.dev1048575", "1.0a1.dev1048576",
            "1.0.post", "1.0post1", "1.0-post1", "1.0_post1", "1.0.r1", "1.0rev1",
            "1.0-1", "1.0-r", "1.0.post0", "1.0.post20261024",
            "1.0a", "1.0alpha1", "1.0.a.1", "1.0-a-1", "1.0beta2", "1.0b", "1.0c",
            "1.0pre1", "1.0preview1", "1.0rc", "1.0.rc.1", "1.0RC1", "1.0-rc_1",
            "1.0a2000000", "1.0rc2000000",
            "v1.0", "V1.0", "v1.0rc1", "1.0a1", "1.0a1.post1", "1.0b2.post345.dev456",
            " 1.0 ", "1.0\t",
        ]
    )
)

_PIN_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^]]*\])?\s*(==|>=)\s*([0-9][^\s,;]*)"
)


def _repo_versions() -> list[str]:
    """Every version string pinned or floored anywhere in the repo's specs.

    Read off disk rather than hard-coded, so this corpus grows with the repo and
    a newly introduced version shape is covered the day it lands.
    """
    patterns = (
        "requirements*.txt",
        "requirements*.in",
        "environments/*.txt",
        "environments/*.in",
        "environments/*.lock",
    )
    found: set[str] = set()
    for pattern in patterns:
        for path in glob.glob(str(REPO_ROOT / pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.split("#")[0].strip().rstrip("\\").strip()
                    match = _PIN_RE.match(stripped)
                    if match is not None:
                        found.add(match.group(3))
    return sorted(found)


def _cmp(left, right) -> int:
    return (left > right) - (left < right)


def _disagreements(versions: list[str]) -> list[tuple[str, str]]:
    """Ordered pairs on which the gate's key and PEP 440 order differently.

    Three-way, so a pair the key calls EQUAL while PEP 440 orders it (the
    1,048,576 boundary did exactly that) is a disagreement too.
    """
    keys = {v: _version_key(v) for v in versions}
    oracle = {v: Version(v) for v in versions}
    return [
        (left, right)
        for left, right in itertools.permutations(versions, 2)
        if _cmp(keys[left], keys[right]) != _cmp(oracle[left], oracle[right])
    ]


def _write_pair(tmp_path, in_text: str, lock_lines: list[str]):
    in_file = tmp_path / "requirements-lock.in"
    in_file.write_text(in_text, encoding="utf-8")
    out_file = tmp_path / "requirements.lock"
    out_file.write_text(
        "".join(f"{line} \\\n    {_HASH}\n" for line in lock_lines), encoding="utf-8"
    )
    return in_file, out_file


@pytest.mark.parametrize(("lower", "higher"), _PRERELEASE_REGRESSIONS)
def test_a_prerelease_sorts_below_its_own_final_release(lower, higher):
    """The exact fail-open: each pair returned True from the old key."""
    assert _version_key(lower) < _version_key(higher)
    # And the oracle agrees, so the assertion above is not just this gate's own
    # opinion about what the answer should be.
    assert Version(lower) < Version(higher)


@pytest.mark.parametrize(("left", "right"), _PRERELEASE_REGRESSIONS + _ALREADY_CORRECT)
def test_regression_pairs_match_the_oracle(left, right):
    assert (_version_key(left) >= _version_key(right)) == (
        Version(left) >= Version(right)
    )


@pytest.mark.parametrize(("lower", "higher"), _DATE_STAMPED_DEV)
def test_a_date_stamped_dev_release_sorts_below_what_it_develops_towards(lower, higher):
    """No dev number is large enough to lift a dev release over its parent."""
    assert Version(lower) < Version(higher)
    assert _version_key(lower) < _version_key(higher)
    assert not _version_key(lower) >= _version_key(higher)


@pytest.mark.parametrize(("lower", "higher"), _LOCAL_ORDER)
def test_two_local_versions_order_by_pep440_segments(lower, higher):
    assert Version(lower) < Version(higher)
    assert _version_key(lower) < _version_key(higher)


@pytest.mark.parametrize(("left", "right"), _LOCAL_EQUAL)
def test_equivalent_local_spellings_compare_equal(left, right):
    assert Version(left) == Version(right)
    assert _version_key(left) == _version_key(right)


def test_every_ordered_pair_of_the_corpus_matches_the_oracle():
    """Agreement over all ordered pairs, not a hand-picked sample."""
    assert len(_CORPUS) >= 100, f"corpus looks under-collected: {len(_CORPUS)}"
    # Anti-vacuity: the corpus must actually contain the shapes this test exists
    # to cover, or agreement over it would prove nothing about them.
    assert any("b" in v or "rc" in v or "a1" in v for v in _CORPUS)
    assert any("!" in v for v in _CORPUS)
    assert sum("+" in v for v in _CORPUS) >= 10
    assert any(".dev" in v for v in _CORPUS)
    assert any(".post" in v or "rev" in v for v in _CORPUS)
    # Date-stamped dev numbers on a pre-release, a post-release and a bare
    # release, each past the old 1,048,576 sentinel.
    for shape in (r"(a|rc)\d+\.dev\d{7,}$", r"\.post\d+\.dev\d{7,}$", r"^1\.0\.dev\d{7,}$"):
        assert any(re.search(shape, v) for v in _CORPUS), shape

    disagreements = _disagreements(_CORPUS)
    assert not disagreements, (
        "_version_key disagrees with PEP 440 on these ordered pairs: "
        f"{disagreements[:10]}"
    )


def test_every_ordered_pair_of_the_live_repo_versions_matches_the_oracle():
    """The same comparison over the versions this repo actually pins today.

    This is the drift check. The corpus above is fixed, so it cannot notice a
    version shape that enters the lockfiles later; this one reads them.
    """
    versions = _repo_versions()
    assert len(versions) >= 150, f"repo corpus looks under-collected: {len(versions)}"
    # Anti-vacuity: at least one real pre-release must be present, because that
    # is the defect class. `0.58b0` is on main today.
    assert any(
        re.search(r"\d(a|b|rc)\d*$", v) for v in versions
    ), "no pre-release found in the repo's pins; this test would prove nothing"

    disagreements = _disagreements(versions)
    assert not disagreements, (
        "_version_key disagrees with PEP 440 on live repo pins: "
        f"{disagreements[:10]}"
    )


def test_unreadable_version_raises_rather_than_guessing():
    """Fail closed and loudly. The old key guessed from the digits it found."""
    for bad in ("", "not-a-version", "1.0.0bad!", "::1"):
        with pytest.raises(ValueError, match="not a PEP 440 version"):
            _version_key(bad)


def test_non_ascii_lookalike_letters_are_rejected_with_valueerror():
    """PEP 440 segments are ASCII; Unicode case folding must not widen them.

    Each probe swaps one ASCII letter for a character that Python's Unicode
    IGNORECASE folds onto it: the dotless i (U+0131) into "preview", the long s
    (U+017F) into "post", and the Kelvin sign (U+212A) as a local segment. The
    first one used to match the grammar and then raise KeyError from the alias
    table, which no caller handled. packaging 26.1 and 26.3 reject all three;
    26.0, 25.0 and 24.2 accept them (each measured), so this test deliberately
    does not consult the installed `packaging`.
    """
    for bad in (
        "1.0prev" + chr(0x0131) + "ew1",
        "1.0po" + chr(0x017F) + "t1",
        "1.0+" + chr(0x212A),
    ):
        with pytest.raises(ValueError, match="not a PEP 440 version"):
            _version_key(bad)


def test_trailing_zeros_do_not_change_the_order():
    assert _version_key("1.0") == _version_key("1.0.0")
    assert Version("1.0") == Version("1.0.0")


def test_check_pair_reports_a_prerelease_sitting_below_its_declared_floor(tmp_path):
    """End to end, through the gate's own comparison site.

    This is the test that covers the security property rather than the helper:
    a declared CVE floor of a final release, against a lock pinning a
    pre-release of that same release. The old key reported no problem here.
    """
    in_file = tmp_path / "requirements-lock.in"
    in_file.write_text("opentelemetry-instrumentation>=0.66\n", encoding="utf-8")
    out_file = tmp_path / "requirements.lock"
    out_file.write_text(
        textwrap.dedent(
            """\
            opentelemetry-instrumentation==0.66b1 \\
                --hash=sha256:a53b324a331d807ab68d442bc777a9de1894d82e6fff082cdeae5b5e4355bd68
            """
        ),
        encoding="utf-8",
    )
    problems = check_pair(in_file, out_file)
    assert any("below floor" in problem for problem in problems), problems


def test_check_pair_reports_a_date_stamped_dev_pin_below_its_post_release_floor(tmp_path):
    """The dev-sentinel fail-open, end to end.

    ``2.9.0.post0`` is a version shape this repo already pins. A lock pinning a
    date-stamped dev release of it sits below a floor of it, and the
    ``(1 << 20,)`` sentinel reported no problem here.
    """
    in_file, out_file = _write_pair(
        tmp_path, "python-dateutil>=2.9.0.post0\n", ["python-dateutil==2.9.0.post0.dev20261024"]
    )
    problems = check_pair(in_file, out_file)
    assert any("below floor" in problem for problem in problems), problems


def test_check_pair_still_accepts_a_pin_that_meets_its_floor(tmp_path):
    """The companion direction, so the test above is not passing for free."""
    in_file = tmp_path / "requirements-lock.in"
    in_file.write_text("opentelemetry-instrumentation>=0.66b1\n", encoding="utf-8")
    out_file = tmp_path / "requirements.lock"
    out_file.write_text(
        textwrap.dedent(
            """\
            opentelemetry-instrumentation==0.66b1 \\
                --hash=sha256:a53b324a331d807ab68d442bc777a9de1894d82e6fff082cdeae5b5e4355bd68
            """
        ),
        encoding="utf-8",
    )
    assert check_pair(in_file, out_file) == []


@pytest.mark.parametrize(
    ("in_text", "lock_line"),
    [
        ("idna>=3.x\n", "idna==3.10"),  # unreadable floor
        ("idna>=3.7\n", "idna==3.10.x"),  # unreadable compiled pin
    ],
)
def test_check_pair_reports_a_version_it_cannot_order_instead_of_raising(
    tmp_path, in_text, lock_line
):
    """An unreadable version is a reported problem, not a traceback.

    Both sides pass the gate's own line parsers, so only the version grammar
    can refuse them. Before, the ValueError escaped check_pair and aborted the
    run before any other problem was printed.
    """
    in_file, out_file = _write_pair(tmp_path, in_text, [lock_line])
    problems = check_pair(in_file, out_file)
    assert len(problems) == 1, problems
    assert "cannot compare" in problems[0]
    assert "not a PEP 440 version" in problems[0]
    assert "the floor is unchecked" in problems[0]


def test_main_annotates_an_unreadable_floor_and_still_reports_the_rest(
    tmp_path, monkeypatch, capsys
):
    """End to end through main(): rc 1, a GitHub ::error:: line, other problems kept."""
    in_file, _ = _write_pair(
        tmp_path,
        "idna>=3.x\naiohttp>=3.14.2\n",
        ["idna==3.10", "aiohttp==3.14.1"],
    )
    monkeypatch.setattr(freshness, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        freshness, "LOCKFILE_PAIRS", (("requirements-lock.in", "requirements.lock"),)
    )
    monkeypatch.setattr(freshness, "_git_tracked_in_files", lambda: [in_file.name])
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr(sys, "argv", ["check_lockfile_freshness.py", "--check"])

    assert freshness.main() == 1
    lines = capsys.readouterr().out.splitlines()
    unreadable = [line for line in lines if "cannot compare idna==3.10" in line]
    below = [line for line in lines if "aiohttp>=3.14.2 floor" in line and "below floor" in line]
    assert len(unreadable) == 1 and unreadable[0].startswith("::error::"), lines
    assert len(below) == 1 and below[0].startswith("::error::"), lines
