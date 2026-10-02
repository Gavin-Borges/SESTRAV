"""pre-push: the header summary names every bannered section, in file order.

scripts/hooks/pre-push opens each check with a banner line that starts
"# ---- " and summarises the checks in the comment block at the top of the
file. That summary has fallen behind the banners before: it described
Checks 1 to 5 while the file carried more banners than that. This test pins
the summary to the banners, so a check cannot be added, removed or reordered
without the summary following it.

Instrument: a banner is any line matching ``^\\s*# ---- `` (any indentation,
so Check 1b's indented banner counts). A summary entry is a line of the
leading comment block that starts ``# Check <name> --`` or
``# Shared setup``. The two lists must be equal, in order.

Scope: this checks names and order only. Check 6's second pass has no banner
of its own, so no banner-based instrument can see it; the summary's Check 6
entry describes it in prose, which this test does not read.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SOURCE = REPO_ROOT / "scripts" / "hooks" / "pre-push"

ANY_BANNER = re.compile(r"^\s*# ---- ")
BANNER = re.compile(r"^\s*# ---- (?:Check (\w+):|(Shared setup)\b)")
ENTRY = re.compile(r"^# (?:Check (\w+) --|(Shared setup)\b)")


def _label(match: re.Match[str]) -> str:
    return match.group(1) or match.group(2)


def _hook_lines() -> list[str]:
    return HOOK_SOURCE.read_text(encoding="utf-8").splitlines()


def _summary(lines: list[str]) -> list[str]:
    """The leading comment block: every line up to the first that is not a comment."""
    block: list[str] = []
    for line in lines:
        if not line.startswith("#"):
            break
        block.append(line)
    return block


def test_every_banner_is_a_check_or_the_shared_setup() -> None:
    banners = [line for line in _hook_lines() if ANY_BANNER.match(line)]
    assert banners, "no '# ---- ' banner found"
    unknown = [line for line in banners if not BANNER.match(line)]
    assert not unknown, f"banner of a kind the summary does not describe: {unknown}"


def test_the_summary_lists_every_banner_in_file_order() -> None:
    lines = _hook_lines()
    banners = [_label(m) for line in lines if (m := BANNER.match(line))]
    entries = [_label(m) for line in _summary(lines) if (m := ENTRY.match(line))]
    assert len(banners) >= 2, banners
    assert entries == banners, f"summary {entries} != banners {banners}"
