"""A Dependabot pip block excludes every nested pip block's directory.

.github/dependabot.yml has one pip block for "/" and another for "/environments".
The "/" job also scans subdirectories, so without an exclusion it claims every
environments/ lock as well, and its dependency-graph snapshot of those locks went
stale. On this repository, Dependabot alerts on those locks stayed open after the
locks were fixed (alerts #142, #183, #203) or were raised against versions the
locks no longer pinned (alerts #204, #205, #206); these are alert numbers, not
pull requests. The root block therefore lists "environments/**" under
exclude-paths; the "/environments" block still covers those files.

Two properties, each with a bite case built on a tmp_path fixture:

1. For every pair of pip blocks where one block's directory is an ancestor of
   another's, the ancestor's exclude-paths contains "<relative path>/**".
2. No requirements file or pyproject.toml directly in a pip block's directory
   carries a pip include or constraint option (-r, -c, --requirement,
   --constraint) pointing into a path that block excludes, so a file the job
   still reads never depends on one it was told to skip.

FAILS IF: the root pip block drops the exclusion, a nested pip block is added
without one, or a root manifest starts including or constraining an excluded file.
"""

from __future__ import annotations

import posixpath
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPENDABOT_YML = REPO_ROOT / ".github" / "dependabot.yml"

_INCLUDE_RE = re.compile(
    r"^\s*(?:-r|-c|--requirement|--constraint)(?:\s*=\s*|\s*)(?P<target>[^\s#]+)"
)


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} did not parse to a mapping"
    return data


def _pip_blocks(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [u for u in config.get("updates") or [] if u.get("package-ecosystem") == "pip"]


def _dirs(block: dict[str, Any]) -> list[str]:
    raw = [block["directory"]] if "directory" in block else list(block.get("directories") or [])
    return [str(d).strip("/") for d in raw]


def _excludes(block: dict[str, Any]) -> list[str]:
    return [str(p) for p in block.get("exclude-paths") or []]


def _nested_pairs(config: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    """Each (ancestor block, ancestor directory, descendant path relative to it)."""
    blocks = _pip_blocks(config)
    pairs = []
    for anc in blocks:
        for anc_dir in _dirs(anc):
            for desc in blocks:
                for desc_dir in _dirs(desc):
                    if desc_dir == anc_dir:
                        continue
                    if anc_dir == "":
                        pairs.append((anc, anc_dir, desc_dir))
                    elif desc_dir.startswith(anc_dir + "/"):
                        pairs.append((anc, anc_dir, desc_dir[len(anc_dir) + 1 :]))
    return pairs


def nested_exclusion_violations(config: dict[str, Any]) -> list[str]:
    violations = []
    for block in _pip_blocks(config):
        for d in _dirs(block):
            if any(ch in d for ch in "*?["):
                violations.append(f"pip directory {d!r} is a glob; this guard does not handle it")
    for anc, anc_dir, rel in _nested_pairs(config):
        wanted = f"{rel}/**"
        if wanted not in _excludes(anc):
            violations.append(f"pip block '/{anc_dir}' lacks exclude-paths entry {wanted!r}")
    return violations


def _glob_regex(pattern: str) -> re.Pattern[str]:
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _is_excluded(rel_path: str, patterns: list[str]) -> bool:
    """True if the path, or any directory above it, matches an exclude pattern."""
    parts = rel_path.split("/")
    candidates = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
    regexes = [_glob_regex(p.strip("/")) for p in patterns]
    return any(rx.fullmatch(c) for rx in regexes for c in candidates)


def _manifests(base: Path) -> list[Path]:
    found = set(base.glob("requirements*.txt")) | set(base.glob("requirements*.in"))
    found.add(base / "pyproject.toml")
    return sorted(p for p in found if p.is_file())


def excluded_reference_violations(config: dict[str, Any], repo_root: Path) -> list[str]:
    violations = []
    for block in _pip_blocks(config):
        patterns = _excludes(block)
        if not patterns:
            continue
        for d in _dirs(block):
            for manifest in _manifests(repo_root / d if d else repo_root):
                text = manifest.read_text(encoding="utf-8")
                for n, line in enumerate(text.splitlines(), start=1):
                    m = _INCLUDE_RE.match(line)
                    if not m:
                        continue
                    target = posixpath.normpath(m.group("target").replace("\\", "/"))
                    if _is_excluded(target, patterns):
                        violations.append(f"{manifest.name} line {n} points into excluded {target}")
    return violations


_FIXTURE_TEMPLATE = """\
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
{root_extra}
  - package-ecosystem: "pip"
    directory: "/environments"
    schedule:
      interval: "weekly"
"""

_ROOT_EXCLUDE = """\
    exclude-paths:
      - "environments/**"
"""


def _write_fixture(tmp_path: Path, *, with_exclude: bool) -> Path:
    path = tmp_path / "dependabot.yml"
    extra = _ROOT_EXCLUDE if with_exclude else ""
    path.write_text(_FIXTURE_TEMPLATE.format(root_extra=extra), encoding="utf-8")
    return path


def test_real_config_excludes_each_nested_pip_directory() -> None:
    config = _load(DEPENDABOT_YML)
    pairs = _nested_pairs(config)
    assert pairs, "expected the root and environments pip blocks to nest; this guard is vacuous"
    assert nested_exclusion_violations(config) == []


def test_real_pip_manifests_do_not_point_into_excluded_paths() -> None:
    config = _load(DEPENDABOT_YML)
    assert _manifests(REPO_ROOT), "no root pip manifest found; this guard is vacuous"
    assert excluded_reference_violations(config, REPO_ROOT) == []


def test_bite_missing_nested_exclusion_is_caught(tmp_path: Path) -> None:
    bad = nested_exclusion_violations(_load(_write_fixture(tmp_path, with_exclude=False)))
    assert len(bad) == 1
    assert "environments/**" in bad[0]
    good = nested_exclusion_violations(_load(_write_fixture(tmp_path, with_exclude=True)))
    assert good == []


def test_bite_root_manifest_pointing_into_excluded_path_is_caught(tmp_path: Path) -> None:
    config = _load(_write_fixture(tmp_path, with_exclude=True))
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "x.txt").write_text("six==1.17.0\n", encoding="utf-8")
    manifest = tmp_path / "requirements.txt"

    manifest.write_text("six==1.17.0\n", encoding="utf-8")
    assert excluded_reference_violations(config, tmp_path) == []

    options = (
        "-r environments/x.txt",
        "-c ./environments/x.txt",
        "--requirement=environments/x.txt",
    )
    for option in options:
        manifest.write_text(f"six==1.17.0\n{option}\n", encoding="utf-8")
        bad = excluded_reference_violations(config, tmp_path)
        assert len(bad) == 1, option
        assert "environments/x.txt" in bad[0], option


def test_include_outside_the_excluded_path_is_allowed(tmp_path: Path) -> None:
    config = _load(_write_fixture(tmp_path, with_exclude=True))
    (tmp_path / "requirements.txt").write_text("-r requirements.in\n", encoding="utf-8")
    assert excluded_reference_violations(config, tmp_path) == []
