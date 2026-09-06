import os
import re
import subprocess  # nosec B404 - fixed argv, no shell, reads git ls-files only
import sys
import math
from typing import List

# Credential-class identifier, then an assignment. Entropy is applied to the
# captured value, not used as a second pass over the whole line. The old
# scanner required `keyword\s*=\s*["']` and therefore missed YAML/JSON `:`
# assignment and names like AWS_SECRET_ACCESS_KEY (keyword is not adjacent
# to `=`). Suffixes after the keyword are allowed; `author =` is not, because
# `or` is not a `_`-separated suffix.
# No left anchor. An earlier revision required `(?:^|[^a-z0-9])` before the keyword,
# which silently dropped every camelCase and run-together credential name that the
# previous scanner caught: accessToken, sessionToken, mytoken, authtoken, apitoken,
# userpassword, dbpassword, clientsecret. Measured: 8 names went from BLOCK to allow.
# The anchor was never needed for the `author =` exclusion either, which is enforced
# by the `\s*[=:]` requirement below: in `author = "..."` the characters after `auth`
# are `or`, not a `_`-separated suffix, so the assignment part cannot match.
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)"
    r"(api[_-]?key|token|secret|password|passwd|auth|private[_-]?key)"
    r"(?:[_-][a-z0-9]+)*['\"]?\s*[=:]\s*['\"]([^'\"]+)['\"]"
)

# Refuse a vacuous pass over an empty walk (wrong cwd, or every file excluded).
MIN_SCANNED_FILES = 10


def calculate_entropy(s: str) -> float:
    if not s:
        return 0.0
    entropy = 0.0
    for x in range(256):
        p_x = float(s.count(chr(x))) / len(s)
        if p_x > 0:
            entropy += -p_x * math.log(p_x, 2)
    return entropy


EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".ci_test_venv",
    ".pytest_cache",
    ".hypothesis",
    ".snakemake",
    "__pycache__",
    "release_artifacts",
    "results",
    "scratch",
    ".pytest_tmp2",
    "_local",
    ".claude",
    ".cursor",
    ".codex",
    ".agents",
    ".ruff_cache",
    ".mypy_cache",
    "build",
}

EXCLUDE_FILES = {"apply-branch-ruleset.ps1", "apply_protection.sh", "check_secrets.py"}

_SCAN_SUFFIXES = (
    ".py",
    ".sh",
    ".ps1",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".md",
    ".toml",
    ".cff",
    ".in",
    ".def",
)


def scan_file(path: str) -> List[int]:
    # Returns only the line NUMBERS of credential-like assignments. The matched
    # text is deliberately never stored or returned, so a flagged value cannot be
    # logged or leaked downstream.
    flagged_line_numbers: List[int] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                # finditer, not search: a line can carry more than one assignment,
                # and search() inspects only the FIRST. A short decoy earlier on the
                # line then shields a real secret later on it, which is a
                # one-character bypass. Measured: `token = "abc"; password = "<36
                # chars>"` went from BLOCK to allow under search().
                for match in CREDENTIAL_ASSIGNMENT.finditer(line):
                    val = match.group(2)
                    # Whitespace inside the captured value means prose, not a
                    # credential. This is what keeps sentences like
                    # `A password: "must be at least twelve characters"` quiet, and
                    # it discriminates on the VALUE rather than on the whole line.
                    if any(ch.isspace() for ch in val):
                        continue
                    if len(val) > 8 and calculate_entropy(val) > 3.0:
                        flagged_line_numbers.append(line_no)
                        break
    except (OSError, UnicodeDecodeError):
        return flagged_line_numbers
    return flagged_line_numbers


def _is_scannable_name(name: str) -> bool:
    if name in EXCLUDE_FILES:
        return False
    return name.endswith(_SCAN_SUFFIXES) or name.startswith("Dockerfile")


def _tracked_paths(root: str) -> List[str]:
    """Repo-relative paths of tracked files, or [] outside a work tree.

    Returning [] on failure keeps this an ADDITIVE safety net: a non-git
    checkout scans exactly what the walk found, as before, rather than erroring.
    """
    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if out.returncode != 0:
        return []
    return [line for line in out.stdout.splitlines() if line]


def iter_scanned_files(root: str) -> List[str]:
    found: List[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if _is_scannable_name(name):
                found.append(os.path.join(dirpath, name))

    # EXCLUDE_DIRS prunes the walk by directory NAME, which is right for build
    # output and virtualenvs but wrong for anything TRACKED: a tracked file is
    # published content, and publishing it is exactly the thing this gate exists
    # to stop. Measured 2026-09-04: 26 tracked files under results/ had scannable
    # suffixes and were never opened, so a credential committed to, say,
    # results/data_bias_audit.md passed CI silently.
    #
    # Additive by construction. Nothing is removed from EXCLUDE_DIRS, so
    # untracked material under those names - .venv, __pycache__, _local, the
    # gitignored assistant trees - stays unscanned and the walk's cost is
    # unchanged. Only tracked files are pulled back in.
    seen = {os.path.normcase(os.path.abspath(p)) for p in found}
    for rel in _tracked_paths(root):
        if not _is_scannable_name(os.path.basename(rel)):
            continue
        absolute = os.path.abspath(os.path.join(root, rel))
        key = os.path.normcase(absolute)
        if key not in seen and os.path.isfile(absolute):
            found.append(absolute)
            seen.add(key)
    return found


def scan_tree(root: str, min_files: int = MIN_SCANNED_FILES) -> int:
    paths = iter_scanned_files(root)
    if len(paths) < min_files:
        print(
            f"[ERROR] scanned {len(paths)} files (floor {min_files}); "
            "refusing a vacuous pass. Run from the repository root."
        )
        return 1
    has_error = False
    for path in paths:
        for line_no in scan_file(path):
            print(
                f"[FLAGGED] {path}:{line_no} (credential-like assignment; value not shown)"
            )
            has_error = True
    if has_error:
        print("\n[ERROR] Potential secrets detected. Action blocked.")
        return 1
    print("[SUCCESS] No secrets detected.")
    return 0


def main() -> None:
    sys.exit(scan_tree("."))


if __name__ == "__main__":
    main()
