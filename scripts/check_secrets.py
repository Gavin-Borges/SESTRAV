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

# Unquoted value. Requiring quotes above missed the everyday leak shape entirely:
# `AWS_SECRET_ACCESS_KEY=<value>` in a .env and `export API_KEY=<value>` in a shell
# script both went BLOCK-to-allow because neither value is quoted.
# This pattern is format-scoped, and the scope is a language fact rather than a
# heuristic: in Python, JSON and TOML a string literal is ALWAYS quoted, so a bare
# right-hand side there is an expression and can never be a hardcoded credential.
# Applying it to .py was measured to flag `token = match.group(1)` inside
# check_doc_commit_refs.py's SHA_RE loop - a 13-character value with entropy 3.7,
# clearing both floors below - which turns this gate red on real tracked code.
# It captures no quoted value on purpose; it is run IN ADDITION to the pattern
# above, never instead of it, because a bare match ends at the first quote and
# could otherwise consume a keyword that a later quoted match needed.
CREDENTIAL_ASSIGNMENT_BARE = re.compile(
    r"(?i)"
    r"(api[_-]?key|token|secret|password|passwd|auth|private[_-]?key)"
    r"(?:[_-][a-z0-9]+)*['\"]?\s*[=:]\s*([^\s'\"#,;)\]}]+)"
)

# Formats in which an unquoted scalar IS the string literal.
_BARE_VALUE_SUFFIXES = (".yml", ".yaml", ".sh", ".env", ".md", ".txt", ".cfg", ".ini")

# Refuse a vacuous pass over an empty walk (wrong cwd, or every file excluded).
MIN_SCANNED_FILES = 10


def allows_bare_value(path: str) -> bool:
    name = os.path.basename(path)
    return name.endswith(_BARE_VALUE_SUFFIXES) or name.startswith("Dockerfile")


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


# Paths this run could not READ at all. A file that cannot be opened has not been
# cleared, so scan_tree turns this into a failure rather than letting it pass
# quietly. It is module-level because scan_file returns line numbers and must keep
# that signature; scan_tree resets it at the start of every run.
UNREADABLE_PATHS: List[str] = []


def scan_file(path: str) -> List[int]:
    # Returns only the line NUMBERS of credential-like assignments. The matched
    # text is deliberately never stored or returned, so a flagged value cannot be
    # logged or leaked downstream.
    flagged_line_numbers: List[int] = []
    # The bare pattern is ADDITIVE, never a replacement: the quoted pattern runs on
    # every format, so no line that is caught today can stop being caught.
    patterns = [CREDENTIAL_ASSIGNMENT]
    if allows_bare_value(path):
        patterns.append(CREDENTIAL_ASSIGNMENT_BARE)
    try:
        # errors="surrogateescape", NOT the default "strict", and this is the whole
        # point of the change. Under "strict" a single byte that is not valid UTF-8
        # raised UnicodeDecodeError, the except clause below returned the lines
        # found SO FAR, and nothing was printed - so the rest of that file was never
        # examined and the run still reported success.
        #
        # Measured 2026-09-20 on two fixtures identical except for one byte:
        #   valid UTF-8                      -> [2]   credential on line 2 FLAGGED
        #   same file, one 0xff byte line 1  -> []    nothing reported
        # One unreadable byte anywhere above a secret hid the secret.
        #
        # surrogateescape maps undecodable bytes to lone surrogates instead of
        # raising, so the scan runs to the end of the file. Credential values are
        # ASCII by construction (the patterns below match quoted or bare tokens
        # with no whitespace), so the smuggled bytes cannot mask a match.
        with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
            for line_no, line in enumerate(f, 1):
                flagged = False
                for pattern in patterns:
                    # finditer, not search: a line can carry more than one assignment,
                    # and search() inspects only the FIRST. A short decoy earlier on the
                    # line then shields a real secret later on it, which is a
                    # one-character bypass. Measured: `token = "abc"; password = "<36
                    # chars>"` went from BLOCK to allow under search().
                    for match in pattern.finditer(line):
                        val = match.group(2)
                        # Whitespace inside the captured value means prose, not a
                        # credential. This is what keeps sentences like
                        # `A password: "must be at least twelve characters"` quiet, and
                        # it discriminates on the VALUE rather than on the whole line.
                        if any(ch.isspace() for ch in val):
                            continue
                        if len(val) > 8 and calculate_entropy(val) > 3.0:
                            flagged = True
                            break
                    if flagged:
                        break
                if flagged:
                    flagged_line_numbers.append(line_no)
    except OSError:
        # The file could not be opened or read at all (permissions, a vanished
        # path, a device error). That is NOT a clean result: nothing about this
        # file has been cleared. Record it so scan_tree can fail closed.
        #
        # UnicodeDecodeError is deliberately no longer caught here. With
        # errors="surrogateescape" above it can no longer be raised by the read,
        # and catching it was what made an undecodable byte look like a clean file.
        UNREADABLE_PATHS.append(path)
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


def _ignored_paths(root: str, candidates: List[str]) -> set:
    """Absolute, normcased paths among *candidates* that git ignores.

    Empty on any failure, which keeps this SUBTRACTIVE step fail-open: if git
    cannot answer, the walk scans exactly what it found, as before.

    One subprocess for the whole candidate list. `git check-ignore` exits 0 when
    at least one path is ignored and 1 when none are, so 1 is a normal answer
    and only anything else is treated as failure.
    """
    if not candidates:
        return set()
    rels = []
    for absolute in candidates:
        try:
            rels.append(os.path.relpath(absolute, root).replace("\\", "/"))
        except ValueError:  # different drive on Windows; cannot be inside root
            continue
    if not rels:
        return set()
    try:
        out = subprocess.run(
            ["git", "-C", root, "check-ignore", "--stdin", "-z"],
            input="\0".join(rels),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    if out.returncode not in (0, 1):
        return set()
    return {
        os.path.normcase(os.path.abspath(os.path.join(root, rel)))
        for rel in out.stdout.split("\0")
        if rel
    }


def iter_scanned_files(root: str) -> List[str]:
    found: List[str] = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if _is_scannable_name(name):
                found.append(os.path.join(dirpath, name))

    # EXCLUDE_DIRS prunes by directory NAME, so a gitignored file sitting at the
    # REPO ROOT has no directory to prune and the walk opens it anyway. STATE.md
    # is the live case: gitignored, absent from HEAD, and full of prose about
    # credential patterns, so it turned this gate red locally while CI stayed
    # green. A gate that is red for a reason CI can never see is a gate people
    # learn to skip, which is the actual cost.
    #
    # Only files that are BOTH untracked and ignored are dropped, so `git add -f`
    # is not a way past this gate. TWO independent things ensure that, and the
    # distinction is recorded because the plausible answer is the wrong one:
    #
    #   1. OPERATIVE: `git check-ignore` without --no-index does not report a
    #      TRACKED file as ignored at all. It exits 1 with empty output, so a
    #      force-added file is never subtracted here in the first place.
    #   2. REDUNDANT BUT REAL: even if it were subtracted, _tracked_paths reads
    #      `git ls-files`, which reports the INDEX, so the union below re-adds
    #      it. Demonstrated by forcing --no-index on: the force-add test still
    #      passes, and fails only once BOTH layers are removed.
    #
    # An earlier draft of this comment credited (2) alone, which is the layer
    # that does not currently do the work.
    ignored = _ignored_paths(root, found)
    if ignored:
        found = [p for p in found if os.path.normcase(os.path.abspath(p)) not in ignored]

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
    UNREADABLE_PATHS.clear()
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
    # A file that could not be READ has not been cleared. Reported as its own
    # failure rather than folded into [FLAGGED], because the two mean opposite
    # things: FLAGGED is "we looked and found something", this is "we could not
    # look". Silently treating the second as a pass is the defect this gate had.
    if UNREADABLE_PATHS:
        for path in UNREADABLE_PATHS:
            print(f"[UNREADABLE] {path} (could not be opened; NOT cleared)")
        print(
            f"\n[ERROR] {len(UNREADABLE_PATHS)} file(s) could not be read, so this "
            "run cannot certify them. Action blocked."
        )
        return 1
    if has_error:
        print("\n[ERROR] Potential secrets detected. Action blocked.")
        return 1
    print("[SUCCESS] No secrets detected.")
    return 0


def main() -> None:
    sys.exit(scan_tree("."))


if __name__ == "__main__":
    main()
