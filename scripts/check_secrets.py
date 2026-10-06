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
    r"(api[_-]?key|token|secret|password|passwd|pwd|credentials?|auth"
    r"|private[_-]?key)"
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
    r"(api[_-]?key|token|secret|password|passwd|pwd|credentials?|auth"
    r"|private[_-]?key)"
    r"(?:[_-][a-z0-9]+)*['\"]?\s*[=:]\s*([^\s'\"#,;)\]}]+)"
)

# Makefiles get their own bare form. A lone `:` there separates a rule's target
# from its prerequisites (`auth-check: scripts/check.py`), so only Make's assignment
# operators count: `=`, `:=`, `::=`, `?=`, `+=` and `!=`. The shared form above
# would also mis-read `API_TOKEN := value` as the operator ":" and the value "=",
# which the length floor then drops.
CREDENTIAL_ASSIGNMENT_MAKE = re.compile(
    r"(?i)"
    r"(api[_-]?key|token|secret|password|passwd|pwd|credentials?|auth"
    r"|private[_-]?key)"
    r"(?:[_-][a-z0-9]+)*\s*(?:::=|[:?+!]?=)\s*([^\s'\"#,;)\]}]+)"
)

# Shapes probe2 measured as missed, re-measured on this revision before being added: 8 of 9
# planted 24-character secrets were NOT detected while the plain `API_KEY = "<v>"` control was.
# Each pattern below captures the value as group 2, so all of them drop into the same
# no-whitespace, length and entropy floors as the assignment patterns above - none of them
# lowers the bar, they only widen what reaches it. They are ADDITIVE: no line caught today
# stops being caught.

# 1. A type annotation between the name and the `=`. The pattern above requires a quote
#    directly after `[=:]`, so `API_TOKEN: str = "<v>"` read the `:` as the operator and the
#    annotation as the value, which the length floor then dropped.
CREDENTIAL_ANNOTATED = re.compile(
    r"(?i)"
    r"(api[_-]?key|token|secret|password|passwd|pwd|credentials?|auth|private[_-]?key)"
    r"(?:[_-][a-z0-9]+)*\s*:\s*[A-Za-z_][A-Za-z0-9_.\[\], |]*\s*=\s*"
    r"[bfu]?['\"]([^'\"]+)['\"]"
)

# A RAW prefix is deliberately excluded from every prefix class below. Measured on the
# live tree: the sole false positive the five patterns produced was
# `VERSION_TOKEN = rf"..."` in tools/check_version_carriers.py - a regex template whose
# name ends in TOKEN. A raw literal is a regex or a path template, not a credential, so
# excluding `r` removes that class rather than allowlisting one line of it. The class is
# `[bfu]` and ONE character is enough: all 24 legal Python string prefixes were compiled,
# the non-raw set is exactly {b, f, u} case-insensitively, and every legal
# multi-character prefix contains `r`.

# 2. A string PREFIX before the quote. `f"..."`, `b"..."`, `rb"..."` and the rest put a letter
#    where the pattern above demands a quote. At least one prefix character is required here,
#    so this never duplicates a match the unprefixed pattern already makes.
CREDENTIAL_PREFIXED_STRING = re.compile(
    r"(?i)"
    r"(api[_-]?key|token|secret|password|passwd|pwd|credentials?|auth|private[_-]?key)"
    r"(?:[_-][a-z0-9]+)*['\"]?\s*[=:]\s*[bfu]['\"]([^'\"]+)['\"]"
)

# 3. A default handed to os.getenv / os.environ.get. There is no assignment operator between
#    the credential NAME and the value at all: the name is the first argument and the secret
#    is the second, so every assignment pattern above is blind to it by construction.
CREDENTIAL_ENV_DEFAULT = re.compile(
    r"(?i)(getenv|environ\.get)\(\s*[bfu]?['\"][^'\"]*"
    r"(?:api[_-]?key|token|secret|password|passwd|pwd|credential|auth|private[_-]?key)"
    r"[^'\"]*['\"]\s*,\s*[bfu]?['\"]([^'\"]+)['\"]"
)

# 4. An Authorization header. The scheme name is the keyword, and the credential follows a
#    SPACE rather than an operator, so nothing above reaches it. Quoting is optional because
#    the header is as often built in a string as assigned.
CREDENTIAL_BEARER = re.compile(
    r"(?i)(bearer|authorization\s*:\s*bearer)\s+[bfu]?['\"]?"
    r"([A-Za-z0-9._~+/=-]{9,})"
)

# 5. A credential in a URL query string. `?token=<v>` carries the keyword and the value with
#    no quote between them, inside what is otherwise an ordinary quoted URL.
CREDENTIAL_QUERY_PARAM = re.compile(
    r"(?i)[?&](api[_-]?key|token|secret|password|passwd|pwd|auth)"
    r"=([^&'\"\s>]+)"
)

# A credential embedded in a URL's userinfo. Keyword-independent for the same
# reason the vendor formats are: `postgres://user:<value>@host/db` names nothing
# the patterns above recognise, so the assignment rules never saw it. The value
# is group 2 so this pattern drops straight into the same length and entropy
# gate as the others, which also keeps documentation placeholders quiet: a
# literal `://user:password@host` has an 8-character value and does not clear
# the `len > 8` floor.
#
# Measured over all 547 scanned files at b080b7ef: zero hits with the floors
# applied AND zero with no filter at all, so this adds no false positive to the
# tree it is being introduced on.
URL_EMBEDDED_CREDENTIAL = re.compile(
    r"([a-z][a-z0-9+.\-]*://[^\s:@/]+):([^\s@/]+)@"
)

# NOT added, and the measurement is recorded so it is not re-proposed blindly.
# An audit recommended widening the keyword group to cover `*_KEY` names such as
# ENCRYPTION_KEY, which the alternation above cannot reach: it carries
# `api_key` and `private_key` but no bare `key`, and the group matches SUFFIXES
# after the keyword, never prefixes before it.
#
# The obvious form, `[a-z0-9]+[_-]key`, was measured against every scanned file
# at b080b7ef, 547 of them, and produces SIX hits on tracked code, every one a
# false positive. Re-measured after PRs #540, #541 and #543 landed: 549 files
# scanned, the SAME six hits at the same lines. The count of files moves; the
# finding does not, which is why the six are named individually below, by
# SYMBOL rather than line number so the reference cannot rot:
# RATCHET_KEY = "exempt_ledger_citation_ceiling"
# (in scripts/check_doc_line_citations.py), BASELINE_KEY =
# "iedb_ebv_hpv16_tcell" (in src/continuous_validation.py), and four
# `score_key` lines in src/external_benchmark_comparison.py - the one that
# assigns it a string literal, and the three that pass it as a dict-rename key
# to a literal column name.
# Those are column names and lookup keys: long, mixed-alphabet, and well over
# the entropy floor. Adding the pattern would turn this gate red on legitimate
# code, which is the failure the bare-value pattern above already documents for
# a different rule. `credentials?` and `pwd` were measured the same way and
# produce zero hits, which is why they ARE in the alternation.

# Vendor-issued credential FORMATS, matched without a keyword and without an
# entropy floor. Both patterns above need an ASSIGNMENT: a credential-class name,
# then `=` or `:`, then the value. That is the right shape for a home-made secret
# and the wrong one for a vendor token, which is self-identifying - it is a
# credential wherever it appears, assigned to an innocuous name, embedded in a
# URL, or sitting in a file on its own.
#
# Ported 2026-09-23 from scripts/hooks/pre-commit's Gate 2 CRED_PATTERNS, which is
# a LOCAL hook and runs on no CI machine. Measured before porting: this scanner -
# the only content gate CI runs, at .github/workflows/security.yml - flagged NONE
# of them. A bare token assigned to `default_cred`, the same token inside a clone
# URL, and a PEM private-key header all passed. So the two gates disagreed about
# what a secret is, and the weaker one was the one guarding the public remote.
#
# Written as Python re rather than the hook's POSIX ERE, but deliberately not
# "improved": a pattern that differs between the two gates is a pattern whose
# behaviour has to be reasoned about twice.
VENDOR_CREDENTIAL_FORMATS = tuple(
    re.compile(p)
    for p in (
        r"sk-ant-api[0-9A-Za-z_-]{20,}",
        r"sk-[a-zA-Z0-9]{48}",
        r"AIza[0-9A-Za-z_-]{35}",
        # Boundary-anchored, and the anchoring is load-bearing rather than tidy.
        # Protein FASTA uses the 20-letter amino-acid alphabet, a subset of
        # [A-Z], so an unanchored `(AKIA|ASIA)[0-9A-Z]{16}` collides with
        # sequence data - the hook records a real false positive on
        # DENV2_NGC_panel1.fasta:31 ("...FTDPASIAARGYISTRVEMGEAAGIF..."). This
        # scanner does not open .fasta today, but it opens .txt and .md, and the
        # anchored form costs nothing.
        r"(^|[^0-9A-Za-z])(AKIA|ASIA)[0-9A-Z]{16}([^0-9A-Za-z]|$)",
        r"ghp_[a-zA-Z0-9]{36}",
        r"ghs_[a-zA-Z0-9]{36}",
        r"xox[baprs]-[0-9A-Za-z-]{10,}",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"github_pat_[A-Za-z0-9_]{20,}",
        r"gh[oupsr]_[A-Za-z0-9]{20,}",
        r"sk-(proj|svcacct)-[A-Za-z0-9_-]{20,}",
        r"sk-ant-[A-Za-z0-9_-]{20,}",
    )
)

# Formats in which an unquoted scalar IS the string literal.
_BARE_VALUE_SUFFIXES = (
    ".yml",
    ".yaml",
    ".sh",
    ".env",
    ".md",
    ".txt",
    ".cfg",
    ".ini",
    # Both were already in _SCAN_SUFFIXES, so the file was opened and then read with the
    # quoted-value parse only. A container definition and a Quarto document both carry
    # `export NAME=value`, whose value is unquoted, so every such line went unmatched.
    ".def",
    ".qmd",
)

# Refuse a vacuous pass over an empty walk (wrong cwd, or every file excluded).
MIN_SCANNED_FILES = 10


_MAKEFILE_NAMES = ("Makefile", "GNUmakefile", "makefile")


def _normalised(path: str) -> str:
    normalised = path.replace(os.sep, "/").replace("\\", "/")
    return normalised[2:] if normalised.startswith("./") else normalised


def allows_bare_value(path: str) -> bool:
    normalised = _normalised(path)
    name = normalised.rsplit("/", 1)[-1]
    # An extensionless file under _SCAN_DIRS is a shell hook, where an unquoted
    # right-hand side IS the string, as in `.sh`. Matched as a path substring rather
    # than a prefix because pre-push scans pushed blobs under a temporary root, and
    # limited to extensionless names so a helper module placed there (a `.py`) keeps
    # its own format's rules.
    in_hook_dir = any(("/" + normalised).find("/" + d) >= 0 for d in _SCAN_DIRS)
    return (
        name.endswith(_BARE_VALUE_SUFFIXES)
        or name.startswith("Dockerfile")
        or (in_hook_dir and "." not in name)
    )


def is_makefile(path: str) -> bool:
    return _normalised(path).rsplit("/", 1)[-1] in _MAKEFILE_NAMES


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

# Repo-relative POSIX paths, NOT basenames. Keyed to the path deliberately: a
# basename set excludes a file of that name ANYWHERE in the tree, so a future
# tests/fixtures/check_secrets.py, or any vendored copy, would be skipped by a
# gate nobody had asked to skip it. That is a WIDENING of a security gate's
# blind spot, and it widens silently as the tree grows.
#
# This is the FILE-name analogue of a defect this same function already had and
# already fixed for DIRECTORY names: EXCLUDE_DIRS prunes the walk by directory
# name, which measurably hid 26 tracked files under results/ until tracked files
# were pulled back in below. Same mechanism, same direction, one level down.
#
# Verified 2026-09-20 before narrowing, because an exclusion that is load-bearing
# cannot simply be tightened: scan_file returns ZERO findings for all three of
# these paths, so the gate is green with or without them. They are kept, rather
# than deleted, as a deliberate guard for the day one of them gains an example
# credential pattern - check_secrets.py is exactly the file where that would
# happen.
EXCLUDE_PATHS = frozenset(
    {
        "scripts/apply-branch-ruleset.ps1",
        "tools/apply_protection.sh",
        "scripts/check_secrets.py",
    }
)

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
    # Added 2026-09-23. Each was measured to be a live hole, not a precaution:
    # scan_file finds a planted assignment in every one of these formats, so the
    # detection layer was never the problem - _is_scannable simply never handed
    # the file over. See the block below for the measurement.
    ".smk",
    ".qmd",
    ".ipynb",
    ".lock",
    # .cfg, .ini and .env were already listed in _BARE_VALUE_SUFFIXES above,
    # which decides how a value is parsed ONCE A FILE IS OPEN. They were absent
    # here, so no file of those types was ever opened and that branch was
    # unreachable. Adding them makes an existing, declared capability live.
    ".cfg",
    ".ini",
    ".env",
)

# Text files whose NAME carries no extension. Extensionless files cannot be
# selected by suffix at all, which is how all four git hooks - the very scripts
# that enforce this repo's credential policy - went unscanned.
_SCAN_NAMES = frozenset(
    {
        "Makefile",
        "Snakefile",
        "LICENSE",
        "CODEOWNERS",
    }
)

# Directories whose tracked contents are executable text whatever they are named.
# A name list would go stale the day a fifth hook is added; a directory rule
# covers it without an edit.
_SCAN_DIRS = ("scripts/hooks/",)


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
    patterns = [
        CREDENTIAL_ASSIGNMENT,
        CREDENTIAL_ANNOTATED,
        CREDENTIAL_PREFIXED_STRING,
        CREDENTIAL_ENV_DEFAULT,
        CREDENTIAL_BEARER,
        CREDENTIAL_QUERY_PARAM,
        URL_EMBEDDED_CREDENTIAL,
    ]
    if allows_bare_value(path):
        patterns.append(CREDENTIAL_ASSIGNMENT_BARE)
    elif is_makefile(path):
        patterns.append(CREDENTIAL_ASSIGNMENT_MAKE)
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
                # Keyword-independent pass. Runs only when the assignment
                # patterns found nothing, purely to save work: it is ADDITIVE, so
                # no line that is caught today can stop being caught.
                if not flagged:
                    for vendor in VENDOR_CREDENTIAL_FORMATS:
                        if vendor.search(line):
                            flagged = True
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


def _is_scannable(rel_path: str) -> bool:
    """Decide scannability from a REPO-RELATIVE path, not from a basename.

    Both callers pass a path relative to the scan root. Separators are
    normalised to '/' so the same EXCLUDE_PATHS entries work on Windows, where
    os.walk yields backslashes while `git ls-files` yields forward slashes.
    """
    rel = rel_path.replace(os.sep, "/").replace("\\", "/")
    if rel.startswith("./"):
        rel = rel[2:]
    if rel in EXCLUDE_PATHS:
        return False
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith(_SCAN_DIRS):
        return True
    return (
        name.endswith(_SCAN_SUFFIXES)
        or name in _SCAN_NAMES
        or name.startswith("Dockerfile")
    )


def _tracked_paths(root: str) -> List[str]:
    """Repo-relative paths of tracked files, or [] outside a work tree.

    Returning [] on failure keeps this an ADDITIVE safety net: a non-git
    checkout scans exactly what the walk found, as before, rather than erroring.
    """
    try:
        # -z, and the sibling _ignored_paths below already uses it on both its
        # input and its output. Without it git applies core.quotePath, which
        # defaults to true: a path holding a byte outside ASCII comes back
        # wrapped in double quotes with octal escapes, as
        # `"caf\303\251_config.py"`. That string names no file on disk, so the
        # os.path.isfile() guard in iter_scanned_files drops it silently.
        #
        # Measured 2026-09-23, and the blast radius is narrower than it looks:
        # the os.walk finds such a file anywhere it is not pruned, so the defect
        # bites only where this function is load-bearing - a tracked file under
        # an EXCLUDE_DIRS name. In a throwaway repo with two credential-bearing
        # tracked files under results/, the ASCII-named one was scanned and the
        # non-ASCII-named one was not.
        #
        # -z also makes the split unambiguous for the other quoting case, a path
        # containing a literal newline, which splitlines() turned into two
        # entries that each named nothing.
        # encoding="utf-8" is NOT decoration, and -z alone does not fix this.
        # git writes path bytes as UTF-8; text=True decodes with
        # locale.getpreferredencoding(), which is cp1252 on this Windows box. The
        # bytes caf\xc3\xa9 came back decoded as two characters and re-encoded to
        # caf\xc3\x83\xc2\xa9, a double-encoded name that matches nothing on
        # disk, so os.path.isfile() in iter_scanned_files dropped it exactly as
        # the quoted form did. Fixing the quoting without fixing the decoding
        # moves the failure rather than removing it.
        out = subprocess.run(
            ["git", "-C", root, "ls-files", "-z"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            check=False,
        )
    except OSError:
        return []
    if out.returncode != 0:
        return []
    return [entry for entry in out.stdout.split("\0") if entry]


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
            # Same reason as _tracked_paths above, on both directions: the
            # candidate paths written in and the ignored paths read back.
            encoding="utf-8",
            errors="surrogateescape",
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
            absolute = os.path.join(dirpath, name)
            # Relative to the scan root, so EXCLUDE_PATHS is matched against the
            # same shape `git ls-files` produces for the tracked pass below.
            if _is_scannable(os.path.relpath(absolute, root)):
                found.append(absolute)

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
        if not _is_scannable(rel):
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
