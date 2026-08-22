"""ENC-1: enforce `.claude/rules/encoding.md` in CI, not only in the local pre-commit hook.

Two gaps motivated this, both measured rather than assumed (2026-08-22):

1. **No CI gate existed at all.** The em-dash ban is enforced by `scripts/hooks/pre-commit`
   Gate 3, which is a *local* hook - it does not run on a fresh clone, in CI, or for a
   contributor who has not installed it. Nothing under `.github/workflows/` checked encoding.

2. **The hook scans one character; the rule bans a class.** Gate 3 looks for U+2014 only,
   while rule 2 bans en-dashes, curly quotes, ellipsis and non-breaking spaces as well.

The rule's own rationale is CI failure `fa91670`: a bare em-dash in a `print()` call crashed
on a runner without a UTF-8 locale. That failure mode was reproduced directly here - before
this gate landed, `scripts/check_repo_status.py` died at its first status line with
`UnicodeEncodeError` under both `cp1252` (the default Windows console) and `ascii`, exiting 1.

Note the severity ordering, because it is counter-intuitive and is why this test does not
simply mirror Gate 3's single-character check: U+2713/U+2717/U+2264 fail to encode under
`ascii`, `cp1252` AND `latin-1`, whereas the *banned* U+2014 survives `cp1252`. The
characters the rule did not name were the more dangerous ones in practice.

Rule 3 permits Unicode in genuine data (peptide sequences, HLA allele names, scientific
notation). Both tests below are therefore scoped deliberately: the first to strings the
program actually writes to a stream, the second to the specific typographic characters
rule 2 enumerates - never to Unicode as such.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Logging methods treated as emitting to a stream, same as print().
_LOGGING_METHODS = {"info", "warning", "error", "debug", "critical", "exception"}

# The characters `.claude/rules/encoding.md` rules 1-2 name explicitly (eight of them), plus
# U+2011 NON-BREAKING HYPHEN, which the rule does not enumerate but which belongs to the same
# class and is trivially confusable with an ASCII hyphen.
#
# Built with chr() rather than written out, and rather than spelled as "\uXXXX" escapes.
# Escapes are not sufficient: this file is itself a tracked .py file, and
# `test_no_non_ascii_string_literals_outside_the_allowlist` below inspects `ast.Constant`
# VALUES, which resolve "\u2014" back to the em-dash. An escape hides the character from the
# raw-text scan and from the pre-commit hook while leaving it fully visible to the AST scan,
# so the gate would fail on its own source the moment this file is committed. `chr(0x2014)`
# leaves the AST holding an int, so no string literal here carries a non-ASCII character by
# either measure.
BANNED_TYPOGRAPHIC = {
    chr(0x2014): "EM DASH (rule 1)",
    chr(0x2013): "EN DASH",
    chr(0x2018): "LEFT SINGLE QUOTATION MARK",
    chr(0x2019): "RIGHT SINGLE QUOTATION MARK",
    chr(0x201C): "LEFT DOUBLE QUOTATION MARK",
    chr(0x201D): "RIGHT DOUBLE QUOTATION MARK",
    chr(0x2026): "HORIZONTAL ELLIPSIS",
    chr(0x00A0): "NO-BREAK SPACE",
    chr(0x2011): "NON-BREAKING HYPHEN",
}

# Extensions carrying code, config or prose - the set the 2026-08-22 census covered.
TEXT_SUFFIXES = {
    ".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt", ".smk", ".sh", ".json",
}

# Modules allowed to hold non-ASCII in ordinary (non-docstring) string literals. Each entry
# is a measured exception, and each is a KNOWN BLIND SPOT rather than a proof of safety - the
# exemption is file-level, so it also disables the indirect-literal scan for the rest of that
# module. Recorded honestly, because an earlier draft of this comment claimed these modules
# "provably do not reach a locale-dependent stream", and that was FALSE for two of the three:
#
#   app/demo.py                     TWO sinks, not one. Streamlit widget/markdown text
#                                   rendered in a browser (UTF-8 by definition), AND
#                                   matplotlib text baked into the generated PDF scorecard
#                                   (`_build_pdf_scorecard`), which is a glyph-availability
#                                   question, not an encoder one. Neither sink is a terminal.
#   src/external_validation_finalize.py         Report bodies are written through
#   src/external_validation_tier_b_finalize.py  `open(..., encoding="utf-8")`, so those are
#                                   safe. BUT both modules ALSO print to stdout/stderr
#                                   (finalize's `[finalize] ...` lines near the end of main,
#                                   tier_b's `[tier-b-finalize] ...` lines). Their CURRENT
#                                   non-ASCII literals are scientific notation bound for the
#                                   utf-8 report files, which rule 3 permits - but nothing
#                                   here stops a future non-ASCII literal in these two
#                                   modules from reaching those print() calls uncaught.
#
# Anything NOT listed here must stay ASCII, because a string literal can be handed to a
# logger far from where it is written: `src/verify/promote_gnn.py` built its gate messages
# as `GateResult` fields and only later emitted them via `logger.info`, which is exactly the
# path the print()/logging check below cannot see.
NON_ASCII_LITERAL_ALLOWLIST = {
    "app/demo.py",
    "src/external_validation_finalize.py",
    "src/external_validation_tier_b_finalize.py",
}


def _tracked(*patterns: str) -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "ls-files", *patterns],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line]


def _is_emitting_call(node: ast.AST) -> bool:
    """True for `print(...)` and `<anything>.info/warning/error/debug/critical/exception(...)`."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name) and node.func.id == "print":
        return True
    return isinstance(node.func, ast.Attribute) and node.func.attr in _LOGGING_METHODS


def _emitted_non_ascii(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, offending characters) for each emitting call carrying non-ASCII.

    Walks f-string parts too: `ast.walk` reaches the `ast.Constant` segments inside a
    `JoinedStr`, which is where every real instance of this defect has lived.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    hits = []
    for node in ast.walk(tree):
        if not _is_emitting_call(node):
            continue
        literal = "".join(
            sub.value
            for sub in ast.walk(node)
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
        )
        offenders = sorted({c for c in literal if ord(c) > 127})
        if offenders:
            hits.append((node.lineno, "".join(offenders)))
    return hits


def test_no_non_ascii_in_runtime_emitted_strings():
    """A string the program prints must survive an ascii/cp1252 stdout, or the run dies."""
    violations = []
    for path in _tracked("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        for lineno, offenders in _emitted_non_ascii(path):
            codepoints = " ".join(f"U+{ord(c):04X}" for c in offenders)
            violations.append(f"{relative}:{lineno} emits {codepoints}")
    assert not violations, (
        "print()/logging call(s) emit non-ASCII, which raises UnicodeEncodeError on a "
        "runner without a UTF-8 locale (see .claude/rules/encoding.md, CI failure fa91670). "
        "Use the ASCII equivalent - '+/-', 'x', '<=', '[+]', '[x]':\n" + "\n".join(violations)
    )


def _docstring_node_ids(tree: ast.Module) -> set[int]:
    """Ids of Constant nodes that are module/class/function docstrings."""
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def test_no_non_ascii_string_literals_outside_the_allowlist():
    """Catches text that reaches a stream indirectly, which the print()/logging check cannot.

    `promote_gnn.py` is the worked example: its gate messages were built as `GateResult`
    fields and emitted by a `logger.info` many lines away, so no scan of the logging call
    itself could see them.
    """
    violations = []
    for path in _tracked("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in NON_ASCII_LITERAL_ALLOWLIST:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        skip = _docstring_node_ids(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in skip:
                continue
            offenders = sorted({c for c in node.value if ord(c) > 127})
            if offenders:
                codepoints = " ".join(f"U+{ord(c):04X}" for c in offenders)
                violations.append(f"{relative}:{node.lineno} string literal holds {codepoints}")
    assert not violations, (
        "non-ASCII string literal(s) outside NON_ASCII_LITERAL_ALLOWLIST. A literal can be "
        "emitted far from where it is written, so it must stay ASCII unless its module is "
        "listed with a measured reason:\n" + "\n".join(violations)
    )


def test_non_ascii_literal_allowlist_entries_are_tracked_and_still_needed():
    """Keeps the allowlist from outliving its justification.

    Membership is decided by `git ls-files`, deliberately NOT by `Path.is_file()`. An
    untracked entry is inert - the scan above never reaches its `continue` - and worse,
    `is_file()` answers differently in a dev working tree than in a fresh `actions/checkout`
    that materialises only tracked files. That exact mistake made a sibling gate
    (`test_torch_load_weights_only.py`) pass locally and fail the required `test (3.13)` job.
    """
    tracked = {path.relative_to(REPO_ROOT).as_posix() for path in _tracked("*.py")}
    untracked = sorted(NON_ASCII_LITERAL_ALLOWLIST - tracked)
    no_longer_needed = []
    for relative in sorted(NON_ASCII_LITERAL_ALLOWLIST & tracked):
        path = REPO_ROOT / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        skip = _docstring_node_ids(tree)
        has_non_ascii = any(
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
            and any(ord(c) > 127 for c in node.value)
            for node in ast.walk(tree)
        )
        if not has_non_ascii:
            no_longer_needed.append(relative)
    assert not untracked, (
        "NON_ASCII_LITERAL_ALLOWLIST names file(s) that git does not track, so they are never "
        f"scanned and the entry exempts nothing: {untracked}"
    )
    assert not no_longer_needed, (
        "file(s) in NON_ASCII_LITERAL_ALLOWLIST no longer contain any non-ASCII literal and "
        f"should be removed from it: {no_longer_needed}"
    )


def _routes_docstring_to_argparse(tree: ast.Module) -> bool:
    """True if the module passes `__doc__` as an argparse description/epilog."""
    return any(
        isinstance(node, ast.Call)
        and any(
            kw.arg in {"description", "epilog"}
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "__doc__"
            for kw in node.keywords
        )
        for node in ast.walk(tree)
    )


def test_module_docstrings_routed_to_argparse_are_ascii():
    """A docstring handed to argparse is printed, so it is output, not documentation.

    The scans above deliberately skip docstrings, which would otherwise leave this path
    uncovered: 27 tracked modules pass `__doc__` into `argparse` as a description or epilog,
    and argparse writes it to stdout on `--help` through the locale's encoder. Measured, not
    supposed - `scripts/precompute_self_similarity.py` carried U+00D7 in its docstring and
    `PYTHONIOENCODING=ascii python scripts/precompute_self_similarity.py --help` died at
    `argparse.py` `_print_message` with UnicodeEncodeError, exit 1, which is the same
    signature as CI failure fa91670. It survived a cp1252 console only because cp1252 happens
    to map U+00D7; U+2713 and friends would have crashed there too.
    """
    violations = []
    for path in _tracked("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        if not _routes_docstring_to_argparse(tree):
            continue
        offenders = sorted({c for c in (ast.get_docstring(tree) or "") if ord(c) > 127})
        if offenders:
            codepoints = " ".join(f"U+{ord(c):04X}" for c in offenders)
            violations.append(f"{relative} docstring holds {codepoints} and reaches argparse")
    assert not violations, (
        "module docstring(s) are printed by argparse on --help and must be ASCII, or the "
        "command dies with UnicodeEncodeError on a runner without a UTF-8 locale:\n"
        + "\n".join(violations)
    )


def test_the_encoding_scans_actually_cover_the_repository():
    """Keeps the gates above from passing vacuously.

    Every assertion in this module is of the form "no violations found", which is also what
    a scan over an EMPTY file set reports. If `git ls-files` returned nothing - git missing
    from PATH, a source tarball, REPO_ROOT resolving outside the work tree - all four checks
    would go green while inspecting nothing, and CI would read as assurance over an
    unenforced rule. The floors are sanity bounds well under the real counts, not targets to
    keep updated.
    """
    python_files = _tracked("*.py")
    text_files = [path for path in _tracked() if path.suffix in TEXT_SUFFIXES]
    assert len(python_files) > 50, (
        f"git ls-files returned only {len(python_files)} tracked Python file(s) from "
        f"{REPO_ROOT}; the encoding scans are not seeing the repository"
    )
    assert len(text_files) > 100, (
        f"git ls-files returned only {len(text_files)} tracked text file(s) from {REPO_ROOT}; "
        "the banned-character scan is not seeing the repository"
    )
    # The AST walk must still find emitting calls - and the two branches of _is_emitting_call
    # are floored SEPARATELY on purpose. A single combined floor is branch-blind: print() is
    # roughly four fifths of the emit surface here, so losing print() detection entirely would
    # still leave the logging matches above a shared threshold, and the gate would go quiet
    # over the exact call form the CI failure was about.
    unparseable = 0
    print_files = 0
    logging_files = 0
    for path in python_files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            unparseable += 1
            continue
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        if any(isinstance(c.func, ast.Name) and c.func.id == "print" for c in calls):
            print_files += 1
        if any(
            isinstance(c.func, ast.Attribute) and c.func.attr in _LOGGING_METHODS for c in calls
        ):
            logging_files += 1
    assert print_files > 50, (
        f"only {print_files} tracked file(s) matched the print() form; the AST matcher has "
        "probably stopped recognising it, and the emitted-string gate is now near-vacuous"
    )
    assert logging_files > 10, (
        f"only {logging_files} tracked file(s) matched the logging form; the AST matcher has "
        "probably stopped recognising it"
    )
    # A file the scans cannot parse is skipped silently everywhere else in this module, so a
    # syntax/grammar split between the local interpreter and the required `test (3.13)` job
    # would shrink coverage with no signal. Assert the skip set stays empty.
    assert unparseable == 0, (
        f"{unparseable} tracked Python file(s) failed to parse, so they are silently excluded "
        "from every encoding scan in this module"
    )


def test_banned_typographic_characters_are_absent_from_tracked_files():
    """Rule 1 and rule 2: the local pre-commit hook checks only U+2014; this checks the class."""
    violations = []
    for path in _tracked():
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            for char, name in BANNED_TYPOGRAPHIC.items():
                if char in line:
                    violations.append(f"{relative}:{lineno} contains U+{ord(char):04X} {name}")
    assert not violations, (
        "banned typographic character(s) found; use the ASCII equivalent "
        "(.claude/rules/encoding.md rules 1-2):\n" + "\n".join(violations)
    )
