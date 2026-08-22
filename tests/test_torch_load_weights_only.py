"""SEC-13: `torch.load(...)` must pin `weights_only=True` in every tracked Python file.

The original sweep (STATE.md 2026-07-xx security-ci-sweep: "torch.load audit: all runtime
paths enforce weights_only=True (only 2 dev-only checkpoint-inspect scripts use False,
nosec, off serving path)") was a one-off manual audit; this test makes it a standing,
enforced check so a new call site cannot silently regress T2 (arbitrary code execution
via a malicious model file, `docs/threat_model.md`) instead of relying on Bandit B614,
which is Advisory and does not gate the merge (see `SECURITY.md`'s CI gate map).

Scope: the enforced surface is exactly what `git ls-files` reports, because that is the
only code the repository actually ships. The two checkpoint-inspection scripts named in
that audit are gitignored local dev tools, so they fall outside the surface entirely and
need no exemption; see ALLOWED_WEIGHTS_ONLY_FALSE below for why that distinction matters.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Escape hatch for a TRACKED file with a genuine, audited reason to defeat weights_only:
# loading an arbitrary or legacy local checkpoint for manual inspection or migration is
# the one case where `False` is defensible, and such a call site must also carry its own
# `# nosec` justification. Deliberately empty right now - every tracked torch.load call
# pins weights_only=True, so there is nothing to exempt.
#
# It is NOT the home for the two dev-only tools from the SEC-13 audit. Those live at
# scripts/inspect_checkpoint.py and scripts/resave_checkpoint.py, are gitignored by
# basename (.gitignore "inspect_checkpoint.py" / "resave_checkpoint.py"), and so never
# appear in _tracked_python_files() at all. Listing them here was worse than redundant:
# the guard below used to validate entries with Path.is_file(), which is true in a dev
# working tree and false in a fresh `actions/checkout` that materialises only HEAD. The
# suite therefore passed locally and failed the required `test (3.13)` job. Entries must
# name tracked files, and the guard now enforces exactly that, so the two environments
# cannot silently disagree again.
ALLOWED_WEIGHTS_ONLY_FALSE: set[str] = set()


def _tracked_python_files() -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [REPO_ROOT / line for line in out.splitlines() if line]


def _is_torch_load_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _weights_only_value(call: ast.Call) -> object:
    """Return the literal value passed for `weights_only=`, or a sentinel if absent/non-literal."""
    for kw in call.keywords:
        if kw.arg == "weights_only":
            if isinstance(kw.value, ast.Constant):
                return kw.value.value
            return "non-literal"
    return "missing"


def _find_torch_load_calls(path: pathlib.Path) -> list[tuple[int, object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        (node.lineno, _weights_only_value(node))
        for node in ast.walk(tree)
        if _is_torch_load_call(node)
    ]


def test_torch_load_pins_weights_only_true_outside_the_dev_allowlist():
    violations = []
    for path in _tracked_python_files():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in ALLOWED_WEIGHTS_ONLY_FALSE:
            continue
        for lineno, value in _find_torch_load_calls(path):
            if value is not True:
                violations.append(f"{relative}:{lineno} weights_only={value!r}")
    assert not violations, (
        "torch.load(...) call(s) outside the dev-only allowlist do not pin "
        "weights_only=True (T2, docs/threat_model.md):\n" + "\n".join(violations)
    )


def test_dev_allowlist_entries_are_tracked_and_still_use_weights_only_false():
    # Guards the allowlist itself, against the same file set the enforcement test scans.
    # Membership is decided by `git ls-files`, not by Path.is_file(): an untracked entry
    # is inert (the loop above never reaches its `continue`) and, worse, would make the
    # verdict depend on which files happen to sit in the working tree, which is precisely
    # how this test once passed locally and failed CI. Checking against the tracked set
    # gives the same answer in a dev checkout and in a fresh `actions/checkout`.
    tracked = {path.relative_to(REPO_ROOT).as_posix() for path in _tracked_python_files()}
    untracked = sorted(ALLOWED_WEIGHTS_ONLY_FALSE - tracked)
    # An entry that has since been hardened to weights_only=True should be dropped, so the
    # allowlist never stays wider than the current, real exceptions.
    now_compliant = []
    for relative in sorted(ALLOWED_WEIGHTS_ONLY_FALSE & tracked):
        calls = _find_torch_load_calls(REPO_ROOT / relative)
        if calls and all(value is True for _, value in calls):
            now_compliant.append(relative)
    assert not untracked, (
        "ALLOWED_WEIGHTS_ONLY_FALSE names file(s) that git does not track, so they are "
        "never scanned and the entry exempts nothing. Remove them, or commit the file if "
        f"it genuinely needs an audited exemption: {untracked}"
    )
    assert not now_compliant, (
        "file(s) in ALLOWED_WEIGHTS_ONLY_FALSE now pin weights_only=True everywhere and "
        f"should be removed from the allowlist: {now_compliant}"
    )


def test_the_weights_only_scan_actually_covers_the_repository():
    # With an empty allowlist the guard above is vacuously true, so this is the assertion
    # that keeps the suite honest. The enforcement test reports "no violations" over
    # whatever _tracked_python_files() yields; if that ever returns nothing (git missing
    # from PATH, a source-tarball or otherwise non-git checkout, REPO_ROOT resolving
    # outside the work tree) or if no match survives (a refactor to `from torch import
    # load` would slip past _is_torch_load_call entirely), the check would pass while
    # inspecting nothing and CI would stay green over an unenforced T2. A security gate
    # that can succeed by looking at zero call sites is worse than none, because it reads
    # as assurance. The floor is a sanity bound well under the current tracked count, not
    # a target to keep updated.
    tracked = _tracked_python_files()
    assert len(tracked) > 50, (
        f"git ls-files returned only {len(tracked)} tracked Python file(s) from {REPO_ROOT}; "
        "the SEC-13 scan is not seeing the repository"
    )
    with_calls = [path for path in tracked if _find_torch_load_calls(path)]
    assert with_calls, (
        "no tracked file contained a torch.load(...) call, so the enforcement test above "
        "asserted over an empty set; either every call site was removed or the AST matcher "
        "no longer recognises how torch.load is invoked"
    )
