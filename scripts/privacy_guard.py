"""
SESTRAV Automated Privacy & Leak Detection Guard
Scans staged or tracked files for hardcoded workstation paths, exposed credentials,
or accidentally staged private trial/scratch files.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Sensitive & Workstation Path Regexes.
# The separator class is repeated ([/\\]+) so that backslashes escaped inside a
# Python/JSON string literal are still detected, not just single-separator forms.
PATH_PATTERNS = [
    (
        r"[a-zA-Z]:[/\\]+Users[/\\]+[a-zA-Z0-9_\-]+",
        "Hardcoded Windows workstation path under a per-user profile directory",
    ),
    (r"/home/(?!sestrav_user|runner|ubuntu|vscode|node|appuser|app/)[a-zA-Z0-9_\-]+", "Hardcoded Linux User Path"),
]

SECRET_PATTERNS = [
    (r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----", "Private RSA/SSH Key Header"),
    (r"(?i)(api[_\-]?key|secret[_\-]?key|access[_\-]?token)\s*[:=]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]", "Exposed API Key/Token"),
]

PRIVATE_PATH_PREFIXES = [
    "scratch/",
    "results/trials/",
    "dry-runs/",
    "_local/",
    ".env",
]

# Files where example paths or changelog historical notes are legitimately discussed.
# These are EXACT repository-relative paths, never substrings: a file merely *containing*
# one of these names (e.g. "docs/notes_CHANGELOG.md") must not inherit the exemption.
# scripts/privacy_guard.py and scripts/run_pipeline_local.ps1 are deliberately NOT listed -
# neither contains a workstation path, so self-exemption is unnecessary and would only
# create a blind spot in the guard itself.
ALLOWED_PATH_EXCEPTIONS = frozenset(
    {
        "CHANGELOG.md",
        "AGENTS.md",
    }
)

SCAN_EXTENSIONS = {".py", ".md", ".json", ".yaml", ".yml", ".smk", ".sh", ".ps1", ".txt", ".ini", ".toml"}

# Directory prefixes to skip during the filesystem-fallback walk. Matched on whole path
# components (so "venv_notes.py" and "src/scratchpad_utils.py" are still scanned), never
# as bare substrings. Multi-component entries such as "results/trials" match as a prefix.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    ".snakemake",
    "scratch",
    "results/trials",
    "_local",
    ".cache",
}


def normalize_path(file_path: str) -> str:
    """Return a repo-relative POSIX-style path with any leading './' removed."""
    normalized = file_path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_allowed_exception(file_path: str) -> bool:
    """Exact-path membership test for the documented path-pattern exemptions."""
    return normalize_path(file_path) in ALLOWED_PATH_EXCEPTIONS


def is_excluded_dir(file_path: str) -> bool:
    """True when the path lies inside an excluded directory (component-wise match)."""
    normalized = normalize_path(file_path)
    parts = normalized.split("/")
    for entry in EXCLUDE_DIRS:
        entry_parts = entry.strip("/").split("/")
        n = len(entry_parts)
        # Match the excluded entry against any run of consecutive directory components
        # (all components except the final filename).
        for i in range(len(parts) - n):
            if parts[i : i + n] == entry_parts:
                return True
    return False


def find_git_executable() -> str | None:
    git_path = shutil.which("git")
    if git_path:
        return git_path
    
    for common_path in [
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe"),
    ]:
        if os.path.isfile(common_path):
            return common_path
    return None


def get_target_files(scan_all: bool = False) -> list[str]:
    """Get list of files to check via Git or filesystem fallback."""
    git_bin = find_git_executable()

    if git_bin:
        try:
            cmd = [git_bin, "ls-files"] if scan_all else [git_bin, "diff", "--staged", "--name-only", "--diff-filter=d"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
            if files or not scan_all:
                return files
        except Exception as e:
            print(f"Notice: Git command fallback to local file search ({e}).", file=sys.stderr)

    found_files = []
    root_path = Path(".")
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix in SCAN_EXTENSIONS:
            rel_path = path.as_posix()
            if not is_excluded_dir(rel_path):
                found_files.append(rel_path)
    return found_files


def check_file_path_leak(file_path: str) -> list[str]:
    """Check if the file itself should never be tracked in git."""
    issues = []
    normalized = normalize_path(file_path)
    for prefix in PRIVATE_PATH_PREFIXES:
        if normalized.startswith(prefix) or f"/{prefix}" in normalized:
            issues.append(f"Private file pattern detected in git: '{file_path}' (Matches rule '{prefix}')")
    return issues


def scan_file_contents(file_path: str) -> list[str]:
    """Scan content of a file for user path leaks and secrets."""
    issues: list[str] = []
    p = Path(file_path)

    if not p.is_file():
        return issues

    if p.suffix.lower() in [".png", ".jpg", ".gz", ".zip", ".parquet", ".pt", ".joblib", ".pkl"]:
        return issues

    # Exact-path exemption only - a substring test would let any path merely *containing*
    # an exempt name (e.g. "docs/notes_CHANGELOG.md") silently bypass the leak check.
    path_check_exempt = is_allowed_exception(file_path)

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Skip path pattern check for documented exception files (e.g. CHANGELOG historical notes)
                if not path_check_exempt:
                    for pattern, desc in PATH_PATTERNS:
                        if re.search(pattern, line):
                            issues.append(f"{file_path}:{line_num} -> {desc}: '{line.strip()}'")

                for pattern, desc in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        issues.append(f"{file_path}:{line_num} -> {desc}: '{line.strip()}'")
    except Exception as e:
        issues.append(f"Could not scan {file_path}: {e}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="SESTRAV Privacy Guard Scanner")
    parser.add_argument("--all", action="store_true", help="Scan all tracked/repo files instead of only staged files")
    args = parser.parse_args()

    files_to_scan = get_target_files(scan_all=args.all)

    if not files_to_scan:
        print("Privacy Guard: No staged files to scan.")
        sys.exit(0)

    print(f"Privacy Guard: Scanning {len(files_to_scan)} file(s) for local path leaks and exposed secrets...")

    all_issues = []

    for file_path in files_to_scan:
        path_issues = check_file_path_leak(file_path)
        all_issues.extend(path_issues)

        content_issues = scan_file_contents(file_path)
        all_issues.extend(content_issues)

    if all_issues:
        print("\n❌ PRIVACY GUARD FAILED: Found potential privacy/security leaks!")
        print("------------------------------------------------------------------")
        for issue in all_issues:
            print(f"  • {issue}")
        print("------------------------------------------------------------------")
        print("Please sanitize the above files before committing to open-source repository.")
        sys.exit(1)

    print("✅ Privacy Guard: All scanned files are clean and sanitized.")
    sys.exit(0)


if __name__ == "__main__":
    main()
