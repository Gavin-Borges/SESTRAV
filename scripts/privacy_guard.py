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

# Sensitive & Workstation Path Regexes
PATH_PATTERNS = [
    (r"[a-zA-Z]:[/\\]Users[/\\][a-zA-Z0-9_\-]+", "Hardcoded Windows Workstation Path (e.g. C:\\Users\\username)"),
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

# Files where example paths or changelog historical notes are legitimately discussed
ALLOWED_PATH_EXCEPTIONS = [
    "CHANGELOG.md",
    "privacy_guard.py",
    "run_pipeline_local.ps1",
    "AGENTS.md",
]

SCAN_EXTENSIONS = {".py", ".md", ".json", ".yaml", ".yml", ".smk", ".sh", ".ps1", ".txt", ".ini", ".toml"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".hypothesis", ".snakemake", "scratch", "results/trials", "_local", ".cache"}


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
            if not any(ex in rel_path for ex in EXCLUDE_DIRS):
                found_files.append(rel_path)
    return found_files


def check_file_path_leak(file_path: str) -> list[str]:
    """Check if the file itself should never be tracked in git."""
    issues = []
    normalized = file_path.replace("\\", "/")
    for prefix in PRIVATE_PATH_PREFIXES:
        if normalized.startswith(prefix) or f"/{prefix}" in normalized:
            issues.append(f"Private file pattern detected in git: '{file_path}' (Matches rule '{prefix}')")
    return issues


def scan_file_contents(file_path: str) -> list[str]:
    """Scan content of a file for user path leaks and secrets."""
    issues = []
    p = Path(file_path)

    if not p.is_file():
        return issues

    if p.suffix.lower() in [".png", ".jpg", ".gz", ".zip", ".parquet", ".pt", ".joblib", ".pkl"]:
        return issues

    is_allowed_exception = any(exc in file_path for exc in ALLOWED_PATH_EXCEPTIONS)

    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                # Skip path pattern check for documented exception files (e.g. CHANGELOG historical notes)
                if not is_allowed_exception:
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
