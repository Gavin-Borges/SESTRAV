import os
import re
import sys
import math
from typing import List

# Keywords associated with credentials
KEYWORD_PATTERN = re.compile(
    r"(?i)(api_key|token|secret|password|passwd|auth|private_key)\s*=\s*[\'\"].{4,}[\'\"]"
)


# High entropy strings (likely random API keys / secrets)
def calculate_entropy(s: str) -> float:
    if not s:
        return 0.0
    entropy = 0.0
    for x in range(256):
        p_x = float(s.count(chr(x))) / len(s)
        if p_x > 0:
            entropy += -p_x * math.log(p_x, 2)
    return entropy


# Folders to exclude from secret scan
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
}

# Files to exclude from secret scan to avoid false positives in ruleset/scanning scripts
EXCLUDE_FILES = {"apply-branch-ruleset.ps1", "apply_protection.sh", "check_secrets.py"}


def scan_file(path: str) -> List[int]:
    # Returns only the line NUMBERS of credential-like assignments. The matched
    # text is deliberately never stored or returned, so a flagged value cannot be
    # logged or leaked downstream.
    flagged_line_numbers: List[int] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                # Check for standard assignment keywords
                if KEYWORD_PATTERN.search(line):
                    # Check if the matched string is actually high entropy to filter false positives
                    # Find assignments of format name = "value"
                    matches = re.findall(r"=\s*[\'\"]([^\'\"]+)[\'\"]", line)
                    for val in matches:
                        if len(val) > 8 and calculate_entropy(val) > 3.0:
                            flagged_line_numbers.append(line_no)
                            break
    except (OSError, UnicodeDecodeError):
        # Unreadable or non-UTF-8 files cannot contain a scannable line; skip them.
        return flagged_line_numbers
    return flagged_line_numbers


def main() -> None:
    has_error = False
    for root, dirs, files in os.walk("."):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f in EXCLUDE_FILES:
                continue
            if f.endswith(
                (
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
            ) or f.startswith("Dockerfile"):
                path = os.path.join(root, f)
                flagged = scan_file(path)
                for line_no in flagged:
                    # Report only the location; the matched value is never captured (no clear-text logging).
                    print(
                        f"[FLAGGED] {path}:{line_no} (credential-like assignment; value not shown)"
                    )
                    has_error = True

    if has_error:
        print("\n[ERROR] Potential secrets detected. Action blocked.")
        sys.exit(1)
    else:
        print("[SUCCESS] No secrets detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
