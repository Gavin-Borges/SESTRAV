import os
import re
import sys
import math
from typing import List, Tuple

# Keywords associated with credentials
KEYWORD_PATTERN = re.compile(
    r'(?i)(api_key|token|secret|password|passwd|auth|private_key)\s*=\s*[\'\"].{4,}[\'\"]'
)

# High entropy strings (likely random API keys / secrets)
def calculate_entropy(s: str) -> float:
    if not s:
        return 0.0
    entropy = 0.0
    for x in range(256):
        p_x = float(s.count(chr(x))) / len(s)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

# Folders to exclude from secret scan
EXCLUDE_DIRS = {
    '.git', '.venv', '.ci_test_venv', '.pytest_cache', '.hypothesis',
    '.snakemake', '__pycache__', 'release_artifacts', 'results', 'scratch',
    '.pytest_tmp2'
}

# Files to exclude from secret scan to avoid false positives in ruleset/scanning scripts
EXCLUDE_FILES = {
    'apply-branch-ruleset.ps1',
    'apply_protection.sh',
    'check_secrets.py'
}

def scan_file(path: str) -> List[Tuple[int, str]]:
    secrets_found: List[Tuple[int, str]] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                # Check for standard assignment keywords
                if KEYWORD_PATTERN.search(line):
                    # Check if the matched string is actually high entropy to filter false positives
                    # Find assignments of format name = "value"
                    matches = re.findall(r'=\s*[\'\"]([^\'\"]+)[\'\"]', line)
                    for val in matches:
                        if len(val) > 8 and calculate_entropy(val) > 3.0:
                            secrets_found.append((line_no, line.strip()))
                            break
    except Exception:
        pass
    return secrets_found

def main() -> None:
    has_error = False
    for root, dirs, files in os.walk('.'):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f in EXCLUDE_FILES:
                continue
            if f.endswith(('.py', '.sh', '.ps1', '.yaml', '.yml', '.json', '.txt', '.md')):
                path = os.path.join(root, f)
                found = scan_file(path)
                for line_no, line in found:
                    print(f'[SECRET DETECTED] {path}:{line_no}: {line}')
                    has_error = True
                    
    if has_error:
        print("\n[ERROR] Potential secrets detected. Action blocked.")
        sys.exit(1)
    else:
        print("[SUCCESS] No secrets detected.")
        sys.exit(0)

if __name__ == '__main__':
    main()
