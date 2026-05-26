"""
Build PRIME training peptide list for overlap analysis.

PRIME 2.1 does not ship a public training peptide CSV in the GitHub repo.
This script:
  1) Searches a local PRIME2.1 install for peptide-list files
  2) Falls back to IEDB-derived proxy from data/external/predig_train_modf.csv
     (documented conservative proxy — both tools train on IEDB-family data)

Usage:
    python -m src.fetch_prime_train_peptides \\
        [--wsl-prime-root ~/tools/sestrav_external/PRIME2.1] \\
        [--output data/external/prime_train_peptides.csv]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime, timezone

import pandas as pd

DEFAULT_PROXY = "data/external/predig_train_modf.csv"
DEFAULT_OUTPUT = "data/external/prime_train_peptides.csv"


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    low = name.lower()
    for c in df.columns:
        if c.lower() == low:
            return c
    return None


def from_proxy(proxy_path: str) -> pd.DataFrame:
    train = pd.read_csv(proxy_path)
    col = _find_col(train, "Epitope") or _find_col(train, "epitope") or train.columns[0]
    peps = sorted(set(train[col].astype(str).str.upper()))
    return pd.DataFrame(
        {
            "epitope": peps,
            "source": "proxy_iedb_family_predig_train_modf",
            "note": "PRIME published train list not in public PRIME2.1 zip; proxy from PredIG-Path training export (IEDB-family).",
        }
    )


def is_safe_wsl_path(path: str) -> bool:
    """Validate that the path only contains safe alphanumeric or path characters."""
    return bool(re.match(r"^[a-zA-Z0-9_\-\.\/\\~: ]+$", path))


def search_wsl_prime_root(wsl_root: str) -> pd.DataFrame | None:
    """Return peptides if a train-list file is found under PRIME install."""
    if not is_safe_wsl_path(wsl_root):
        print(f"[prime-train] Unsafe wsl_root path: {wsl_root}", file=sys.stderr)
        return None

    wsl_bin = shutil.which("wsl")
    if not wsl_bin:
        if os.path.exists(r"C:\Windows\System32\wsl.exe"):
            wsl_bin = r"C:\Windows\System32\wsl.exe"
        else:
            wsl_bin = "wsl"

    try:
        # Check command and run wsl find
        out = subprocess.check_output(  # nosec B603
            [
                wsl_bin,  # nosec B607
                "find",
                wsl_root,
                "-maxdepth",
                "4",
                "-type",
                "f",
                "(",
                "-iname",
                "*train*pep*",
                "-o",
                "-iname",
                "*immunogenic*pep*",
                "-o",
                "-iname",
                "*training*data*",
                ")",
                "-not",
                "-path",
                "*/PerRank/*",
            ],
            text=True,
            timeout=60,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    paths = [p.strip() for p in out.splitlines() if p.strip()][:5]
    for rel in paths:
        if not is_safe_wsl_path(rel):
            continue
        try:
            cat = subprocess.check_output([wsl_bin, "head", "-3", rel], text=True)  # nosec B603
        except subprocess.SubprocessError:
            continue
        if not cat or ("\t" not in cat and "," not in cat):
            continue
        try:
            full = subprocess.check_output([wsl_bin, "cat", rel], text=True)  # nosec B603
            from io import StringIO

            df = pd.read_csv(StringIO(full), sep=None, engine="python")
            col = _find_col(df, "epitope") or _find_col(df, "peptide") or df.columns[0]
            peps = sorted(set(df[col].astype(str).str.upper()))
            return pd.DataFrame(
                {"epitope": peps, "source": f"prime_install:{rel}", "note": "Extracted from local PRIME install"}
            )
        except (pd.errors.ParserError, ValueError, KeyError, FileNotFoundError, IndexError):
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch/build PRIME training peptide list")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--proxy", default=DEFAULT_PROXY)
    parser.add_argument(
        "--wsl-prime-root",
        default="~/tools/sestrav_external/PRIME2.1",
        help="WSL path to PRIME2.1 install",
    )
    args = parser.parse_args()

    out_df = search_wsl_prime_root(args.wsl_prime_root)
    if out_df is None:
        if not os.path.isfile(args.proxy):
            print(f"[prime-train] Missing proxy {args.proxy}", file=sys.stderr)
            sys.exit(1)
        out_df = from_proxy(args.proxy)
        print(
            "[prime-train] Using IEDB-family proxy from predig_train_modf.csv "
            "(PRIME train list not found in public install)",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    meta_path = args.output.replace(".csv", "_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"generated_utc={datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"n_peptides={len(out_df)}\n")
        f.write(f"source={out_df['source'].iloc[0]}\n")
        f.write(f"note={out_df['note'].iloc[0]}\n")

    print(f"[prime-train] Wrote {args.output} ({len(out_df)} peptides)")


if __name__ == "__main__":
    main()
