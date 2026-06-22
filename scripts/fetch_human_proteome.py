"""Download the UniProt human reference proteome (UP000005640) for hard-decoy generation.

The file is ~100 MB. Running this script once is sufficient; subsequent calls
skip the download if the FASTA already exists at the target path.

Usage
-----
    python scripts/fetch_human_proteome.py
    python scripts/fetch_human_proteome.py --output data/proteomes/human.fasta

References
----------
UniProt Consortium. UniProt: the Universal Protein Database.
  Nucleic Acids Res. 2023;51(D1):D523-D531.
  DOI: 10.1093/nar/gkac1052
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls

# UniProt REST API endpoint for reference proteome download
# UP000005640 = Homo sapiens (canonical + isoforms reviewed, Swiss-Prot)
_UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/stream"
    "?compressed=false"
    "&format=fasta"
    "&query=%28proteome%3AUP000005640%29+AND+%28reviewed%3Atrue%29"
)

DEFAULT_OUTPUT = "data/proteomes/human_uniprot_UP000005640.fasta"


def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download url to dest, printing progress."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".fasta.tmp")
    downloaded = 0
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:  # nosec B310  # noqa: S310
        while chunk := resp.read(chunk_size):
            out.write(chunk)
            downloaded += len(chunk)
            print(f"\r  {downloaded / 1e6:.1f} MB downloaded...", end="", flush=True)
    print()
    tmp.rename(dest)
    print(f"Saved: {dest} ({downloaded / 1e6:.1f} MB)")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _count_sequences(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.startswith(">"))


def fetch(output: str | Path = DEFAULT_OUTPUT, force: bool = False) -> int:
    dest = Path(output)
    if dest.exists() and not force:
        n_seqs = _count_sequences(dest)
        print(f"Already present: {dest} ({n_seqs} sequences). Use --force to re-download.")
        return 0

    _download(_UNIPROT_URL, dest)

    n_seqs = _count_sequences(dest)
    sha = _sha256(dest)
    print("\nVerification:")
    print(f"  Sequences : {n_seqs:,}")
    print(f"  SHA256    : {sha}")
    print("\nNext step:")
    print(f"  python scripts/generate_hard_decoys.py --fasta {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download UniProt human reference proteome (UP000005640) for hard-decoy generation.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Destination FASTA path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists",
    )
    args = parser.parse_args(argv)
    return fetch(args.output, args.force)


if __name__ == "__main__":
    sys.exit(main())
