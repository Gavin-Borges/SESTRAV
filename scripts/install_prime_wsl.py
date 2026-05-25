#!/usr/bin/env python3
"""Install PRIME 2.1 and MixMHCpred in WSL/Linux for SESTRAV external validation."""
from __future__ import annotations

import io
import os
import subprocess  # nosec B404
import sys
import urllib.request
import zipfile
import hashlib
from pathlib import Path

INSTALL_DIR = Path.home() / "tools" / "sestrav_external"
PRIME_URL = "https://github.com/GfellerLab/PRIME/archive/refs/heads/master.zip"
MIX_URL = "https://github.com/GfellerLab/MixMHCpred/archive/refs/heads/master.zip"
PRIME_SHA_ENV = "SESTRAV_PRIME_ZIP_SHA256"
MIX_SHA_ENV = "SESTRAV_MIXMHCPRED_ZIP_SHA256"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_expected_hash(data: bytes, expected_sha256: str | None, label: str) -> None:
    if not expected_sha256:
        raise RuntimeError(f"[install-prime] Missing required SHA256 for {label}")
    actual = _sha256_bytes(data)
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"[install-prime] SHA256 mismatch for {label}: expected {expected_sha256}, got {actual}"
        )


def _safe_extract_zip(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    dest_root = dest_dir.resolve()
    for member in zf.infolist():
        target = (dest_dir / member.filename).resolve()
        if dest_root not in target.parents and target != dest_root:
            raise RuntimeError(f"[install-prime] Unsafe zip member path: {member.filename}")
    zf.extractall(dest_dir)


def download_zip(url: str, dest_dir: Path, rename_to: str, expected_sha256: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / rename_to
    if target.exists():
        print(f"[install-prime] Already present: {target}")
        return target
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"URL must start with http:// or https://: {url}")
    print(f"[install-prime] Downloading {url}")
    data = urllib.request.urlopen(url, timeout=120).read()  # nosec B310 # nosemgrep
    _verify_expected_hash(data, expected_sha256, rename_to)
    tmp_dir = dest_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        _safe_extract_zip(zf, tmp_dir)
    extracted = next(tmp_dir.iterdir())
    extracted.rename(target)
    tmp_dir.rmdir()
    print(f"[install-prime] Extracted to {target}")
    return target


def compile_prime(prime_dir: Path) -> Path:
    cc = prime_dir / "lib" / "PRIME.cc"
    out = prime_dir / "lib" / "PRIME"
    prime_x = prime_dir / "lib" / "PRIME.x"
    if not out.exists() or not prime_x.exists():
        if not cc.exists():
            raise FileNotFoundError(f"Missing {cc}")
        print("[install-prime] Compiling PRIME.cc")
        subprocess.run(  # nosec B603 B607
            ["g++", "-O3", str(cc), "-o", str(out)],
            check=True,
        )
    # run_PRIME.pl invokes PRIME.x on Linux; replace with compiled binary.
    import shutil

    shutil.copy2(out, prime_x)
    prime_x.chmod(0o755)
    out.chmod(0o755)
    (prime_dir / "MixMHCpred").chmod(0o755) if (prime_dir / "MixMHCpred").exists() else None
    mix_script = INSTALL_DIR / "MixMHCpred" / "MixMHCpred"
    if mix_script.exists():
        mix_script.chmod(0o755)
    return out


def main() -> None:
    prime_sha = os.environ.get(PRIME_SHA_ENV, "da5e65fd4b142857f42167b15495600b3a6416641c73854a8a2cbc00d67ee6a8")
    mix_sha = os.environ.get(MIX_SHA_ENV, "9b3d96813368df42622ce13a96eed09b447133699d8eaf5a4793da8561981c9b")
    prime_dir = download_zip(PRIME_URL, INSTALL_DIR, "PRIME2.1", expected_sha256=prime_sha)
    download_zip(MIX_URL, INSTALL_DIR, "MixMHCpred", expected_sha256=mix_sha)
    prime_bin = compile_prime(prime_dir)
    path_line = (
        f'export PATH="{prime_dir}:{INSTALL_DIR / "MixMHCpred"}:$PATH"'
    )
    bashrc = Path.home() / ".bashrc"
    text = bashrc.read_text(encoding="utf-8") if bashrc.exists() else ""
    if "sestrav_external" not in text:
        bashrc.write_text(text + "\n" + path_line + "\n", encoding="utf-8")
    print(f"[install-prime] PRIME binary: {prime_bin}")
    print(path_line)



if __name__ == "__main__":
    main()
