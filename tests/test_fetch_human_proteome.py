"""Tests for scripts/fetch_human_proteome.py.

Network-dependent download is not tested here; only pure-logic functions
(_sha256, _count_sequences, fetch skip-if-present, CLI parsing) are covered.
"""

import hashlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from fetch_human_proteome import _count_sequences, _sha256, fetch, main  # noqa: E402

# ---------------------------------------------------------------------------
# _count_sequences
# ---------------------------------------------------------------------------


def test_count_sequences_single(tmp_path):
    fa = tmp_path / "test.fasta"
    fa.write_text(">sp|P00001|TEST_HUMAN TestProt\nACDEFGHIKL\n")
    assert _count_sequences(fa) == 1


def test_count_sequences_multiple(tmp_path):
    fa = tmp_path / "multi.fasta"
    fa.write_text(">s1\nAAAA\n>s2\nCCCC\n>s3\nGGGG\n")
    assert _count_sequences(fa) == 3


def test_count_sequences_multiline(tmp_path):
    fa = tmp_path / "multiline.fasta"
    fa.write_text(">s1\nAAAA\nCCCC\n>s2\nGGGG\n")
    assert _count_sequences(fa) == 2


def test_count_sequences_empty(tmp_path):
    fa = tmp_path / "empty.fasta"
    fa.write_text("")
    assert _count_sequences(fa) == 0


def test_count_sequences_no_header(tmp_path):
    fa = tmp_path / "nohdr.fasta"
    fa.write_text("ACDEFGHIKL\n")
    assert _count_sequences(fa) == 0


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------


def test_sha256_known_content(tmp_path):
    f = tmp_path / "known.txt"
    content = b"hello world"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256(f) == expected


def test_sha256_empty_file(tmp_path):
    f = tmp_path / "empty.bin"
    f.write_bytes(b"")
    assert _sha256(f) == hashlib.sha256(b"").hexdigest()


def test_sha256_reproducible(tmp_path):
    f = tmp_path / "rep.bin"
    data = b"ACDEFGHIKLMNPQRSTVWY" * 1000
    f.write_bytes(data)
    assert _sha256(f) == _sha256(f)


# ---------------------------------------------------------------------------
# fetch - skip-if-present (no network)
# ---------------------------------------------------------------------------


def test_fetch_skips_when_file_exists(tmp_path, capsys):
    dest = tmp_path / "human.fasta"
    dest.write_text(">sp|P00001|TEST_HUMAN\nACDE\n")
    rc = fetch(output=str(dest), force=False)
    assert rc == 0
    captured = capsys.readouterr()
    assert "Already present" in captured.out


def test_fetch_reports_sequence_count_on_skip(tmp_path, capsys):
    dest = tmp_path / "human.fasta"
    dest.write_text(">s1\nAAAA\n>s2\nCCCC\n")
    fetch(output=str(dest), force=False)
    captured = capsys.readouterr()
    assert "2 sequences" in captured.out


# ---------------------------------------------------------------------------
# main() - CLI argument parsing (no-network paths only)
# ---------------------------------------------------------------------------


def test_main_skips_existing_file(tmp_path, capsys):
    dest = tmp_path / "proteome.fasta"
    dest.write_text(">s1\nACDE\n")
    rc = main([f"--output={dest}"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Already present" in captured.out


def test_main_force_flag_skips_check(tmp_path, monkeypatch):
    dest = tmp_path / "proteome.fasta"
    dest.write_text(">s1\nACDE\n")
    calls = []

    def fake_download(url, dest_path, chunk_size=None):
        calls.append(url)
        dest_path.write_text(">s1\nACDE\n")

    monkeypatch.setattr("fetch_human_proteome._download", fake_download)
    rc = main([f"--output={dest}", "--force"])
    assert rc == 0
    assert len(calls) == 1
