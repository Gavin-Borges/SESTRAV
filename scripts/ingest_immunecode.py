"""Ingest ImmuneCODE MIRA SARS-CoV-2 data and extract non-immunogenic peptide
negatives for the SESTRAV immunogenicity prediction pipeline.

ImmuneCODE MIRA (Adaptive Biotechnologies) maps SARS-CoV-2 peptides to TCR
beta sequences from COVID-19 patient blood samples. Peptide-allele pairs that
appear in the assay data but accumulate zero TCR template matches are treated
as experimentally confirmed negatives for MHC Class I immunogenicity.

Negative definition (MIRA-specific)
-------------------------------------
A (peptide, HLA allele) pair is labelled 0 (non-immunogenic) when the raw
peptide entry (which may be a semicolon-separated pool) appears in the assay
data with zero TCR template rows - meaning no clonal T-cell expansion was
observed. Individual peptides within a matched pool are excluded because it
is ambiguous which pool member drove the response.

Source: Nolan S, et al. A large-scale database of T-cell receptor beta
  sequences and binding associations from natural and synthetic exposure to
  SARS-CoV-2. Research Square (2020). PMID 32665443.
Download (release 002.1, as linked from the publisher's listing page
https://clients.adaptivebiotech.com/pub/covid-2020 on 2026-10-12):
  https://adaptivepublic.blob.core.windows.net/publishedproject-supplements/covid-2020/ImmuneCODE-MIRA-Release002.1.zip

Release 002.1 is refused
------------------------
That archive downloads and matches the pin, and is then refused, because it
cannot supply what this script writes: each row of its peptide-detail tables is
one TCR binding, with its experiment and its peptides but no HLA allele, and HLA
appears only as each subject's genotype in subject-metadata.csv. A genotype says
which alleles a subject carries, not which one presented a peptide, so pairing
them would invent data. See _refuse_a_table_without_restriction.

Usage
-----
  Auto-download (requires network access; release 002.1 is refused, see above):
    python scripts/ingest_immunecode.py

  A local archive, hashed against the pin and refused if laid out like 002.1:
    python scripts/ingest_immunecode.py --input path/to/archive.zip

  A local CSV with a peptide column and an HLA restriction column per row:
    python scripts/ingest_immunecode.py --input path/to/peptide-detail.csv

  Dry run - print stats, no file write:
    python scripts/ingest_immunecode.py --dry-run

  Custom output path:
    python scripts/ingest_immunecode.py --output data/my_negatives.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import random
import re
import sys
import tempfile
import urllib.request
import zipfile
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls

# sha256_file is src.artifact_integrity.sha256_file, which reads to end of file.
from _dataset_utils import sha256_file, write_provenance  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Release 002.1 as the publisher's listing page, MIRA_LISTING_URL, linked it on
# 2026-10-12. The URL this script held before, on clients.adaptivebiotech.com,
# answered HTTP 404 to both HEAD and GET that day.
MIRA_DOWNLOAD_URL = (
    "https://adaptivepublic.blob.core.windows.net/publishedproject-supplements/"
    "covid-2020/ImmuneCODE-MIRA-Release002.1.zip"
)
MIRA_LISTING_URL = "https://clients.adaptivebiotech.com/pub/covid-2020"

# The start URL and every redirect target are checked before urllib opens them:
# each must be https on one of these netlocs exactly (no port, no userinfo, no
# look-alike host), so a redirect anywhere else fails closed. The first is the
# host the listing page links the archive to; the second hosts the listing page
# and the URL this script held before. The structural check (it opens as a ZIP
# and every member passes its CRC-32) catches a transfer cut short or damaged in
# transit; the sha256 pin below is what rules out a SUBSTITUTED archive.
MIRA_DOWNLOAD_HOSTS = frozenset(
    {"adaptivepublic.blob.core.windows.net", "clients.adaptivebiotech.com"}
)

# ImmuneCODE MIRA release 002.1: the archive the auto-download must match, and the
# one an --input archive is compared with; the sidecar names this release only for
# an input that matched. Measured 2026-10-12 by streaming MIRA_DOWNLOAD_URL and
# hashing the bytes. The
# MD5 of those bytes equalled the Content-MD5 header the same server sent, which
# shows the transfer arrived as served; coming from the same host, it says
# nothing about authenticity. Every member also passed its CRC-32. If the bytes
# served there ever change, this pin turns that into a refusal.
MIRA_PINNED_SHA256 = "ad0b0476f6e2c07aaec9e38cdb7f79e16c2b14e01f2ea706a2b2d86cb4667d9f"
MIRA_PINNED_SIZE_BYTES = 3856733
MIRA_PINNED_RELEASE = "002.1"

# Seconds the opener may wait on the connect and on each socket read. It bounds
# a stalled server, not the total transfer time of the archive.
DOWNLOAD_TIMEOUT_SECONDS = 60

# Preferred CSV filenames inside the ZIP archive, tried in order. Release 002.1
# carries peptide-detail-ci.csv and peptide-detail-cii.csv and no
# peptide-detail-cov2.csv, so the second name is the one it matches.
_CANDIDATE_CSVS: list[str] = [
    "peptide-detail-cov2.csv",
    "peptide-detail-ci.csv",
]

# The columns _extract_negatives reads a row's peptide and its restricting HLA
# allele from, each tried in order.
_PEPTIDE_COLUMNS: list[str] = ["Peptide", "peptide", "Antigen Sequence", "antigen_sequence"]
_ALLELE_COLUMNS: list[str] = ["HLA Restrictions", "HLA_Restrictions", "MHC", "hla", "Allele"]

# NCBI taxon ID for SARS-CoV-2 (Wuhan-Hu-1 reference strain).
_VIRUS_TAXON_ID: int = 2697049

_REFERENCE_PMID: str = "32665443"

# Standard 20 amino acids (MHC Class I canonical set).
_VALID_AA: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

# MHC Class I canonical peptide length bounds.
_MIN_LEN: int = 8
_MAX_LEN: int = 11

# Exactly 23 schema columns required by SESTRAV negatives pipeline.
SCHEMA_COLUMNS: list[str] = [
    "peptide",
    "label",
    "hla_allele",
    "virus",
    "protein",
    "strain",
    "source_type",
    "database_source",
    "tcr_alpha_cdr3",
    "tcr_beta_cdr3",
    "virus_family",
    "negative_origin",
    "assay_type",
    "assay_quality_tier",
    "assay_quality_weight",
    "reference_pmid",
    "iedb_assay_id",
    "infection_phase",
    "antigen_latency_program",
    "assay_context",
    "cross_reactivity_tested",
    "virus_taxon_id",
    "is_quarantined",
]

# Fixed values applied to every output row.
_FIXED_FIELDS: dict[str, object] = {
    "label": 0,
    "virus": "SARS-CoV-2",
    "protein": np.nan,
    "strain": "SARS-CoV-2 (Wuhan-Hu-1)",
    "source_type": "Virus",
    "database_source": "ImmuneCODE",
    "tcr_alpha_cdr3": np.nan,
    "tcr_beta_cdr3": np.nan,
    "virus_family": "Coronaviridae",
    "negative_origin": "immunecode_mira",
    "assay_type": "MIRA_TCR_sequencing",
    "assay_quality_tier": 1,
    "assay_quality_weight": 0.9,
    "reference_pmid": _REFERENCE_PMID,
    "iedb_assay_id": np.nan,
    "infection_phase": np.nan,
    "antigen_latency_program": np.nan,
    "assay_context": np.nan,
    "cross_reactivity_tested": np.nan,
    "virus_taxon_id": _VIRUS_TAXON_ID,
    "is_quarantined": False,
}

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZIP utilities
# ---------------------------------------------------------------------------


def _safe_extract_zip(zip_ref: zipfile.ZipFile, dest_dir: str) -> None:
    """Extract a ZIP archive, rejecting path-traversal members (zip-slip guard)."""
    dest_root = os.path.realpath(dest_dir)
    for member in zip_ref.infolist():
        target = os.path.realpath(os.path.join(dest_dir, member.filename))
        if target != dest_root and not target.startswith(dest_root + os.sep):
            raise RuntimeError(f"Unsafe zip member path: {member.filename}")
    zip_ref.extractall(dest_dir)


def _find_csv_in_dir(search_dir: str) -> str | None:
    """Return the first MIRA peptide-detail CSV path found under search_dir."""
    for candidate in _CANDIDATE_CSVS:
        for root, _dirs, files in os.walk(search_dir):
            if candidate in files:
                return os.path.join(root, candidate)
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _check_download_url(url: str) -> None:
    """Refuse a URL that is not https on one of MIRA_DOWNLOAD_HOSTS exactly."""
    parts = urlsplit(url)
    # urlsplit drops tab, CR and LF, but urllib.request keeps them, so a URL
    # with any control, space or non-ASCII character is refused outright.
    if not all(32 < ord(ch) < 127 for ch in url):
        reason = "it holds a control, space or non-ASCII character"
    elif parts.scheme != "https" or parts.netloc not in MIRA_DOWNLOAD_HOSTS:
        reason = f"not https on {' or '.join(sorted(MIRA_DOWNLOAD_HOSTS))}"
    else:
        return
    shown = url.split("?", 1)[0].split("#", 1)[0]  # a query may hold a signed token
    raise RuntimeError(f"Refusing ImmuneCODE download URL {shown!r}: {reason}")


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check each redirect target with _check_download_url before urllib requests it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_archive(path: str) -> tuple[bytes, bool]:
    """Read the archive at path once; return its bytes and whether they are the pinned archive.

    The verdict and the extraction both come from these same bytes. Hashing the
    file by path and then reopening it to extract read it twice, so a file
    replaced in between was extracted while the provenance sidecar recorded the
    earlier bytes as verified against the pin.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    matches = (
        len(data) == MIRA_PINNED_SIZE_BYTES
        and hashlib.sha256(data).hexdigest() == MIRA_PINNED_SHA256
    )
    return data, matches


def _zip_is_intact(path: str) -> bool:
    """True when path opens as a ZIP archive and every member passes its CRC-32."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception:  # any failure to open or read it means it cannot be used
        return False


def _download_mira(dest_dir: str) -> str:
    """Download the MIRA ZIP from Adaptive Biotechnologies and verify it against the pin.

    The bytes stream to a .part file, bounded at MIRA_PINNED_SIZE_BYTES so an
    endless body cannot fill the disk, and that file takes the archive's name only
    once it opens as an undamaged ZIP AND its size and sha256 equal the pinned
    values. Any failure, KeyboardInterrupt included, removes it. That removal is
    best effort: if it raises (a Windows sharing violation, for example), the
    error is dropped, so it never replaces the failure itself.
    Returns the local path of the downloaded file.
    Raises RuntimeError with an actionable message if the download fails.
    """
    os.makedirs(dest_dir, exist_ok=True)
    zip_name = MIRA_DOWNLOAD_URL.rsplit("/", 1)[1]
    zip_path = os.path.join(dest_dir, zip_name)
    partial = zip_path + ".part"
    log.info("Downloading ImmuneCODE MIRA from %s", MIRA_DOWNLOAD_URL)
    opener = urllib.request.build_opener(_CheckedRedirectHandler)
    try:
        _check_download_url(MIRA_DOWNLOAD_URL)
        with (
            opener.open(MIRA_DOWNLOAD_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            open(partial, "wb") as output,
        ):
            written = 0
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MIRA_PINNED_SIZE_BYTES:
                    raise RuntimeError(
                        f"the transfer exceeds the pinned {MIRA_PINNED_SIZE_BYTES} bytes"
                    )
                output.write(chunk)
        if not _zip_is_intact(partial):
            raise RuntimeError("the downloaded file is not a complete, undamaged ZIP archive")
        # Structural integrity says the bytes arrived whole; only the digest says
        # they are the release this script attributes its rows to.
        if written != MIRA_PINNED_SIZE_BYTES:
            raise RuntimeError(
                f"the downloaded archive is {written} bytes, not the pinned "
                f"{MIRA_PINNED_SIZE_BYTES}"
            )
        actual = sha256_file(partial)
        if actual != MIRA_PINNED_SHA256:
            raise RuntimeError(
                "the downloaded archive does not match the pinned release 002.1: "
                f"expected sha256 {MIRA_PINNED_SHA256}, got {actual}"
            )
        os.replace(partial, zip_path)
    except Exception as exc:
        raise RuntimeError(
            f"Download failed: {exc}\n"
            "The pinned archive is published at:\n"
            f"  {MIRA_DOWNLOAD_URL}\n"
            f"as linked from {MIRA_LISTING_URL}. A copy downloaded by hand can be\n"
            "passed with --input PATH and is checked against the same pin, but release\n"
            f"{MIRA_PINNED_RELEASE} is refused by either route: it records HLA only as\n"
            "subject genotypes, not as the allele restricting each peptide."
        ) from exc
    finally:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass  # best effort: a locked .part must not replace the download's own outcome
    return zip_path


# A ZIP local file header. Every archive starts with one, including a truncated one
# that has lost the end-of-central-directory record `zipfile.is_zipfile` looks for.
_ZIP_LOCAL_FILE_HEADER = b"PK\x03\x04"


class DamagedArchive(zipfile.BadZipFile):
    """An --input that begins as a ZIP archive but cannot be opened as one."""


def _refuse_a_damaged_archive(input_path: str) -> None:
    """Name a damaged archive as one, rather than letting it reach the CSV reader.

    `zipfile.is_zipfile` reads the end-of-central-directory record, so a truncated
    archive answers False and used to fall through to `pandas.read_csv`. What the
    reader then reported depended on the FILENAME, not on the bytes: pandas infers
    zip compression from a `.zip` extension and raised `BadZipFile`, while the same
    bytes under any other name raised `UnicodeDecodeError` on a compressed byte,
    which says nothing about the real problem. This makes the verdict depend on the
    bytes instead.
    """
    with open(input_path, "rb") as handle:
        signature = handle.read(len(_ZIP_LOCAL_FILE_HEADER))
    if signature == _ZIP_LOCAL_FILE_HEADER:
        raise DamagedArchive(
            f"{input_path} starts as a ZIP archive but cannot be opened as one: it is "
            "truncated or damaged. Download the ImmuneCODE MIRA archive again, or pass "
            "the extracted peptide-detail CSV instead."
        )


def _load_from_zip(zip_path: str) -> tuple[pd.DataFrame, str, bool]:
    """Extract a MIRA ZIP, locate the peptide-detail CSV, and load it.

    Returns (DataFrame, zip_path_for_provenance, matches_pin), where matches_pin
    says whether the archive's bytes are the pinned archive (see _read_archive).
    The rows are read from the matching member of those same bytes, not from the
    extracted copy, which anything able to write the temporary directory could
    replace between extraction and the read.
    """
    log.info("Extracting ZIP: %s", zip_path)
    data, matches_pin = _read_archive(zip_path)
    # ignore_cleanup_errors: a handle still open on an extracted member makes the
    # cleanup raise on Windows, and that error would replace whatever this function
    # was already failing with - a missing peptide-detail CSV, say - on the way out.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
            _safe_extract_zip(zf, tmp)
            csv_path = _find_csv_in_dir(tmp)
            if csv_path is None:
                raise FileNotFoundError(
                    f"No MIRA peptide-detail CSV found in '{zip_path}'. "
                    f"Expected one of: {_CANDIDATE_CSVS}"
                )
            member = _member_extracted_to(zf, tmp, csv_path)
            log.info("Reading CSV from archive: %s", os.path.basename(csv_path))
            with zf.open(member) as handle:
                df = pd.read_csv(handle, low_memory=False)
    return df, zip_path, matches_pin


def _member_extracted_to(zf: zipfile.ZipFile, dest_dir: str, path: str) -> str:
    """Name the archive member that extraction into dest_dir wrote to path."""
    target = os.path.realpath(path)
    for info in zf.infolist():
        if not info.is_dir() and os.path.realpath(os.path.join(dest_dir, info.filename)) == target:
            return info.filename
    raise RuntimeError(
        f"No member of the archive maps to the extracted {os.path.basename(path)}; "
        "refusing to read the extracted copy instead."
    )


def _load_csv(input_path: str | None) -> tuple[pd.DataFrame, str, bool]:
    """Load a MIRA peptide-detail CSV from a ZIP, a CSV, or by auto-download.

    Dispatch order:
      1. input_path is None -> attempt download from MIRA_DOWNLOAD_URL.
      2. input_path is a ZIP (detected by zipfile.is_zipfile) -> extract then read.
      3. input_path is a CSV -> read directly.

    Returns (DataFrame, source_description, verified). source_description is
    used only for the provenance sidecar: MIRA_DOWNLOAD_URL for the auto-download,
    whose archive is deleted with its temporary directory, else input_path.
    verified says whether the rows came from bytes that are the pinned archive,
    judged on the bytes that were extracted; a CSV is never verified, because
    nothing ties its rows to the archive.
    """
    if input_path is None:
        log.info("No --input given; attempting auto-download.")
        # ignore_cleanup_errors: _download_mira removes its .part file only best
        # effort, because on Windows a second handle on it makes the removal raise.
        # A file left behind that way makes this directory's cleanup raise too, and
        # that PermissionError would escape in place of the download's own message.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            zip_path = _download_mira(tmp)
            df, _zip_path, matches_pin = _load_from_zip(zip_path)
            # _download_mira verified the .part file before renaming it, so this
            # fails only if the archive changed after that check.
            if not matches_pin:
                raise RuntimeError(
                    "The downloaded MIRA archive no longer matches the pinned sha256 "
                    "it was verified against; refusing to use it."
                )
            return df, MIRA_DOWNLOAD_URL, True

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"--input path not found: {input_path}")

    if zipfile.is_zipfile(input_path):
        df, source, matches_pin = _load_from_zip(input_path)
        log.info(
            "--input %s the pinned release %s archive.",
            "matches" if matches_pin else "is not",
            MIRA_PINNED_RELEASE,
        )
        return df, source, matches_pin

    _refuse_a_damaged_archive(input_path)

    # Direct CSV path.
    log.info("Loading CSV: %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    return df, input_path, False


# ---------------------------------------------------------------------------
# Allele normalization
# ---------------------------------------------------------------------------


def _normalize_allele(raw: str) -> str | None:
    """Normalize an HLA allele string to HLA-X*GG:PP (Class I only).

    Handles:
      HLA-A*02:01  -> HLA-A*02:01  (already canonical)
      HLA-A*0201   -> HLA-A*02:01  (8-digit, no colon)
      A*02:01      -> HLA-A*02:01  (missing HLA prefix)
      A*0201       -> HLA-A*02:01  (missing prefix, 8-digit)

    Returns None for Class II alleles (HLA-D*) or unrecognized formats.
    Only HLA-A, HLA-B, and HLA-C (Class I) are accepted.
    """
    if not isinstance(raw, str):
        return None
    allele = raw.strip()
    # Already canonical HLA-[ABC]*DD:DD
    if re.match(r"^HLA-[ABC]\*\d{2}:\d{2}$", allele):
        return allele
    # 8-digit no colon: HLA-A*0201 -> HLA-A*02:01
    m = re.match(r"^HLA-([ABC])\*(\d{2})(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Missing prefix, canonical digits: A*02:01 -> HLA-A*02:01
    m = re.match(r"^([ABC])\*(\d{2}):(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Missing prefix, 8-digit: A*0201 -> HLA-A*02:01
    m = re.match(r"^([ABC])\*(\d{2})(\d{2})$", allele)
    if m:
        return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}"
    # Cannot normalize or not Class I.
    return None


# ---------------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------------


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in df.columns, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _require_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    """Return the first matching column, or raise ValueError with details."""
    col = _detect_column(df, candidates)
    if col is None:
        raise ValueError(
            f"Cannot find '{role}' column in DataFrame. "
            f"Tried: {candidates}. "
            f"Available columns: {df.columns.tolist()}"
        )
    return col


class UnsupportedLayout(RuntimeError):
    """An input with no per-row peptide or HLA restriction, which cannot supply pairs."""


def _refuse_a_table_without_restriction(df: pd.DataFrame, source: str) -> None:
    """Refuse a table that cannot supply (peptide, HLA allele) pairs, and say why.

    Release 002.1 is such a table. Read 2026-10-12 from the pinned archive: it has
    no peptide-detail-cov2.csv; its peptide-detail-ci.csv has the columns TCR
    BioIdentity, TCR Nucleotide Sequence, Experiment, ORF Coverage, Amino Acids,
    Start Index in Genome and End Index in Genome, so no peptide column this script
    reads and no HLA column at all; and its readme says subject-metadata.csv holds
    "HLA when available" and is joined to the detail files by Experiment. That file
    has two columns each for HLA-A, HLA-B and HLA-C per subject: a genotype, not
    the allele that presented a given peptide. No pair is invented from it.
    """
    missing = [
        f"{role} column (tried {', '.join(names)})"
        for role, names in (
            ("peptide", _PEPTIDE_COLUMNS),
            ("HLA restriction", _ALLELE_COLUMNS),
        )
        if _detect_column(df, names) is None
    ]
    if not missing:
        return
    name = source.replace("\\", "/").rsplit("/", 1)[-1]
    message = (
        f"{name} cannot supply the (peptide, HLA allele) pairs this script writes: it has "
        f"no {' and no '.join(missing)}. Refusing rather than inventing pairs."
    )
    if _detect_column(df, _ALLELE_COLUMNS) is None:
        message += (
            f" ImmuneCODE MIRA release {MIRA_PINNED_RELEASE} is laid out this way: each row"
            " of its peptide-detail tables is one TCR binding, with its experiment and its"
            " peptides ('Amino Acids') but no HLA allele, and HLA appears only as each"
            " subject's genotype in subject-metadata.csv (two columns each for HLA-A, HLA-B"
            " and HLA-C). A genotype says which alleles a subject carries, not which one"
            " presented a peptide. To ingest ImmuneCODE negatives, pass --input a table"
            " that records the restricting allele for each peptide."
        )
    raise UnsupportedLayout(message)


# ---------------------------------------------------------------------------
# Negative extraction
# ---------------------------------------------------------------------------


def _extract_negatives(df: pd.DataFrame) -> list[dict[str, object]]:
    """Extract non-immunogenic (peptide, hla_allele) pairs from MIRA data.

    Strategy
    --------
    1. Detect the peptide, allele, and TCR template columns by name.
    2. Mark rows where the TCR column is non-null and non-empty as hits.
    3. Group by (peptide_raw, allele_raw) and sum hit counts per group.
    4. Groups with sum = 0 indicate pools or peptides that were assayed but
       showed no T-cell expansion -> candidate negatives.
    5. Expand semicolon-separated peptide pools into individual sequences.
    6. Expand semicolon-separated allele fields and normalize each allele.
    7. Apply length (8-11 aa) and amino acid (standard 20 aa) filters.

    Pools that DO have TCR matches are intentionally excluded: it is ambiguous
    which pool member drove the response, so no individual peptide within a
    matched pool can be safely labelled as a negative.
    """
    pep_col = _require_column(df, _PEPTIDE_COLUMNS, "peptide")
    allele_col = _require_column(df, _ALLELE_COLUMNS, "HLA allele")
    tcr_col = _detect_column(
        df,
        ["Amino Acids", "amino_acids", "CDR3b", "cdr3b", "TCR Beta CDR3"],
    )

    log.info(
        "Columns detected - peptide: '%s', allele: '%s', TCR: '%s'",
        pep_col,
        allele_col,
        tcr_col if tcr_col else "(not found)",
    )

    if tcr_col is None:
        log.warning(
            "No TCR template column found (tried: Amino Acids, CDR3b, ...). "
            "Cannot identify zero-match rows. "
            "Provide a file with an 'Amino Acids' column that includes null "
            "entries for tested but non-expanding peptides."
        )
        return []

    # Boolean mask: True where a TCR template entry is present.
    has_tcr: pd.Series = df[tcr_col].notna() & (df[tcr_col].astype(str).str.strip() != "")
    df = df.copy()
    df["_has_tcr"] = has_tcr.astype(int)

    # Aggregate TCR hit count per (raw peptide, raw allele) pair.
    grp = df.groupby([pep_col, allele_col], sort=False)["_has_tcr"].sum().reset_index()

    n_total = len(grp)
    neg_mask = grp["_has_tcr"] == 0
    n_neg_pools = int(neg_mask.sum())
    n_pos_pools = n_total - n_neg_pools

    log.info(
        "Grouped %d unique (peptide, allele) entries: "
        "%d with TCR hits, %d with zero hits (candidate negatives).",
        n_total,
        n_pos_pools,
        n_neg_pools,
    )

    if n_neg_pools == 0:
        log.warning(
            "No zero-TCR-count (peptide, allele) pairs found. "
            "The input file may contain only enriched TCR-match rows. "
            "Consider providing the full MIRA catalog that includes "
            "zero-template-count entries for tested but non-expanding pools."
        )
        return []

    neg_pools = grp[neg_mask]

    # Expand each zero-count pool into individual (peptide, allele) records.
    records: list[dict[str, object]] = []
    skipped_len = 0
    skipped_aa = 0
    skipped_allele = 0

    for _, row in neg_pools.iterrows():
        raw_pep_str = str(row[pep_col])
        raw_allele_str = str(row[allele_col])

        peptides_raw = [p.strip() for p in raw_pep_str.split(";") if p.strip()]
        alleles_raw = [a.strip() for a in raw_allele_str.split(";") if a.strip()]

        for pep in peptides_raw:
            # Reject non-ASCII BEFORE upper-casing: str.upper() folds U+0131 to "I"
            # and U+00DF to "SS", so the filters below would pass on residues and
            # lengths the input never had.
            if not pep.isascii():
                skipped_aa += 1
                continue
            pep = pep.upper()
            if not (_MIN_LEN <= len(pep) <= _MAX_LEN):
                skipped_len += 1
                continue
            if not all(c in _VALID_AA for c in pep):
                skipped_aa += 1
                continue
            for allele_raw in alleles_raw:
                allele = _normalize_allele(allele_raw)
                if allele is None:
                    skipped_allele += 1
                    continue
                records.append({"peptide": pep, "hla_allele": allele})

    log.info(
        "Expanded to %d raw (peptide, allele) records. "
        "Skipped: %d wrong length, %d invalid AA, %d non-Class-I allele.",
        len(records),
        skipped_len,
        skipped_aa,
        skipped_allele,
    )
    return records


# ---------------------------------------------------------------------------
# Output construction
# ---------------------------------------------------------------------------


def _build_output_df(records: list[dict[str, object]]) -> pd.DataFrame:
    """Assemble the 23-column SESTRAV schema DataFrame from extracted records.

    Applies deduplication on (peptide, hla_allele) keeping the highest
    assay_quality_weight (all ImmuneCODE rows share the same weight, so this
    effectively removes exact duplicates that arise from pool expansion).
    """
    if not records:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame(records)

    # Apply fixed metadata fields.
    for col, val in _FIXED_FIELDS.items():
        df[col] = val

    # Dedup on (peptide, hla_allele), keep highest quality weight row.
    df = (
        df.sort_values("assay_quality_weight", ascending=False)
        .drop_duplicates(subset=["peptide", "hla_allele"])
        .reset_index(drop=True)
    )

    # Deterministic sort for byte-stable output across re-runs.
    df = df.sort_values(["peptide", "hla_allele"]).reset_index(drop=True)

    # Enforce column order and return exactly the 23 schema columns.
    return df[SCHEMA_COLUMNS]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

# The checkout holding this script. An --input inside it is recorded in the
# provenance sidecar relative to it.
_REPO_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _provenance_source(source: str) -> str:
    """Return how the provenance sidecar records a source: never as an absolute path.

    The auto-download is recorded by MIRA_DOWNLOAD_URL, a path inside the repo
    by its repo-relative POSIX path, and any other path by its basename alone,
    prefixed "external:", so no home, temporary or drive path is written.
    """
    if source == MIRA_DOWNLOAD_URL:
        return source
    # Both sides in one resolved form, so a symlinked or dotted spelling of
    # either cannot move a file in or out of the repo.
    path = os.path.realpath(source)
    root = os.path.realpath(_REPO_ROOT)
    try:
        inside = os.path.commonpath([root, path]) == root
    except ValueError:  # on Windows, a path on a different drive from the repo
        inside = False
    if inside:
        return os.path.relpath(path, root).replace(os.sep, "/")
    return "external:" + os.path.basename(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract SARS-CoV-2 non-immunogenic negatives from ImmuneCODE MIRA "
            "and produce a SESTRAV 23-column negatives CSV."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "Path to a MIRA ZIP archive or a peptide-detail CSV. Every row must carry "
            "a peptide and the HLA allele restricting it; release "
            f"{MIRA_PINNED_RELEASE} does not and is refused. If omitted, the script "
            f"downloads {MIRA_DOWNLOAD_URL}, checks it against the pin, and refuses "
            "it for the same reason."
        ),
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default="data/immunecode_sars_negatives.csv",
        help=("Output CSV path (default: data/immunecode_sars_negatives.csv)."),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics and a sample of output rows; do not write files.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ingest_immunecode."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    # Deterministic seeds for reproducibility.
    random.seed(42)
    np.random.seed(42)

    # Load raw data. A damaged archive is reported like every other refusal here:
    # DamagedArchive, or a BadZipFile from an archive whose member fails its CRC-32
    # while it is extracted, is logged and exits 1 rather than escaping as a traceback.
    # A table that cannot supply (peptide, HLA allele) pairs is refused the same way.
    try:
        df_raw, source_path, verified = _load_csv(args.input)
        _refuse_a_table_without_restriction(df_raw, source_path)
    except (FileNotFoundError, RuntimeError, zipfile.BadZipFile) as exc:
        log.error("%s", exc)
        return 1

    log.info("Loaded %d rows from source.", len(df_raw))

    # Extract zero-TCR-count negatives.
    records = _extract_negatives(df_raw)
    if not records:
        log.error(
            "No negative records extracted. "
            "Check the --input file and the log messages above for details."
        )
        return 1

    # Build the 23-column output DataFrame.
    df_out = _build_output_df(records)

    unique_peptides = int(df_out["peptide"].nunique())
    unique_alleles = int(df_out["hla_allele"].nunique())
    log.info(
        "Final output: %d unique (peptide, hla_allele) negative pairs "
        "(%d unique peptides, %d unique alleles).",
        len(df_out),
        unique_peptides,
        unique_alleles,
    )

    allele_counts = df_out["hla_allele"].value_counts().head(10)
    log.info("Allele distribution (top 10):\n%s", allele_counts.to_string())

    if args.dry_run:
        log.info("--dry-run: no files written.")
        print(df_out.head(20).to_string(index=False))
        return 0

    # Write output CSV.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    log.info("Wrote %d rows -> %s", len(df_out), args.output)

    # Write provenance sidecar.
    write_provenance(
        args.output,
        sources=[_provenance_source(source_path)],
        row_count=len(df_out),
        extra={
            "database": "ImmuneCODE MIRA",
            # Named only when this run verified the input's bytes against the pin;
            # otherwise nothing in the run established which release it came from.
            "release": MIRA_PINNED_RELEASE if verified else None,
            "pinned_archive_sha256": MIRA_PINNED_SHA256 if verified else None,
            "archive_verified_against_pin": verified,
            "reference_pmid": _REFERENCE_PMID,
            "label": 0,
            "virus": "SARS-CoV-2",
            "negative_origin": "immunecode_mira",
            "unique_peptides": unique_peptides,
            "unique_alleles": unique_alleles,
            "allele_distribution": allele_counts.to_dict(),
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
