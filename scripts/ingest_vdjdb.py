import argparse
import os
import re
import sys
import urllib.request
import zipfile
from urllib.parse import unquote, urlsplit

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ssl_fix  # noqa: F401, E402 - patch SSL before any network calls
from _dataset_utils import (
    normalize_peptides,
    sha256_file,
    validate_against_schema,
    write_provenance,
)

# The ONE VDJdb release this pipeline ingests. Resolving "latest" instead meant
# that when upstream published a new release the URL, the digest and the size all
# moved together, so the corpus changed with no error and the sidecar recorded
# only that a fetch had happened. Pinning the tag, the asset name, the archive's
# sha256 and its byte count makes an upstream change a loud failure instead.
#
# Read 2026-10-12 from the release API for this tag:
#   GET https://api.github.com/repos/antigenomics/vdjdb-db/releases/tags/2026-06-03-ZENODO
# which reports digest "sha256:9ae7724a..." and size 41017409 for the asset below.
# To move the pin, re-run that request for the new tag and copy the asset's
# reported digest and size. The API's values are compared with these literals
# before any byte is downloaded, and the bytes themselves, downloaded or already
# on disk, are compared with the literal sha256 before anything is extracted.
VDJDB_PINNED_RELEASE_TAG = "2026-06-03-ZENODO"
VDJDB_PINNED_ASSET_NAME = "vdjdb-2026-06-03.zip"
VDJDB_PINNED_SHA256 = "9ae7724ab2ce9fdb09b1fb82a90c7462f75269f14f669238f8c154a80239cfc1"
VDJDB_PINNED_SIZE_BYTES = 41017409

VDJDB_RELEASES_API = (
    "https://api.github.com/repos/antigenomics/vdjdb-db/releases/tags/" + VDJDB_PINNED_RELEASE_TAG
)

# The asset URL comes from the release API response, so it is checked against
# literals before it is opened: https, the netloc github.com exactly (no port,
# no userinfo, no look-alike host) and a path under this repository's release
# downloads with no ".." segment, encoded or not. Every redirect target is
# checked before urllib follows it, and must pass the same check or be https on
# one of VDJDB_ASSET_HOSTS. VDJDB_PINNED_SHA256, not this allowlist, is what
# vouches for the bytes that arrive.
VDJDB_DOWNLOAD_PATH_PREFIX = "/antigenomics/vdjdb-db/releases/download/"

# The hosts GitHub is known to redirect release-asset downloads to. Other
# githubusercontent.com hosts, such as raw. and gist., serve user-writable
# content and are refused; so is any host GitHub moves to that is not listed.
VDJDB_ASSET_HOSTS = frozenset(
    {"objects.githubusercontent.com", "release-assets.githubusercontent.com"}
)

# Seconds the opener may wait on the connect and on each socket read. It bounds
# a stalled server, not the total transfer time of the archive.
DOWNLOAD_TIMEOUT_SECONDS = 60

# Total uncompressed bytes a VDJdb archive may extract to. fetch_vdjdb extracts
# only an archive whose bytes matched VDJDB_PINNED_SHA256, so the pin, not this
# bound, is what keeps an arbitrary archive out; the bound caps what extraction
# can write if the pin is ever moved to an archive nobody measured. The pinned
# archive's central directory, read 2026-10-12, declares 11 members totalling
# 425,297,801 bytes uncompressed, which this admits with room to spare.
VDJDB_MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024


def _check_asset_url(url: str, *, redirect: bool = False) -> None:
    """Refuse a URL outside the allowlist above; redirect=True adds VDJDB_ASSET_HOSTS."""
    parts = urlsplit(url)
    segments = unquote(parts.path).replace("\\", "/").split("/")
    on_repo = (
        parts.netloc == "github.com"
        and parts.path.startswith(VDJDB_DOWNLOAD_PATH_PREFIX)
        and ".." not in segments
    )
    allowed = on_repo or (redirect and parts.netloc in VDJDB_ASSET_HOSTS)
    # urlsplit drops tab, CR and LF, but urllib.request keeps them, so a URL
    # with any control, space or non-ASCII character is refused outright.
    printable = all(32 < ord(ch) < 127 for ch in url)
    if parts.scheme != "https" or not allowed or not printable:
        shown = url.split("?", 1)[0].split("#", 1)[0]  # a query may hold a signed token
        raise RuntimeError(f"Refusing VDJdb download URL {shown!r}: not an allowed https location")


class _CheckedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check each redirect target with _check_asset_url before urllib requests it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_asset_url(newurl, redirect=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _safe_extract_zip(zip_ref: zipfile.ZipFile, dest_dir: str) -> None:
    """Extract with a zip-slip guard and a bound on total uncompressed size.

    Both checks run over the whole member list BEFORE anything is written, so a
    refusal leaves the destination untouched. The size bound is on the TOTAL,
    because many small members can exceed any per-member limit between them.

    A member cannot write more than the size its own header declares: ZipExtFile
    stops at that count and then verifies the CRC, so summing the declared sizes
    is a sound bound and needs no streaming.
    """
    dest_root = os.path.realpath(dest_dir)
    total = 0
    for member in zip_ref.infolist():
        target = os.path.realpath(os.path.join(dest_dir, member.filename))
        if target != dest_root and not target.startswith(dest_root + os.sep):
            raise RuntimeError(f"Unsafe zip member path (path traversal): {member.filename}")
        total += member.file_size
    if total > VDJDB_MAX_EXTRACTED_BYTES:
        raise RuntimeError(
            f"VDJdb archive would extract {total} bytes, over the "
            f"{VDJDB_MAX_EXTRACTED_BYTES} byte limit; refusing to extract it"
        )
    zip_ref.extractall(dest_dir)


def _resolve_vdjdb_asset() -> tuple[str, str, int]:
    """Resolve the PINNED VDJdb release's .zip asset: its URL, sha256 and size.

    The endpoint names VDJDB_PINNED_RELEASE_TAG, so "latest" is never consulted,
    and the response is checked against the pin on four axes - the tag it
    reports (a response with no tag fails it), the asset's name, its digest and
    its size - before any byte is downloaded. The values RETURNED are the pinned
    literals rather than the API's, so an API that misreports a digest cannot
    choose which bytes this pipeline accepts.
    """
    import json

    req = urllib.request.Request(VDJDB_RELEASES_API)
    req.add_header("Accept", "application/vnd.github+json")
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        data = json.loads(resp.read())
    tag = data.get("tag_name")
    if tag != VDJDB_PINNED_RELEASE_TAG:
        raise RuntimeError(
            f"VDJdb release endpoint reported tag {tag!r}, not the pinned "
            f"{VDJDB_PINNED_RELEASE_TAG!r}; refusing to ingest an unpinned release"
        )
    assets = data.get("assets", [])
    for asset in assets:
        if asset.get("name") == VDJDB_PINNED_ASSET_NAME:
            # GitHub reports these as "sha256:<64 hex>" and a byte count. Without
            # them the download cannot be verified or bounded, so it is refused.
            algorithm, _, expected = str(asset.get("digest") or "").partition(":")
            expected = expected.lower()
            if algorithm != "sha256" or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise RuntimeError(f"VDJdb asset {asset['name']} has no valid sha256 digest")
            size = asset.get("size")
            if type(size) is not int or size <= 0:
                raise RuntimeError(f"VDJdb asset {asset['name']} has no usable size")
            if expected != VDJDB_PINNED_SHA256:
                raise RuntimeError(
                    f"VDJdb asset {asset['name']} digest {expected} does not match the pinned "
                    f"{VDJDB_PINNED_SHA256}; the release this pipeline ingests has been "
                    "republished, or the response is not authentic"
                )
            if size != VDJDB_PINNED_SIZE_BYTES:
                raise RuntimeError(
                    f"VDJdb asset {asset['name']} size {size} does not match the pinned "
                    f"{VDJDB_PINNED_SIZE_BYTES} bytes"
                )
            return asset["browser_download_url"], VDJDB_PINNED_SHA256, VDJDB_PINNED_SIZE_BYTES
    found = sorted(str(a.get("name")) for a in assets)
    raise RuntimeError(
        f"No asset named {VDJDB_PINNED_ASSET_NAME} in VDJdb release {tag}; it carries {found}"
    )


class _ExtractionFailed(RuntimeError):
    """Extracting a verified download began and did not finish."""


def _download_verified(
    url: str, zip_path: str, expected_sha256: str, max_bytes: int, extract_to: str
) -> None:
    """Download to a .part file, verify and extract it, and only then rename it to zip_path.

    Any failure removes the .part file, so this only ever puts an archive at
    zip_path once its bytes matched expected_sha256 and it was fully extracted.
    A failure before extraction starts leaves zip_path as it was. zip_path
    (absent, or not the pinned archive, whenever this runs) is removed as
    extraction starts, before anything is written, and again after a failure
    once extraction has started, in case a member was extracted onto it, so no
    failure leaves an archive name beside a directory this run half wrote. An
    Exception raised by extraction itself is re-raised as _ExtractionFailed.
    """
    _check_asset_url(url)
    partial = zip_path + ".part"
    opener = urllib.request.build_opener(_CheckedRedirectHandler)
    extracting = False
    try:
        # The start URL was checked above, and _CheckedRedirectHandler checks
        # each redirect target before it is requested, so nothing is read from,
        # and no .part file is opened for, a URL outside the allowlist.
        with (
            opener.open(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            open(partial, "wb") as output,
        ):
            written = 0
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise RuntimeError(f"VDJdb archive exceeds the expected {max_bytes} bytes")
                output.write(chunk)
        actual = sha256_file(partial)
        if actual != expected_sha256:
            raise RuntimeError(
                f"VDJdb archive checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        with zipfile.ZipFile(partial, "r") as zip_ref:
            extracting = True
            # Extraction may overwrite files beside the old archive, so it goes
            # first: even a kill mid-extraction, which runs no cleanup, then
            # leaves no archive name beside a half-written directory.
            if os.path.exists(zip_path):
                os.remove(zip_path)
            try:
                _safe_extract_zip(zip_ref, extract_to)
            except Exception as exc:
                raise _ExtractionFailed(f"extracting the verified archive failed: {exc}") from exc
        os.replace(partial, zip_path)
    except BaseException:
        if os.path.exists(partial):
            os.remove(partial)
        if extracting and os.path.exists(zip_path):
            os.remove(zip_path)
        raise


def _archive_matches_pin(zip_path: str) -> bool:
    """True when zip_path is byte for byte the pinned archive: its size, then its sha256."""
    return (
        os.path.getsize(zip_path) == VDJDB_PINNED_SIZE_BYTES
        and sha256_file(zip_path) == VDJDB_PINNED_SHA256
    )


# The table fetch_vdjdb returns, by member basename, in order of preference.
_TABLE_NAMES = ("vdjdb.txt", "vdjdb_full.txt")


def _table_member(zip_ref: zipfile.ZipFile) -> str:
    """Name the archive member holding the VDJdb table.

    It is found in the archive's own member list, never by searching the
    directory the archive is extracted into, so a table that another release or
    a manual extraction left there cannot be returned in place of this one.
    """
    members = [m.filename for m in zip_ref.infolist() if not m.is_dir()]
    for name in _TABLE_NAMES:
        for member in members:
            if member.rsplit("/", 1)[-1] == name:
                return member
    raise RuntimeError(f"the VDJdb archive has no member named {' or '.join(_TABLE_NAMES)}")


def fetch_vdjdb(output_dir: str = "data") -> str | None:
    """Return a VDJdb table extracted IN THIS RUN from the pinned archive, or None.

    The only archive used is one whose bytes this run compared with the pin. A
    vdjdb.zip already in output_dir is hashed first and, if it matches, extracted
    again; otherwise the pinned release is downloaded and verified. Every refusal
    or failure returns None: no table already in output_dir is ever returned in
    place of one extracted from verified bytes, whether the archive beside it
    failed the pin, the release endpoint refused, or there was no network.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "vdjdb.zip")
    if os.path.exists(zip_path):
        if _archive_matches_pin(zip_path):
            print(f"{zip_path} matches the pinned sha256; extracting it again.")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    member = _table_member(zip_ref)
                    _safe_extract_zip(zip_ref, output_dir)
            except Exception as e:
                print(f"Failed to extract the pinned VDJdb archive: {e}")
                return None
            return os.path.join(output_dir, *member.split("/"))
        print(f"{zip_path} is not the pinned archive; fetching the pinned release.")
    try:
        url, expected_sha256, size = _resolve_vdjdb_asset()
        print(f"Downloading VDJdb from {url}...")
        _download_verified(url, zip_path, expected_sha256, size, output_dir)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            member = _table_member(zip_ref)
    except _ExtractionFailed as e:
        # (Extraction counts as started on entering _safe_extract_zip, so a
        # refusal from its member checks, which writes nothing, also lands here.)
        print(f"Failed to fetch VDJdb: {e}")
        print(f"Files in {output_dir} may have been partly overwritten; not using them.")
        return None
    except Exception as e:
        print(f"Failed to fetch VDJdb: {e}")
        print(
            f"No table in {output_dir} is used in its place: only one extracted from "
            "the pinned archive in this run is ingested. Pass a VDJdb table with "
            "--input to ingest it unverified."
        )
        return None
    print("Downloaded and extracted VDJdb.")
    return os.path.join(output_dir, *member.split("/"))


def ingest_vdjdb(input_path, output_path, schema_path):
    # True only for a table fetch_vdjdb returned, which it does only for one
    # extracted in this run from an archive whose bytes matched the pin. A local
    # --input is not checked against the pin, and is recorded as unverified.
    verified = False
    if not input_path or not os.path.exists(input_path):
        input_path = fetch_vdjdb()
        if not input_path or not os.path.exists(input_path):
            raise FileNotFoundError(f"Could not find or fetch VDJdb input: {input_path}")
        verified = True

    print(f"Ingesting VDJdb from {input_path}")
    df = pd.read_csv(input_path, sep="\t")

    # --- Column name normalization across VDJdb releases ---
    # Old format: 'Epitope', 'MHC A', 'Epitope species', 'Epitope gene', 'CDR3.alpha', 'CDR3.beta'
    # New format (2026+): 'antigen.epitope', 'mhc.a', 'antigen.species', 'antigen.gene', 'cdr3', 'gene'
    _col_map = {
        "antigen.epitope": "Epitope",
        "mhc.a": "MHC A",
        "antigen.species": "Epitope species",
        "antigen.gene": "Epitope gene",
    }
    for old_name, new_name in _col_map.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})

    if "Epitope" not in df.columns or "MHC A" not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(
            "VDJdb input missing expected columns 'Epitope'/'antigen.epitope' and 'MHC A'/'mhc.a'"
        )

    # --- Species filter: exclude non-viral entries ---
    # SESTRAV is a viral antigen immunogenicity predictor. VDJdb also contains
    # human autoantigens/tumor antigens, synthetic controls, and bacterial/fungal
    # epitopes which are biologically incompatible with viral training and must
    # be excluded. Confirmed non-viral species from VDJdb 2026-06-03 release.
    _EXCLUDE_SPECIES = frozenset(
        [
            # Human autoantigens and tumor/neoantigen epitopes
            "HomoSapiens",
            "Homo sapiens",
            # Synthetic / non-natural peptides
            "Synthetic",
            # Bacteria
            "KlebsiellaOxytoca",
            "M.tuberculosis",
            "E.Coli",
            "StreptomycesKanamyceticus",
            # Fungi
            "AspergillusOryzae",
            "FusariumOxysporum",
            "CryptococcusNeoformans",
            # Plants
            "Wheat",
        ]
    )
    _species_col = "Epitope species"
    if _species_col in df.columns:
        n_before = len(df)
        df = df[~df[_species_col].isin(_EXCLUDE_SPECIES)]
        n_excluded = n_before - len(df)
        if n_excluded > 0:
            print(
                f"Excluded {n_excluded} non-viral rows "
                f"(human autoantigens, bacteria, fungi, synthetic controls)."
            )

    # --- CDR3 extraction ---
    # New VDJdb (2026+) has one CDR3 per row with 'gene' column (TRA or TRB).
    # Pivot into alpha/beta columns keyed on (Epitope, MHC A).
    _new_cdr3_format = (
        "cdr3" in df.columns and "gene" in df.columns and "CDR3.alpha" not in df.columns
    )

    if _new_cdr3_format:
        # Pivot: for each (Epitope, MHC A), extract TRA and TRB CDR3 sequences.
        tra = (
            df[df["gene"] == "TRA"][["Epitope", "MHC A", "cdr3"]]
            .drop_duplicates(subset=["Epitope", "MHC A"])
            .rename(columns={"cdr3": "CDR3.alpha"})
        )
        trb = (
            df[df["gene"] == "TRB"][["Epitope", "MHC A", "cdr3"]]
            .drop_duplicates(subset=["Epitope", "MHC A"])
            .rename(columns={"cdr3": "CDR3.beta"})
        )
        # Start from TRB (dominant chain), left-join alpha
        pivot = trb.merge(tra, on=["Epitope", "MHC A"], how="outer")
        _alpha_col, _beta_col = "CDR3.alpha", "CDR3.beta"
    else:
        pivot = None
        # Legacy CDR3 column detection
        _alpha_cols = ["CDR3.alpha", "cdr3.alpha", "CDR3_alpha", "cdr3_alpha"]
        _beta_cols = ["CDR3.beta", "cdr3.beta", "CDR3_beta", "cdr3_beta", "CDR3", "cdr3"]
        _alpha_col = next((c for c in _alpha_cols if c in df.columns), None)
        _beta_col = next((c for c in _beta_cols if c in df.columns), None)

    # Sort deterministically before deduplication so the retained CDR3 is stable across runs.
    sort_cols = ["Epitope", "MHC A"]
    if not _new_cdr3_format and _beta_col:
        sort_cols.append(_beta_col)
    df = df.sort_values(sort_cols, na_position="last")

    # Keep unique epitope/allele pairs (many TCRs bind the same epitope).
    unique_epitopes = df.drop_duplicates(subset=["Epitope", "MHC A"]).copy()
    unique_epitopes = unique_epitopes[unique_epitopes["MHC A"].str.contains("HLA", na=False)].copy()

    # Map VDJdb taxonomy species names to SESTRAV canonical virus labels.
    # VDJdb uses its own naming conventions; these must align with IEDB-sourced
    # labels in the main dataset so dedup and per-virus evaluation work correctly.
    _VDJDB_VIRUS_MAP: dict[str, str] = {
        "InfluenzaA": "IAV",
        "InfluenzaB": "IBV",
        "HPV16": "HPV",
        "HPV18": "HPV",
        "InfluenzaA(H1N1)": "IAV",
        "InfluenzaA(H3N2)": "IAV",
        "YellowFeverVirus": "YFV",
        "SARSCoV2": "SARS-CoV-2",
        "SARS-CoV": "SARS-CoV",
    }
    raw_species = unique_epitopes.get("Epitope species", pd.Series(dtype="object")).fillna(
        "Unknown"
    )
    mapped_virus = raw_species.map(lambda v: _VDJDB_VIRUS_MAP.get(str(v).strip(), str(v).strip()))

    # Build the v5 DataFrame
    df_v4 = pd.DataFrame(
        {
            "peptide": unique_epitopes["Epitope"].values,
            "label": 1,  # VDJdb contains known positive binders
            "virus": mapped_virus.values,
            "protein": unique_epitopes.get("Epitope gene", pd.Series(dtype="object"))
            .fillna("Unknown")
            .values,
            "strain": "Unknown",
            "hla_allele": unique_epitopes["MHC A"].values,
            "source_type": "Virus",
            "database_source": "VDJdb",
            "is_quarantined": False,
        }
    )

    # Attach CDR3 sequences
    if _new_cdr3_format and pivot is not None:
        df_v4 = df_v4.merge(
            pivot, left_on=["peptide", "hla_allele"], right_on=["Epitope", "MHC A"], how="left"
        )
        df_v4 = df_v4.drop(columns=["Epitope", "MHC A"], errors="ignore")
        df_v4 = df_v4.rename(columns={"CDR3.alpha": "tcr_alpha_cdr3", "CDR3.beta": "tcr_beta_cdr3"})
    else:
        if _alpha_col:
            df_v4["tcr_alpha_cdr3"] = (
                unique_epitopes[_alpha_col]
                .where(unique_epitopes[_alpha_col].notna(), other=None)
                .values
            )
        else:
            df_v4["tcr_alpha_cdr3"] = None
        if _beta_col:
            df_v4["tcr_beta_cdr3"] = (
                unique_epitopes[_beta_col]
                .where(unique_epitopes[_beta_col].notna(), other=None)
                .values
            )
        else:
            df_v4["tcr_beta_cdr3"] = None

    # Ensure CDR3 columns exist
    for col in ["tcr_alpha_cdr3", "tcr_beta_cdr3"]:
        if col not in df_v4.columns:
            df_v4[col] = None

    cdr3_alpha_count = int(df_v4["tcr_alpha_cdr3"].notna().sum())
    cdr3_beta_count = int(df_v4["tcr_beta_cdr3"].notna().sum())
    if cdr3_alpha_count or cdr3_beta_count:
        print(f"CDR3 sequences captured - alpha: {cdr3_alpha_count}, beta: {cdr3_beta_count} rows")
    else:
        print(
            "CDR3 columns not found in this VDJdb release; tcr_alpha_cdr3/tcr_beta_cdr3 set to null."
        )

    df_v4 = df_v4.dropna(subset=["peptide", "hla_allele"])

    # Keep only high-resolution Class I alleles (e.g. HLA-A*02:01).
    valid_allele_mask = df_v4["hla_allele"].str.match(r"^[A-Z]+-[A-Z]\*\d{2}:\d{2}$").fillna(False)
    print(f"Dropped {int((~valid_allele_mask).sum())} entries due to low-resolution HLA alleles.")
    df_v4 = df_v4[valid_allele_mask]

    df_v4 = normalize_peptides(df_v4)
    df_v4 = df_v4.drop_duplicates(subset=["peptide", "hla_allele"])
    df_v4 = df_v4.sort_values(["peptide", "hla_allele"]).reset_index(drop=True)

    # Replace NaN with None so JSON schema validation sees null, not NaN.
    for col in ["tcr_alpha_cdr3", "tcr_beta_cdr3"]:
        if col in df_v4.columns:
            df_v4[col] = df_v4[col].where(df_v4[col].notna(), other=None)

    validate_against_schema(df_v4, schema_path)
    # An empty result used to be written out: an empty CSV plus a sidecar recording
    # row_count 0, at exit 0. That is exactly the shape a truncated extraction leaves
    # behind - a vdjdb.txt holding only its header reaches here with zero rows - and a
    # downstream build reading that sidecar cannot tell it from a release that
    # genuinely contributed nothing. Refuse before either file is written.
    if len(df_v4) == 0:
        raise ValueError(
            f"VDJdb ingest produced no rows from {os.path.basename(input_path)}: the table "
            "is truncated, or every row was filtered out. Refusing to write an empty "
            "artifact and its provenance sidecar."
        )
    print(f"Extracted {len(df_v4)} unique positive epitopes from VDJdb.")
    df_v4.to_csv(output_path, index=False)
    write_provenance(
        output_path,
        sources=[input_path],
        row_count=len(df_v4),
        extra={
            "database": "VDJdb",
            # Which release the rows came from, recorded only when this run
            # verified the archive's bytes against the pin. Otherwise the release
            # fields are None, because nothing checked which release it was.
            "archive_verified_against_pin": verified,
            "pinned_release_tag": VDJDB_PINNED_RELEASE_TAG if verified else None,
            "pinned_asset_name": VDJDB_PINNED_ASSET_NAME if verified else None,
            "pinned_asset_sha256": VDJDB_PINNED_SHA256 if verified else None,
            "url": VDJDB_RELEASES_API if verified else None,
            "tcr_alpha_cdr3_populated": cdr3_alpha_count,
            "tcr_beta_cdr3_populated": cdr3_beta_count,
        },
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest VDJdb TSV into SESTRAV v5 schema")
    parser.add_argument(
        "--input", required=False, help="Path to VDJdb TSV file (will fetch if omitted)"
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument(
        "--schema",
        default="data/immunogenicity_dataset_v5_schema.json",
        help="Schema path for validation (default: v5)",
    )
    args = parser.parse_args()

    ingest_vdjdb(args.input, args.output, args.schema)
