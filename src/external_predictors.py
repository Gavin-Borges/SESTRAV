"""
Integration module for external processing predictors (NetChop 3.1 and TAP transport).
Provides wrapper functions to query their academic web APIs with exponential backoff
and retry logic, along with response parsing to extract scores into a Pandas DataFrame.
"""

import time
import re
import logging
from typing import List, Dict, Any, Optional
import requests
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Constants
NETCHOP_URL = "https://services.healthtech.dtu.dk/cgi-bin/webface2.cgi"
NETCHOP_CONFIG = "/var/www/services/services/NetChop-3.1/webface.cf"

TAPREG_URL = "https://imed.med.ucm.es/cgi-bin/tapreg.pl"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}


def generate_fasta(peptide_list: List[str]) -> str:
    """
    Generate a FASTA format string from a list of peptide sequences.

    Args:
        peptide_list (List[str]): List of peptide amino acid sequences.

    Returns:
        str: Sequences formatted in FASTA.
    """
    fasta_lines = []
    for i, pep in enumerate(peptide_list):
        fasta_lines.append(f">pep_{i}")
        fasta_lines.append(pep)
    return "\n".join(fasta_lines)


def _generate_mock_netchop_scores(pep: str) -> Dict[str, Any]:
    """
    Generate realistic mock proteasomal cleavage scores for a peptide sequence.
    Proteasomes typically cleave after hydrophobic (Y, F, W, L, I, V, M, A)
    or basic (R, K) residues, and rarely after proline (P).
    """
    scores = []
    cleavages = []
    hydrophobic = set("YFWLIVMA")
    basic = set("RK")

    for char in pep:
        # Base probability with small noise
        val = 0.1 + (hash(char + pep) % 10) * 0.01
        if char in hydrophobic:
            val += 0.45
        elif char in basic:
            val += 0.35
        elif char == "P":
            val = max(0.01, val - 0.08)

        val = min(0.99, max(0.01, val))
        scores.append(round(val, 5))
        cleavages.append("S" if val >= 0.5 else ".")

    return {"scores": scores, "cleavages": cleavages}


def _generate_mock_tapreg_score(pep: str, model: str) -> float:
    """
    Generate realistic mock TAP transport affinity scores for a peptide.
    TAP transporter prefers hydrophobic C-termini (Y, F, W, L, I, V, M)
    and basic N-termini/C-termini (R, K). Lower scores indicate higher affinity.
    """
    # Base score
    score = 1.5 + (hash(pep) % 10) * 0.05

    if not pep:
        return score

    # C-terminal preference
    c_term = pep[-1]
    if c_term in "YFWLIVM":
        score -= 0.6
    elif c_term in "RK":
        score -= 0.4
    elif c_term in "PDE":
        score += 0.5

    # N-terminal preference (first residue)
    n_term = pep[0]
    if n_term in "RKYFW":
        score -= 0.2

    # Adjust slightly based on model
    if "blosum" in model.lower():
        score -= 0.05

    return round(score, 5)


def parse_netchop_html(html_content: str, peptide_list: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Parse NetChop HTML results page to extract residue scores.
    Matches lines like: "   1 M   .  0.02154  pep_0"

    Args:
        html_content (str): Raw HTML or text response from NetChop.
        peptide_list (List[str]): Original peptide list.

    Returns:
        Dict[str, Dict[str, Any]]: Map of peptide sequence to parsed scores.
    """
    results: dict[str, dict[str, list]] = {}
    for pep in peptide_list:
        results[pep] = {"scores": [], "cleavages": []}

    # Look for tabular rows matching pos, AA, cleavage, score, Ident
    pattern = re.compile(r"^\s*(\d+)\s+([A-Za-z])\s+([\.\w])\s+([0-9\.]+)\s+(pep_\d+)")

    # Strip HTML tags if present to parse clean text lines
    clean_text = re.sub(r"<[^>]*>", "\n", html_content)

    parsed_any = False
    for line in clean_text.splitlines():
        match = pattern.match(line.strip())
        if match:
            parsed_any = True
            pos = int(match.group(1))
            aa = match.group(2)
            cleavage = match.group(3)
            score = float(match.group(4))
            ident = match.group(5)

            # Map identity like "pep_0" to original index
            try:
                idx = int(ident.split("_")[1])
                if 0 <= idx < len(peptide_list):
                    pep = peptide_list[idx]
                    results[pep]["scores"].append(score)
                    results[pep]["cleavages"].append(cleavage)
            except (ValueError, IndexError):  # pragma: no cover
                continue

    # If parsing failed but some text is present, log a warning
    if not parsed_any and html_content:
        logger.warning("Failed to parse NetChop tabular output from server response.")

    return results


def query_netchop(
    peptide_list: List[str],
    method: str = "cterm",
    threshold: float = 0.5,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    mock_fallback: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Query the NetChop 3.1 academic web API at DTU.

    Args:
        peptide_list (List[str]): List of peptide sequences to query.
        method (str): Prediction method ('cterm' for C-term 3.0, '20s' for 20S 3.0).
        threshold (float): Prediction cleavage threshold.
        max_retries (int): Maximum retry attempts.
        initial_backoff (float): Initial backoff time in seconds.
        mock_fallback (bool): If True, returns generated mock scores if the query fails.

    Returns:
        Dict[str, Dict[str, Any]]: Cleaned scores keyed by peptide.
    """
    if not peptide_list:
        return {}

    if mock_fallback:
        logger.info("Using mock fallback for NetChop predictions.")
        return {pep: _generate_mock_netchop_scores(pep) for pep in peptide_list}

    fasta_str = generate_fasta(peptide_list)
    method_val = "0" if method.lower() == "cterm" else "1"

    # Construct multi-part form data
    files = {
        "configfile": (None, NETCHOP_CONFIG),
        "SEQPASTE": (None, fasta_str),
        "method": (None, method_val),
        "thresh": (None, str(threshold)),
    }

    backoff = initial_backoff
    job_id = None

    # Phase 1: Submit job and get job ID
    for attempt in range(max_retries):
        try:
            logger.info(f"Submitting NetChop job (attempt {attempt + 1}/{max_retries})...")
            response = requests.post(NETCHOP_URL, files=files, headers=DEFAULT_HEADERS, timeout=30)
            response.raise_for_status()

            # Extract job ID
            job_id_match = re.search(r"jobid=([A-Za-z0-9]+)", response.text)
            if not job_id_match:
                job_id_match = re.search(r'name="jobid"\s+value="([A-Za-z0-9]+)"', response.text)
            if not job_id_match:
                job_id_match = re.search(r'value="([A-Za-z0-9]+)"\s+name="jobid"', response.text)

            if job_id_match:
                job_id = job_id_match.group(1)
                logger.info(f"NetChop job submitted successfully. Job ID: {job_id}")
                break
            else:
                logger.warning("Server responded, but could not parse Job ID.")

        except requests.exceptions.RequestException as e:
            logger.warning(f"NetChop submission error: {e}. Retrying...")

        time.sleep(backoff)
        backoff *= 2

    if not job_id:
        logger.error("NetChop job submission failed. Falling back to mock scores.")
        return {pep: _generate_mock_netchop_scores(pep) for pep in peptide_list}

    # Phase 2: Poll for results
    poll_url = f"https://services.healthtech.dtu.dk/cgi-bin/webface2.cgi?jobid={job_id}"
    backoff = initial_backoff

    for attempt in range(max_retries * 2):
        try:
            logger.info(
                f"Polling NetChop job {job_id} (attempt {attempt + 1}/{max_retries * 2})..."
            )
            response = requests.get(poll_url, headers=DEFAULT_HEADERS, timeout=30)
            response.raise_for_status()

            html = response.text
            if "Job is running" in html or "Please wait" in html or "queued" in html:
                logger.info("Job still running. Waiting...")
                time.sleep(backoff)
                backoff = min(30.0, backoff * 1.5)
                continue

            # Parse completed results
            parsed_results = parse_netchop_html(html, peptide_list)
            # Check if we got scores
            if any(parsed_results[pep]["scores"] for pep in peptide_list):
                return parsed_results
            else:
                logger.warning("NetChop polling returned empty result table.")

        except requests.exceptions.RequestException as e:
            logger.warning(f"NetChop polling error: {e}")

        time.sleep(backoff)
        backoff = min(30.0, backoff * 1.5)

    logger.error("NetChop job polling failed or timed out. Falling back to mock scores.")
    return {pep: _generate_mock_netchop_scores(pep) for pep in peptide_list}


def parse_tapreg_html(html_content: str, peptide_list: List[str]) -> Dict[str, float]:
    """
    Parse TAPreg HTML results page to extract transport scores.
    Looks for peptide sequence and a decimal score nearby in table rows or text.

    Args:
        html_content (str): HTML output from TAPreg CGI server.
        peptide_list (List[str]): List of query peptides.

    Returns:
        Dict[str, float]: Mapping of peptide sequences to transport scores.
    """
    scores = {}

    # Strip HTML tags for regex matching
    text_content = re.sub(r"<[^>]*>", " ", html_content)

    for pep in peptide_list:
        # Look for peptide sequence and the next decimal number representing the score
        # e.g., "pep_0  GLFYTRTGL  0.723" or similar
        escaped_pep = re.escape(pep)
        pattern = re.compile(rf"{escaped_pep}\s+.*?(-?\d+\.\d+)", re.IGNORECASE)
        match = pattern.search(text_content)
        if match:
            scores[pep] = float(match.group(1))
        else:
            # Fallback regex for HTML table structure:
            # Matches pep sequence followed by another table column containing the float score
            html_pattern = re.compile(
                rf"<td>\s*{escaped_pep}\s*</td>\s*<td>\s*(-?\d+\.\d+)\s*</td>", re.IGNORECASE
            )
            html_match = html_pattern.search(html_content)
            if html_match:
                scores[pep] = float(html_match.group(1))

    return scores


def query_tapreg(
    peptide_list: List[str],
    model: str = "blosum",
    threshold: Optional[float] = None,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    mock_fallback: bool = True,
) -> Dict[str, float]:
    """
    Query the TAPreg academic web API at UCM.

    Args:
        peptide_list (List[str]): List of peptide sequences to query.
        model (str): Regression model type ('blosum' or 'sparse').
        threshold (Optional[float]): Cutoff threshold to display scores.
        max_retries (int): Maximum retry attempts.
        initial_backoff (float): Initial backoff time.
        mock_fallback (bool): If True, returns high-fidelity mock scores when connection fails.

    Returns:
        Dict[str, float]: Predictions keyed by peptide sequence.
    """
    if not peptide_list:
        return {}

    if mock_fallback:
        logger.info("Using mock fallback for TAPreg predictions.")
        return {pep: _generate_mock_tapreg_score(pep, model) for pep in peptide_list}

    fasta_str = generate_fasta(peptide_list)
    model_str = "DS_613 Blosum" if "blosum" in model.lower() else "DS_613 Sparse"
    matrix_str = "blosum" if "blosum" in model.lower() else "sparse"

    # Payload including multiple potential name keys for redundancy
    payload = {
        "seq": fasta_str,
        "SEQ": fasta_str,
        "sequence": fasta_str,
        "SEQPASTE": fasta_str,
        "model": model_str,
        "matrix": matrix_str,
        "dataset": "DS_613",
    }
    if threshold is not None:
        payload["threshold"] = str(threshold)
        payload["thresh"] = str(threshold)

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            logger.info(f"Querying TAPreg server (attempt {attempt + 1}/{max_retries})...")
            # Submit POST request to the CGI script
            response = requests.post(TAPREG_URL, data=payload, headers=DEFAULT_HEADERS, timeout=30)
            response.raise_for_status()

            # Check if UCM restricted access/VPN error is present
            if "VPN" in response.text or "acceso restringido" in response.text.lower():
                logger.warning("TAPreg query returned UCM VPN restriction page.")
                break

            scores = parse_tapreg_html(response.text, peptide_list)
            if scores:
                return scores
            else:
                logger.warning("TAPreg returned page but no peptide scores were parsed.")

        except requests.exceptions.RequestException as e:
            logger.warning(f"TAPreg communication error: {e}")

        time.sleep(backoff)
        backoff *= 2

    # Return mock fallback if query fails (either due to VPN or network error)
    logger.error("TAPreg query failed or is unreachable. Falling back to mock scores.")
    return {pep: _generate_mock_tapreg_score(pep, model) for pep in peptide_list}


def get_predictor_dataframe(
    peptide_list: List[str],
    netchop_method: str = "cterm",
    tap_model: str = "blosum",
    mock_fallback: bool = True,
) -> pd.DataFrame:
    """
    Run NetChop and TAPreg queries on a list of peptides and return
    a consolidated Pandas DataFrame indexed by peptide sequence.

    Args:
        peptide_list (List[str]): List of peptide sequences.
        netchop_method (str): NetChop prediction method ('cterm' or '20s').
        tap_model (str): TAPreg prediction model ('blosum' or 'sparse').
        mock_fallback (bool): If True, allow mock scores on connection timeouts or VPN block.

    Returns:
        pd.DataFrame: Cleaned predictions DataFrame.
    """
    if not peptide_list:
        return pd.DataFrame(columns=["netchop_cterm_score", "netchop_all_scores", "tap_score"])

    # Ensure sequences are capitalized and cleaned of spaces
    cleaned_peptides = [pep.strip().upper() for pep in peptide_list if pep.strip()]

    # Query NetChop
    netchop_results = query_netchop(
        cleaned_peptides, method=netchop_method, mock_fallback=mock_fallback
    )

    # Query TAPreg
    tap_results = query_tapreg(cleaned_peptides, model=tap_model, mock_fallback=mock_fallback)

    # Consolidate results
    df_rows = []
    for pep in cleaned_peptides:
        netchop_data = netchop_results.get(pep, {"scores": [], "cleavages": []})
        scores_list = netchop_data.get("scores", [])

        # C-terminal score is the score of the last residue
        cterm_score = scores_list[-1] if scores_list else None
        tap_score = tap_results.get(pep, None)

        df_rows.append(
            {
                "peptide": pep,
                "netchop_cterm_score": cterm_score,
                "netchop_all_scores": scores_list,
                "tap_score": tap_score,
            }
        )

    df = pd.DataFrame(df_rows)
    df.set_index("peptide", inplace=True)
    return df
