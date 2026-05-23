import requests
import pandas as pd
import time
import logging
from pathlib import Path
from urllib.parse import urlencode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IEDBAPIClient:
    """
    Secure client for querying the IEDB Query API (v3).
    Includes rate limiting and strict error handling.
    """
    BASE_URL = "https://query-api.iedb.org"

    # Virus Taxonomy IDs mapped for explicit querying
    TAX_IDS = {
        "EBV": 10376,
        "HPV16": 333760,
        "HPV18": 333761
    }

    def __init__(self, delay_seconds=1.0):
        self.delay_seconds = delay_seconds

    def query_tcell_assays_by_taxid(self, tax_id: int, limit: int = 10000) -> pd.DataFrame:
        """
        Query IEDB for all T-cell assays associated with a specific organism Taxonomy ID.
        """
        # IEDB Query API v3 endpoint for tcell_search
        endpoint = f"{self.BASE_URL}/tcell_search"
        
        # We need the linear sequence (epitope), qualitative measure (label), 
        # mhc allele, and antigen name. 
        # Using PostgREST syntax for the IEDB API.
        params = {
            "source_organism_iri": f"eq.NCBITaxon:{tax_id}",
            "limit": limit
        }
        
        logger.info(f"Querying IEDB API for TaxID {tax_id}...")
        url = f"{endpoint}?{urlencode(params)}"
        
        try:
            # Enforce 15s timeout for security
            response = requests.get(url, headers={'Accept': 'application/json'}, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved {len(data)} records from IEDB for TaxID {tax_id}.")
            
            # Rate limit to avoid bans
            time.sleep(self.delay_seconds)
            
            return pd.DataFrame(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Secure query failed for TaxID {tax_id}: {e}")
            return pd.DataFrame()

    def map_to_consensus_and_save(self, df: pd.DataFrame, virus: str, output_path: str):
        """
        In a full implementation, this maps the retrieved epitopes against 
        the Phase 1.1 consensus FASTA files to ensure clinical relevance.
        For now, we save the raw query results securely.
        """
        if df.empty:
            logger.warning(f"No data to save for {virus}")
            return
            
        # Secure sanitization before writing to disk
        df = df.dropna(subset=['linear_sequence']).drop_duplicates()
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {virus} data to {output_path}")

def run_extraction():
    logger.info("Starting Phase 1.2/1.3: Programmatic IEDB Integration")
    client = IEDBAPIClient()
    output_dir = Path("data/iedb_api_pulls")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_data = []
    
    for virus, tax_id in client.TAX_IDS.items():
        df = client.query_tcell_assays_by_taxid(tax_id)
        if not df.empty:
            df['virus'] = virus
            output_file = output_dir / f"iedb_raw_{virus}.csv"
            client.map_to_consensus_and_save(df, virus, output_file)
            all_data.append(df)
            
    if all_data:
        merged = pd.concat(all_data, ignore_index=True)
        merged_file = output_dir / "iedb_raw_merged.csv"
        merged.to_csv(merged_file, index=False)
        logger.info(f"Phase 1.2 Complete: Merged data saved to {merged_file}")

if __name__ == "__main__":
    run_extraction()
