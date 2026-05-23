import requests
import time
import os
import logging
from pathlib import Path

# Configure logging for security and auditing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants and Configuration
UNIPROT_REST_API = "https://rest.uniprot.org/uniprotkb/{}.fasta"
OUTPUT_DIR = Path("data/consensus_sequences")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Clinically relevant reference strains (serving as proxies for PaVE/EBVdb consensus)
TARGET_PROTEINS = {
    "HPV16": {
        "E6": "P03126",
        "E7": "P03129"
    },
    "HPV18": {
        "E6": "P06463",
        "E7": "P06788"
    },
    "EBV": {
        "EBNA1": "P03211",
        "LMP1": "P03230",
        "LMP2": "P12908",
        "BZLF1": "P03206"
    }
}

def fetch_fasta(uniprot_id: str) -> str:
    """Securely fetch FASTA sequence from UniProt API with rate limiting."""
    url = UNIPROT_REST_API.format(uniprot_id)
    try:
        # Enforce security with a timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {uniprot_id}: {e}")
        return ""

def main():
    logger.info("Starting Phase 1.1: Consensus Sequence Extraction")
    
    for virus, proteins in TARGET_PROTEINS.items():
        logger.info(f"Processing {virus}...")
        virus_dir = OUTPUT_DIR / virus
        virus_dir.mkdir(exist_ok=True)
        
        for protein_name, uniprot_id in proteins.items():
            output_file = virus_dir / f"{virus}_{protein_name}_{uniprot_id}.fasta"
            
            if output_file.exists():
                logger.info(f"  [SKIP] {output_file.name} already exists.")
                continue
                
            logger.info(f"  [FETCH] Downloading {protein_name} ({uniprot_id})...")
            fasta_data = fetch_fasta(uniprot_id)
            
            if fasta_data:
                # Sanitize the output file write
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(fasta_data)
                logger.info(f"  [SUCCESS] Saved to {output_file}")
            
            # Rate limiting to prevent server bans (Safety/Security protocol)
            time.sleep(1.0)
            
    logger.info("Phase 1.1 Extraction Complete.")

if __name__ == "__main__":
    main()
