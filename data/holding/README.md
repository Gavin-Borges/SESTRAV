# SESTRAV Data Holding Area

Viruses quarantined from training due to insufficient depth.

**Graduation criteria:** >= 50 rows AND >= 10 real tested negatives (negative_origin="tested_negative").

When a virus reaches the graduation threshold, move its rows from this directory
into the main IEDB ingestion pipeline and rebuild v5 (or later).

## Currently in holding

See `docs/data_registry.md` "v4 Composition Audit" section for the quarantine list.
No individual data files are committed here - use IEDB bulk export + ingest_iedb_negatives.py
to build the holding cohort.
