# Data Directory

This directory holds the reference PDF document for the Gas Turbine Maintenance Assistant:

- **Target File:** `GER-3620-P.pdf`
- **Document Title:** GE GER-3620 Rev P — Heavy-Duty Gas Turbine Operating and Maintenance Considerations (60 pages).

## Instructions

If `GER-3620-P.pdf` is not already present in this directory:
1. Place the official `GER-3620-P.pdf` into this `data/` folder.
2. Run the ingestion command:
   ```bash
   python -m scripts.ingest_pdf
   ```
3. The ingestion pipeline computes a SHA-256 fingerprint, extracts 60 pages with 1-based page metadata, detects section headings, generates chunks of ~900 characters with 120 overlap, and persists vectors to `chroma_db/`.
