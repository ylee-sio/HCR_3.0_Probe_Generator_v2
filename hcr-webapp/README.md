# HCR Probe Generator Web App (MVP)

Minimal local web app that wraps `HCRProbeMakerCL-main/v2_0/HCR.py`.

## What it does
- Fetches a sequence from NCBI using an accession or Entrez Gene ID
- Chooses CDS-only or full mRNA FASTA
- Runs the HCR probe generator with a chosen amplifier (B1–B5)
- Saves all intermediates and outputs in your chosen directory
- Provides a ZIP download of the results

## Requirements
- Python 3.9+
- Internet access to NCBI
- NCBI email (required by Entrez)

## Setup
```
pip install -r requirements.txt
```

Optionally set these environment variables:
```
export NCBI_EMAIL="you@lab.org"
export NCBI_API_KEY="your_key_here"
```

## Run
```
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000

## Notes
- For Entrez Gene IDs, the app links to the first available nuccore record.
- BLAST options are not exposed in this MVP.
