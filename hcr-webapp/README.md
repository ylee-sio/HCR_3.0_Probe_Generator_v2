# HCR Probe Generator Web App (MVP)

Minimal local app that wraps `HCRProbeMakerCL-main/v2_0/HCR.py`.

## One-line setup + desktop app
After cloning the repo, run:
```
python start.py
```

What happens:
- A local virtual environment is created
- All Python requirements are installed
- A desktop app opens with a DNA-style icon
- A desktop launcher shortcut is created for future runs

## What it does
- Fetches a sequence from NCBI using an accession or Entrez Gene ID
- Chooses CDS-only or full mRNA FASTA
- Runs the HCR probe generator with a chosen amplifier (B1–B5)
- Saves all intermediates and outputs in your chosen directory
- Provides a ZIP download of the results

## Optional environment variables
```
export NCBI_EMAIL="you@lab.org"
export NCBI_API_KEY="your_key_here"
```

## Dev Run (optional)
```
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000
