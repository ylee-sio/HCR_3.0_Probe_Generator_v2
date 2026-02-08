# HCR Probe Generator Web App (MVP)

Minimal local web app that wraps `HCRProbeMakerCL-main/v2_0/HCR.py`.

## One‑line setup + run
From the repo root, run:
```
python3 hcr-webapp/start.py
```

This will:
- Create a local virtual environment
- Install all Python requirements
- Launch the web app at `http://127.0.0.1:8000`

## Optional NCBI settings
```
export NCBI_EMAIL="you@lab.org"
export NCBI_API_KEY="your_key_here"
```

## Dev Run (optional)
```
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000
