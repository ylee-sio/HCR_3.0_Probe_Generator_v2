# HCR 3.0 PROBEGEN V2

A local desktop app for designing Hybridization Chain Reaction (HCR 3.0) probe pools from NCBI accessions or Entrez Gene IDs. It wraps `HCRProbeMakerCL-main/v2_0/HCR.py` and provides a friendly UI for single‑target and batch workflows, including automated pool assembly and IDT oPool ordering files.

## What you need to provide (minimum)
For **single‑target mode**:
- Output directory (where all files will be written)
- Target ID: NCBI accession or Entrez Gene ID
- Sequence type: CDS or full mRNA
- Max homopolymer length (0–5)
- Amplifier choice (B1–B5)

For **batch mode**:
- Output directory
- Targets list: either upload CSV/XLSX with two columns (`ID`, `Common name`) or enter targets manually
- For each target: sequence type, max homopolymer length (0–5), amplifier

## One‑line install + run
After downloading or cloning the repo, run:
```
python hcr-webapp/start.py
```

This will:
- Create a local virtual environment
- Install all requirements
- Create a desktop launcher (with a DNA‑style icon)
- Open the desktop app

## Optional NCBI settings
NCBI recommends including an email for API calls:
```
export NCBI_EMAIL="you@lab.org"
export NCBI_API_KEY="your_key_here"
```

## Inputs and what they do

### Core inputs (single + batch)
- **Target ID**: NCBI accession (e.g., `NM_001256799`) or Entrez Gene ID (numeric). Used to fetch sequence from NCBI.
- **Sequence type**:
  - `CDS`: fetches coding sequence only
  - `mRNA`: fetches full transcript
- **Max homopolymer length (0–5)**: maximum run length of identical bases allowed (A/T or G/C) when generating probes. Smaller values are stricter.
- **Amplifier (B1–B5)**: initiator set to use for probe design.
- **Desired probe pairs**: optional cap on the number of probe pairs returned.

### Advanced options (single + batch)
- **GC range (e.g., 20–70)**: allowable GC percentage window for candidate probe halves.
- **Max homopolymer A/T run**: overrides the global homopolymer max for A/T only.
- **Max homopolymer G/C run**: overrides the global homopolymer max for G/C only.
- **5' delay (bases)**: number of bases to skip from 5' end before probe design begins.
- **Soft min probe pairs**: skip targets that cannot reach this number of potential pairs.
- **BLAST transcriptome path**: optional transcriptome FASTA for off‑target flagging.

## Batch workflow summary
1. Upload CSV/XLSX or enter targets manually.
2. Configure per‑target options (or apply batch options to all).
3. Drag targets into pools (max 5 targets per pool).
4. Generate pools → per‑pool folders are created:
   - `HL_POOL_XXXXXXXX/` (random 8‑digit id)
   - Each target has its own subfolder with all intermediate files
   - A pooled `*_opool.xlsx` for IDT and a `pool_mapping.txt` file

## Repository layout
- `HCRProbeMakerCL-main/` — upstream probe generator CLI
- `hcr-webapp/` — desktop app + web UI

## Troubleshooting
- If you see `HTTP Error 429: Too Many Requests`, the app automatically retries with a short delay.
- If the folder picker doesn’t appear on macOS, allow Terminal/iTerm to control Finder in **System Settings → Privacy & Security → Automation**.

## License and citation
See upstream `HCRProbeMakerCL-main/README.md` for citation guidance.
