# HCR 3.0 PROBEGEN V2

A local web app for designing Hybridization Chain Reaction (HCR 3.0) probe pools from NCBI accessions or Entrez Gene IDs. It wraps `HCRProbeMakerCL-main/v2_0/HCR.py` and provides a UI for single‑target and batch workflows, including automated pool assembly and IDT oPool ordering files.

Please see note about ordering amplifier hairpins (B1-B5) from IDT at the bottom.

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
From the repo root:
cd [DOWNLOAD_LOCATION]/HCR_3.0_Probe_Generator_v2-main
```
python3 hcr-webapp/start.py
```

This will:
- Create a local virtual environment
- Install all Python requirements
- Launch the web app at `http://127.0.0.1:8000`

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
- `hcr-webapp/` — web app UI

## Troubleshooting
- If you see `HTTP Error 429: Too Many Requests`, the app automatically retries with a short delay.
- If the folder picker doesn’t appear on macOS, allow Terminal/iTerm to control Finder in **System Settings → Privacy & Security → Automation**.

## License and citation
See upstream `HCRProbeMakerCL-main/README.md` for citation guidance.

## NOTE ABOUT HCR AMPLIFIERS
- The following describes how to order amplifers for B1-B5 via IDT.
- Amplifier hairpin sequence are provided here for convenience. 
- Sequences were obtained from the supplentary data file from the original HCR publication (Choi et al. 2014) https://pubs.acs.org/doi/10.1021/nn405717p. 
- Please check with the publication yourself before ordering and testing in case I erroneously entered sequences.

B1H1:
```
CgTAAAggAAgACTCTTCCCgTTTgCTgCCCTCCTCgCATTCTTTCTTgAggAgggCAgCAAACgggAAgAg
```
B1H2:
```
gAggAgggCAgCAAACgggAAgAgTCTTCCTTTACgCTCTTCCCgTTTgCTgCCCTCCTCAAgAAAgAATgC
```

B2H1:
```
ggCggTTTACTggATgATTgATgAggATTTACgAggAgCTCAgTCCATCCTCgTAAATCCTCATCAATCATC
```
B2H2:
```
CCTCgTAAATCCTCATCAATCATCCAgTAAACCgCCgATgATTgATgAggATTTACgAggATggACTgAgCT
```

B3H1:
```
CgggTTAAAgTTgAgTggAgATATAgAggCAgggACAAAgTCTAATCCgTCCCTgCCTCTATATCTCCACTC
```
B3H2:
```
gTCCCTgCCTCTATATCTCCACTCAACTTTAACCCggAgTggAgATATAgAggCAgggACggATTAgACTTT
```

B4H1:
```
gAAgCgAATATggTgAgAgTTggAggTAggTTgAggCACATTTACAgACCTCAACCTACCTCCAACTCTCAC
```
B4H2:
```
CCTCAACCTACCTCCAACTCTCACCATATTCgCTTCgTgAgAgTTggAggTAggTTgAggTCTgTAAATgTg
```

B5H1:
```
ATTggATTTgTAgggTAgATAgAgATTgggAgTgAgCACTTCATATCACTCACTCCCAATCTCTATCTACCC
```
B5H2:
```
CTCACTCCCAATCTCTATCTACCCTACAAATCCAATgggTAgATAgAgATTgggAgTgAgTgATATgAAgTg
```

The options for fluorophores can be found here: https://www.idtdna.com/site/Catalog/Modifications/Dyes.
We've typically ordered hairpin amplifier oligos with Alexa Fluors. We have attempted to conjugate hairpin amplifier oligos with other fluorophores such as ATTO425 and ATTO490LS (these are long Stokes shift fluorophores not offered on IDT), but it is an absolute pain to do this yourself.

Generally, the structure for amplifier hairpin oligos are as follows (and this is what is demonstrated in Choi et al. 2014):
BXH1:
```
[AMPLIFIER SEQUENCE]-[INTERNAL SPACER]-[3' FLUOROPHORE MODIFICATION]
```
BXH2
```
[5' FLUOROPHORE MODIFICATION]-[INTERNAL SPACER]-[AMPLIFIER SEQUENCE]
```

In slight contrast to the above structures, we've found that having the fluorophore modification for BXH1 being on the 5' side of the oligo and having the fluorophore modification for the BXH2 being on the 3' side of the oligo makes no difference. For some reason, at the time of ordering on IDT, this was a less problematic synthesis option- although this does not seem to be an issue anymore. Amplifier hairpin oligos ordered in this fashion worked as expected, but theoretically, the original structure outlined in Choi et al. 2014 should be the optimal method. The following describes exactly how purified fluorescent hairpin amplifier oligos were ordered on IDT.

1. Product: Custom DNA Oligos > DNA Oligos > Single-stranded DNA
2. Synthesis scale: 100nmol
3. Enter the following sequences (two separate oligos, obviously):

This was the exact sequence/code that was entered in the IDT ordering portal for B2H1-5P-ALEXA647:
```
/5Alex647N//iSp18/GGCGGTTTACTGGATGATTGATGAGGATTTACGAGGAGCTCAGTCCATCCTCGTAAATCCTCATCAATCATC
```

This was the exact sequence/code that was entered in the IDT ordering portal for B2H2-3P-ALEXA647:
```
CCTCGTAAATCCTCATCAATCATCCAGTAAACCGCCGATGATTGATGAGGATTTACGAGGATGGACTGAGCT/iSp18//3AlexF647N/
```

Replace the amplifier sequence with your desired amplifier sequence and replace the fluorophore with your desired fluorophore as needed.

4. Purification: HPLC

For a 100nmol scale order that is HPLC purified, the minimum guaranteed yield is 1.5nmol. In most cases, actual delivered yields have been between 3-4nmol. In few cases, actual delivered yields have been between 1.5-2nmol, and yield have never fallen below the guaranteed minimum. In any case, a stock concentration of 3 μM (stock concentration provided by Molecular Instruments in the past) amplifier can be achieved by resuspending a 1 nmol yield with 333 μL 5X SSC. For a 20 μL amplification reaction of a single target, 0.4 μL of 3 μM BXH1 and 0.4 μL of 3 μM BXH2 is required. This theoretically allows for ~ 800 reactions given a 1 nmol yield (but you will definitely get more than this). 

## Some tips about the amplification reaction

If you are performing multiplexed amplification for multiple targets in a miniaturized fashion, I would use hairpin amplifier stocks that have been resuspended with higher concentrations (such as 6uM) so that you end up having to use less hairpin amplifier volume, leading to less dilution of the dextran sulfate in the amplification buffer.



