from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import contextlib
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from Bio import Entrez
from Bio import SeqIO


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent.parent
HCR_ROOT = PROJECT_ROOT / "HCRProbeMakerCL-main" / "v2_0"
HCR_SCRIPT = HCR_ROOT / "HCR.py"

if str(HCR_ROOT) not in sys.path:
    sys.path.append(str(HCR_ROOT))

from mainscript_core import cleanup, variables, idpotentialprobes  # noqa: E402

app = FastAPI()
app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=APP_ROOT / "templates")

JOBS: dict[str, dict[str, str]] = {}


class UserInputError(Exception):
    pass


class ProbeCountRequest(BaseModel):
    target_id: str
    sequence_type: str
    homopolymer_max: int
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None


def _normalize_output_dir(base_dir: str, project_name: str) -> Path:
    if not base_dir.strip():
        raise UserInputError("Output base directory is required.")
    base = Path(base_dir).expanduser().resolve()
    if project_name.strip():
        safe_name = Path(project_name).name
        return base / safe_name
    return base


def _setup_entrez(email: str | None, api_key: str | None, require_email: bool = True) -> None:
    env_email = os.environ.get("NCBI_EMAIL", "").strip()
    Entrez.email = (email or env_email).strip()
    if not Entrez.email:
        if require_email:
            raise UserInputError("NCBI email is required (set NCBI_EMAIL or provide it in the form).")
        Entrez.email = "anonymous@example.com"
    env_key = os.environ.get("NCBI_API_KEY", "").strip()
    Entrez.api_key = (api_key or env_key).strip() or None


def _resolve_nuccore_id(identifier: str) -> str:
    if identifier.isdigit():
        handle = Entrez.elink(dbfrom="gene", db="nuccore", id=identifier)
        record = Entrez.read(handle)
        handle.close()
        links = record[0].get("LinkSetDb", [])
        if not links:
            raise UserInputError("No nuccore records linked to this Entrez Gene ID.")
        ids = links[0].get("Link", [])
        if not ids:
            raise UserInputError("No nuccore records linked to this Entrez Gene ID.")
        return ids[0]["Id"]
    return identifier


def _fetch_fasta(identifier: str, seq_kind: str) -> str:
    nuccore_id = _resolve_nuccore_id(identifier)
    rettype = "fasta_cds_na" if seq_kind == "cds" else "fasta"
    handle = Entrez.efetch(db="nuccore", id=nuccore_id, rettype=rettype, retmode="text")
    data = handle.read()
    handle.close()
    if not data.strip():
        raise UserInputError("NCBI returned empty sequence data for this ID.")
    return data


def _extract_sequences(fasta_text: str) -> list[str]:
    handle = io.StringIO(fasta_text)
    records = list(SeqIO.parse(handle, "fasta"))
    if not records:
        raise UserInputError("NCBI returned no FASTA records for this ID.")
    return [str(record.seq) for record in records if str(record.seq).strip()]


def _compute_probe_count(seq: str, homopolymer_max: int) -> int:
    if not seq or len(seq) < 52:
        return 0
    clean_seq = cleanup(seq)
    with contextlib.redirect_stdout(io.StringIO()):
        fullseq, cdna, _, hpA, hpT, hpC, hpG, position, table = variables(
            clean_seq,
            "B1",
            homopolymer_max,
            homopolymer_max,
            0,
        )
    result = idpotentialprobes(position, fullseq, cdna, table, hpA, hpT, hpC, hpG, 1.0, 0.0)
    try:
        if isinstance(result, (list, tuple)) and len(result) == 2 and result[0] == 0 and result[1] == 0:
            return 0
    except Exception:
        pass
    newlist = result[0]
    try:
        return int(len(newlist))
    except Exception:
        return 0


def _run_hcr(
    amp: str,
    fasta_path: Path,
    output_dir: Path,
    homopolymer_max: int,
    desired_pairs: int | None,
    gc_range: str | None,
    poly_at_max: int | None,
    poly_gc_max: int | None,
    five_prime_delay: int | None,
    blast_ref: str | None,
    soft_min_pairs: int | None,
) -> subprocess.CompletedProcess:
    poly_at_value = poly_at_max if poly_at_max is not None else homopolymer_max
    poly_gc_value = poly_gc_max if poly_gc_max is not None else homopolymer_max
    cmd = [
        os.fspath(Path(os.sys.executable)),
        os.fspath(HCR_SCRIPT),
        "-amp",
        amp,
        "-in",
        os.fspath(fasta_path),
        "-o",
        os.fspath(output_dir),
        "-polyAT",
        str(poly_at_value),
        "-polyCG",
        str(poly_gc_value),
    ]
    if gc_range:
        cmd.extend(["-gc", gc_range])
    if five_prime_delay is not None:
        cmd.extend(["-pause", str(five_prime_delay)])
    if blast_ref:
        cmd.extend(["-blast", blast_ref])
    if soft_min_pairs is not None:
        cmd.extend(["-min", str(soft_min_pairs)])
    if desired_pairs is not None:
        cmd.extend(["-max", str(desired_pairs)])
    return subprocess.run(
        cmd,
        cwd=HCR_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/pick-dir")
def pick_dir():
    try:
        try:
            result = subprocess.run(
                ["osascript", "-e", 'POSIX path of (choose folder)'],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                path = result.stdout.strip()
                if path:
                    return JSONResponse({"selected": True, "path": path})
        except Exception:
            pass

        return JSONResponse({"selected": False, "error": "Folder picker unavailable."})
    except Exception as exc:
        return JSONResponse({"selected": False, "error": str(exc)}, status_code=500)


@app.post("/probe-count")
def probe_count(payload: ProbeCountRequest):
    try:
        target_id = payload.target_id.strip()
        if not target_id:
            raise UserInputError("Target ID is required.")
        if payload.sequence_type not in {"cds", "mrna"}:
            raise UserInputError("Sequence type must be CDS or mRNA.")

        _setup_entrez(payload.ncbi_email, payload.ncbi_api_key, require_email=False)
        fasta_text = _fetch_fasta(target_id, payload.sequence_type)
        sequences = _extract_sequences(fasta_text)
        total = sum(_compute_probe_count(seq, payload.homopolymer_max) for seq in sequences)
        return JSONResponse(
            {
                "count": total,
                "records": len(sequences),
            }
        )
    except UserInputError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/run", response_class=HTMLResponse)
def run(
    request: Request,
    output_base_dir: str = Form(...),
    project_name: str = Form(""),
    target_id: str = Form(...),
    sequence_type: str = Form("cds"),
    amplifier: str = Form("B1"),
    homopolymer_max: int = Form(...),
    desired_pairs: int | None = Form(None),
    gc_range: str | None = Form(None),
    poly_at_max: int | None = Form(None),
    poly_gc_max: int | None = Form(None),
    five_prime_delay: int | None = Form(None),
    blast_ref: str | None = Form(None),
    soft_min_pairs: int | None = Form(None),
    ncbi_email: str = Form(""),
    ncbi_api_key: str = Form(""),
):
    try:
        output_dir = _normalize_output_dir(output_base_dir, project_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        _setup_entrez(ncbi_email, ncbi_api_key, require_email=True)
        fasta_text = _fetch_fasta(target_id.strip(), sequence_type)

        fasta_path = output_dir / "input.fasta"
        fasta_path.write_text(fasta_text)

        result = _run_hcr(
            amplifier.strip().upper(),
            fasta_path,
            output_dir,
            homopolymer_max,
            desired_pairs,
            gc_range.strip() if gc_range else None,
            poly_at_max,
            poly_gc_max,
            five_prime_delay,
            blast_ref.strip() if blast_ref else None,
            soft_min_pairs,
        )

        log_path = output_dir / "hcr_run.log"
        log_path.write_text(result.stdout + "\n" + result.stderr)

        zip_base = output_dir / "hcr_output"
        zip_path = Path(shutil.make_archive(os.fspath(zip_base), "zip", root_dir=output_dir))

        job_id = str(uuid4())
        JOBS[job_id] = {
            "zip_path": os.fspath(zip_path),
            "output_dir": os.fspath(output_dir),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "job_id": job_id,
                "output_dir": os.fspath(output_dir),
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )
    except UserInputError as exc:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(exc)},
            status_code=400,
        )


@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return HTMLResponse("Unknown job id.", status_code=404)
    return FileResponse(
        job["zip_path"],
        filename=Path(job["zip_path"]).name,
        media_type="application/zip",
    )
