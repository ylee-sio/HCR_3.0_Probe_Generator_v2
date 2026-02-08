from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import contextlib
from pathlib import Path
import random
from uuid import uuid4

from fastapi import FastAPI, Form, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from Bio import Entrez
from Bio import SeqIO
import pandas as pd


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
BATCHES: dict[str, dict] = {}
BATCH_PROGRESS: dict[str, dict] = {}


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


def _parse_targets_from_dataframe(df) -> list[dict[str, str]]:
    if df is None or df.empty:
        raise UserInputError("No rows found in the input file.")
    if df.shape[1] < 2:
        raise UserInputError("Input file must have at least two columns.")
    rows = []
    for _, row in df.iterrows():
        accession = str(row.iloc[0]).strip()
        name = str(row.iloc[1]).strip()
        if not accession or accession.lower() == "nan":
            continue
        if not name or name.lower() == "nan":
            name = accession
        rows.append({"accession": accession, "name": name})
    if not rows:
        raise UserInputError("No valid target rows found.")
    return rows


def _write_targets_csv(targets: list[dict[str, str]], dest: Path) -> None:
    df = pd.DataFrame(targets)
    df.to_csv(dest, index=False)


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
    import time
    try:
        from urllib.error import HTTPError
    except Exception:
        HTTPError = Exception

    nuccore_id = _resolve_nuccore_id(identifier)
    rettype = "fasta_cds_na" if seq_kind == "cds" else "fasta"

    retries = 3
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(db="nuccore", id=nuccore_id, rettype=rettype, retmode="text")
            data = handle.read()
            handle.close()
            if not data.strip():
                raise UserInputError("NCBI returned empty sequence data for this ID.")
            return data
        except HTTPError as exc:
            if getattr(exc, "code", None) == 429 and attempt < retries - 1:
                time.sleep(1 + attempt)
                continue
            raise


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


def _resolve_batch_targets(batch: dict) -> list[dict]:
    targets = batch["targets"]
    options = batch.get("options", {})
    batch_opts = options.get("batch", {})
    per_target = options.get("per_target", [])
    resolved = []
    for idx, target in enumerate(targets):
        row_opts = per_target[idx] if idx < len(per_target) else {}
        def pick(key):
            if batch_opts.get("apply_to_all"):
                return batch_opts.get(key)
            return row_opts.get(key) or batch_opts.get(key)

        resolved.append(
            {
                "id": f"t{idx}",
                "accession": target["accession"],
                "name": target["name"],
                "sequence_type": pick("sequence_type"),
                "homopolymer_max": pick("homopolymer_max"),
                "amplifier": pick("amplifier"),
                "desired_pairs": pick("desired_pairs"),
                "gc_range": pick("gc_range"),
                "poly_at_max": pick("poly_at_max"),
                "poly_gc_max": pick("poly_gc_max"),
                "five_prime_delay": pick("five_prime_delay"),
                "soft_min_pairs": pick("soft_min_pairs"),
                "blast_ref": pick("blast_ref"),
            }
        )
    return resolved


def _parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _find_opool_file(target_dir: Path) -> Path | None:
    opool_dir = target_dir / "ProbemakerOut" / "OPOOL"
    if not opool_dir.exists():
        return None
    files = sorted(opool_dir.glob("*oPool.xlsx"))
    return files[0] if files else None


def _generate_pool_output(batch: dict, pools: dict, progress: dict | None = None) -> list[dict]:
    _setup_entrez(None, None, require_email=False)
    output_dir = Path(batch["output_dir"])
    targets = {t["id"]: t for t in _resolve_batch_targets(batch)}

    total_targets = sum(len(ids) for ids in pools.values())
    if progress is not None:
        progress["total"] = total_targets
        progress["done"] = 0
        progress["status"] = "running"

    results = []
    for pool_id, target_ids in pools.items():
        rand_id = f"{random.randint(0, 99999999):08d}"
        pool_folder = output_dir / f"HL_POOL_{rand_id}"
        pool_folder.mkdir(parents=True, exist_ok=True)

        pooled_rows = []
        mapping_lines = []
        row_cursor = 1
        mapping_lines.append(f"Pool ID: HL_POOL_{rand_id}")

        for tid in target_ids:
            target = targets.get(tid)
            if not target:
                continue
            if not target.get("sequence_type") or target.get("homopolymer_max") is None or not target.get("amplifier"):
                raise UserInputError(
                    f"Missing required options for target {target['accession']} ({target['name']})."
                )
            target_label = f"{target['accession']}_{target['name']}"
            target_dir = pool_folder / target_label
            target_dir.mkdir(parents=True, exist_ok=True)

            fasta_text = _fetch_fasta(target["accession"], target["sequence_type"] or "cds")
            fasta_path = target_dir / "input.fasta"
            fasta_path.write_text(fasta_text)

            homopolymer_max = _parse_optional_int(target.get("homopolymer_max")) or 0
            desired_pairs = _parse_optional_int(target.get("desired_pairs"))
            poly_at = _parse_optional_int(target.get("poly_at_max"))
            poly_gc = _parse_optional_int(target.get("poly_gc_max"))
            delay = _parse_optional_int(target.get("five_prime_delay"))
            soft_min = _parse_optional_int(target.get("soft_min_pairs"))
            gc_range = target.get("gc_range")
            blast_ref = target.get("blast_ref")
            amp = (target.get("amplifier") or "B1").upper()

            result = _run_hcr(
                amp,
                fasta_path,
                target_dir,
                homopolymer_max,
                desired_pairs,
                gc_range,
                poly_at,
                poly_gc,
                delay,
                blast_ref,
                soft_min,
            )
            (target_dir / "hcr_run.log").write_text(result.stdout + "\n" + result.stderr)

            opool_file = _find_opool_file(target_dir)
            if not opool_file:
                mapping_lines.append(f"{target_label}: No oPool file found")
                if progress is not None:
                    progress["done"] += 1
                continue

            df = pd.read_excel(opool_file)
            if "Pool name" in df.columns:
                df = df[["Pool name", "Sequence"]]
            else:
                df = df.iloc[:, :2]
                df.columns = ["Pool name", "Sequence"]

            df["Pool name"] = f"HL_POOL_{rand_id}"

            start_row = row_cursor
            end_row = row_cursor + len(df) - 1
            mapping_lines.append(
                f"{target_label}: rows {start_row}-{end_row} (Amplifier {amp})"
            )
            row_cursor = end_row + 1
            pooled_rows.append(df)
            if progress is not None:
                progress["done"] += 1

        if pooled_rows:
            pooled_df = pd.concat(pooled_rows, ignore_index=True)
            pooled_path = pool_folder / f"HL_POOL_{rand_id}_opool.xlsx"
            pooled_df.to_excel(pooled_path, index=False)
        else:
            pooled_path = None

        mapping_path = pool_folder / "pool_mapping.txt"
        mapping_path.write_text("\n".join(mapping_lines))

        results.append(
            {
                "pool_id": f"HL_POOL_{rand_id}",
                "path": str(pool_folder),
                "opool": str(pooled_path) if pooled_path else None,
                "mapping": str(mapping_path),
            }
        )

    if progress is not None:
        progress["status"] = "done"
    return results

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/single", response_class=HTMLResponse)
def single(request: Request):
    return templates.TemplateResponse("single.html", {"request": request})


@app.get("/batch", response_class=HTMLResponse)
def batch_start(request: Request):
    return templates.TemplateResponse("batch_start.html", {"request": request})


@app.post("/batch/prepare")
async def batch_prepare(
    request: Request,
    output_base_dir: str = Form(...),
    project_name: str = Form(""),
    upload_file: UploadFile | None = File(None),
    manual_accessions: list[str] | None = Form(None),
    manual_names: list[str] | None = Form(None),
):
    try:
        output_dir = _normalize_output_dir(output_base_dir, project_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        targets: list[dict[str, str]] = []
        source_csv: Path | None = None

        if upload_file and upload_file.filename:
            content = await upload_file.read()
            suffix = Path(upload_file.filename).suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(io.BytesIO(content))
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(io.BytesIO(content))
            else:
                raise UserInputError("Upload must be a CSV or XLSX file.")
            targets = _parse_targets_from_dataframe(df)
            source_csv = output_dir / "batch_targets.csv"
            _write_targets_csv(targets, source_csv)
        else:
            if not manual_accessions:
                raise UserInputError("Provide a CSV/XLSX file or enter targets manually.")
            manual_rows = []
            for idx, accession in enumerate(manual_accessions):
                accession = accession.strip()
                if not accession:
                    continue
                name = ""
                if manual_names and idx < len(manual_names):
                    name = manual_names[idx].strip()
                if not name:
                    name = accession
                manual_rows.append({"accession": accession, "name": name})
            if not manual_rows:
                raise UserInputError("No valid manual targets provided.")
            targets = manual_rows
            source_csv = output_dir / "batch_targets.csv"
            _write_targets_csv(targets, source_csv)

        batch_id = str(uuid4())
        BATCHES[batch_id] = {
            "output_dir": os.fspath(output_dir),
            "targets": targets,
            "source_csv": os.fspath(source_csv) if source_csv else None,
            "options": {},
            "pools": {},
        }

        return JSONResponse({"batch_id": batch_id})
    except UserInputError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/batch/edit/{batch_id}", response_class=HTMLResponse)
def batch_edit(request: Request, batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        return HTMLResponse("Unknown batch id.", status_code=404)
    options = batch.get("options", {})
    return templates.TemplateResponse(
        "batch_edit.html",
        {
            "request": request,
            "batch_id": batch_id,
            "targets": batch["targets"],
            "options": options,
        },
    )


@app.post("/batch/edit/{batch_id}")
async def batch_edit_save(request: Request, batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        return JSONResponse({"error": "Unknown batch id."}, status_code=404)
    form = await request.form()

    apply_to_all = form.get("apply_to_all") == "on"
    batch_options = {
        "sequence_type": form.get("batch_sequence_type") or None,
        "homopolymer_max": form.get("batch_homopolymer_max") or None,
        "amplifier": form.get("batch_amplifier") or None,
        "desired_pairs": form.get("batch_desired_pairs") or None,
        "gc_range": form.get("batch_gc_range") or None,
        "poly_at_max": form.get("batch_poly_at_max") or None,
        "poly_gc_max": form.get("batch_poly_gc_max") or None,
        "five_prime_delay": form.get("batch_five_prime_delay") or None,
        "soft_min_pairs": form.get("batch_soft_min_pairs") or None,
        "blast_ref": form.get("batch_blast_ref") or None,
        "apply_to_all": apply_to_all,
    }

    per_target = []
    targets = batch["targets"]
    for idx, target in enumerate(targets):
        prefix = f"target_{idx}_"
        per_target.append(
            {
                "sequence_type": form.get(prefix + "sequence_type") or None,
                "homopolymer_max": form.get(prefix + "homopolymer_max") or None,
                "amplifier": form.get(prefix + "amplifier") or None,
                "desired_pairs": form.get(prefix + "desired_pairs") or None,
                "gc_range": form.get(prefix + "gc_range") or None,
                "poly_at_max": form.get(prefix + "poly_at_max") or None,
                "poly_gc_max": form.get(prefix + "poly_gc_max") or None,
                "five_prime_delay": form.get(prefix + "five_prime_delay") or None,
                "soft_min_pairs": form.get(prefix + "soft_min_pairs") or None,
                "blast_ref": form.get(prefix + "blast_ref") or None,
            }
        )

    batch["options"] = {"batch": batch_options, "per_target": per_target}
    return JSONResponse({"batch_id": batch_id})


@app.get("/batch/pools/{batch_id}", response_class=HTMLResponse)
def batch_pools(request: Request, batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        return HTMLResponse("Unknown batch id.", status_code=404)
    targets = _resolve_batch_targets(batch)
    return templates.TemplateResponse(
        "batch_pools.html",
        {
            "request": request,
            "batch_id": batch_id,
            "targets": targets,
        },
    )


def _run_batch_generation(batch_id: str, pools: dict):
    batch = BATCHES.get(batch_id)
    if not batch:
        return
    progress = BATCH_PROGRESS.get(batch_id)
    try:
        results = _generate_pool_output(batch, pools, progress)
        batch["results"] = results
        if progress is not None:
            progress["status"] = "done"
    except Exception as exc:
        if progress is not None:
            progress["status"] = "error"
            progress["error"] = str(exc)


@app.post("/batch/generate/{batch_id}")
async def batch_generate(request: Request, batch_id: str, background_tasks: BackgroundTasks):
    batch = BATCHES.get(batch_id)
    if not batch:
        return JSONResponse({"error": "Unknown batch id."}, status_code=404)
    payload = await request.json()
    pools = payload.get("pools", {})
    if not pools:
        return JSONResponse({"error": "No pools provided."}, status_code=400)
    batch["pools"] = pools
    BATCH_PROGRESS[batch_id] = {"status": "queued", "total": 0, "done": 0, "error": None}
    background_tasks.add_task(_run_batch_generation, batch_id, pools)
    return JSONResponse({"status": "started"})


@app.get("/batch/results/{batch_id}", response_class=HTMLResponse)
def batch_results(request: Request, batch_id: str):
    batch = BATCHES.get(batch_id)
    if not batch:
        return HTMLResponse("Unknown batch id.", status_code=404)
    return templates.TemplateResponse(
        "batch_results.html",
        {
            "request": request,
            "batch_id": batch_id,
            "output_dir": batch.get("output_dir"),
            "results": batch.get("results", []),
        },
    )


@app.get("/batch/progress/{batch_id}")
def batch_progress(batch_id: str):
    progress = BATCH_PROGRESS.get(batch_id)
    if not progress:
        return JSONResponse({"status": "unknown"}, status_code=404)
    return JSONResponse(progress)



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
            "single.html",
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
