from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def _venv_python() -> Path:
    if sys.platform.startswith("win"):
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> int:
    if not _in_venv():
        if not VENV.exists():
            _run([sys.executable, "-m", "venv", str(VENV)])
        py = _venv_python()
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        _run([str(py), str(ROOT / "start.py"), "--run"], cwd=ROOT)
        return 0

    if "--run" in sys.argv:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--app-dir",
                str(ROOT),
            ],
            check=True,
            cwd=ROOT,
            env=env,
        )
        return 0

    print("Run: python3 hcr-webapp/start.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
