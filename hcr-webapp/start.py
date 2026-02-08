from __future__ import annotations

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


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    if not _in_venv():
        if not VENV.exists():
            _run([sys.executable, "-m", "venv", str(VENV)])
        py = _venv_python()
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        _run([str(py), str(ROOT / "start.py"), "--run"])
        return 0

    if "--run" in sys.argv:
        _run([
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ])
        return 0

    print("Run: python3 hcr-webapp/start.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
