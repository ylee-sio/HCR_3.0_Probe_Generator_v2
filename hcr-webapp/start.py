from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
ASSETS = ROOT / "assets"


def _venv_python() -> Path:
    if sys.platform.startswith("win"):
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _desktop_dir() -> Path:
    return Path.home() / "Desktop"


def _create_windows_shortcut(target: Path, icon: Path) -> None:
    desktop = _desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / "HCR 3.0 PROBEGEN V2.lnk"
    script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{shortcut.as_posix()}')
$Shortcut.TargetPath = '{target.as_posix()}'
$Shortcut.WorkingDirectory = '{ROOT.as_posix()}'
$Shortcut.IconLocation = '{icon.as_posix()}'
$Shortcut.Save()
"""
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=False)


def _create_linux_shortcut(target: Path, icon: Path) -> None:
    desktop = _desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / "HCR_3.0_PROBEGEN_V2.desktop"
    shortcut.write_text(
        "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=HCR 3.0 PROBEGEN V2",
                f"Exec={target}",
                f"Icon={icon}",
                "Terminal=false",
            ]
        )
    )
    shortcut.chmod(0o755)


def _create_macos_shortcut(target: Path, icon: Path) -> None:
    desktop = _desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut = desktop / "HCR 3.0 PROBEGEN V2.command"
    shortcut.write_text(f"#!/usr/bin/env bash\n'{target}'\n")
    shortcut.chmod(0o755)


def _create_shortcut() -> None:
    target = _venv_python()
    icon_png = ASSETS / "dna.png"
    icon_ico = ASSETS / "dna.ico"

    system = platform.system().lower()
    if "windows" in system:
        _create_windows_shortcut(target, icon_ico)
    elif "darwin" in system:
        _create_macos_shortcut(target, icon_png)
    else:
        _create_linux_shortcut(target, icon_png)


def main() -> int:
    if not _in_venv():
        if not VENV.exists():
            _run([sys.executable, "-m", "venv", str(VENV)])
        py = _venv_python()
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
        _create_shortcut()
        _run([str(py), str(ROOT / "start.py"), "--run"])
        return 0

    if "--run" in sys.argv:
        os.environ.setdefault("PYTHONPATH", str(ROOT))
        from app.desktop import main as run_app

        return run_app()

    print("Run: python start.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
