from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(port: int) -> None:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    subprocess.run(cmd, check=False)


def main() -> int:
    port = _find_free_port()
    server_thread = threading.Thread(target=_run_server, args=(port,), daemon=True)
    server_thread.start()

    # Give the server a moment to start.
    time.sleep(1.0)

    url = f"http://127.0.0.1:{port}"
    try:
        import webview  # type: ignore

        icon_path = os.path.join(os.path.dirname(__file__), "..", "assets", "dna.png")
        webview.create_window("HCR 3.0 PROBEGEN V2", url, width=1200, height=800, icon=icon_path)
        webview.start()
    except Exception:
        webbrowser.open(url)
        server_thread.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
