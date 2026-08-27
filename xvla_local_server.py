from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "xVLAModel"

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def wait_port_ready(host: str, port: int, *, timeout_s: float = 120.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(1.0)
    return False

def build_xvla_server_code(host: str, port: int, model_dir: Path | None = None) -> str:
    mpath = model_dir if model_dir is not None else MODEL_DIR
    return (
        "from transformers import AutoModel, AutoProcessor; "
        f"mpath={repr(str(mpath))}; "
        "m=AutoModel.from_pretrained(mpath, trust_remote_code=True, local_files_only=True); "
        "p=AutoProcessor.from_pretrained(mpath, trust_remote_code=True, local_files_only=True); "
        f"m.run(p, host={repr(host)}, port={port})"
    )

def start_xvla_server_process(
    host: str,
    port: int,
    *,
    model_dir: Path | None = None,
    stdout: Any = None,
    stderr: Any = None,
    wait_timeout_s: float = 180.0,
) -> subprocess.Popen:

    code = build_xvla_server_code(host, port, model_dir=model_dir)
    proc = subprocess.Popen([sys.executable, "-u", "-c", code], stdout=stdout, stderr=stderr)
    if not wait_port_ready(host, port, timeout_s=wait_timeout_s):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError("X-VLA server failed to start in time.")
    return proc
