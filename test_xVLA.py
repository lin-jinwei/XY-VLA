import json
import subprocess
import sys

import numpy as np
import requests
import json_numpy
from transformers import AutoModel, AutoProcessor

from xvla_local_server import MODEL_DIR, find_free_port, start_xvla_server_process


REQUIRED_FILES = [
    "model.safetensors",
    "config.json",
    "configuration_xvla.py",
    "modeling_xvla.py",
    "processing_xvla.py",
    "tokenizer.json",
    "vocab.json",
    "preprocessor_config.json",
]


def check_model_files() -> None:
    print("== [1/4] Checking required model files ==")
    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).exists()]
    if missing:
        raise RuntimeError("Missing files:\n- " + "\n- ".join(missing))
    size_gb = (MODEL_DIR / "model.safetensors").stat().st_size / (1024 ** 3)
    print(f"Required files: OK, model.safetensors={size_gb:.2f} GB")


def check_local_loading() -> None:
    print("== [2/4] Loading model and processor locally ==")
    _ = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, local_files_only=True
    )
    _ = AutoProcessor.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, local_files_only=True
    )
    print("Local loading: OK")


def infer_once() -> None:
    print("== [3/4] Starting temporary local inference server ==")
    port = find_free_port()
    proc = start_xvla_server_process(
        "127.0.0.1",
        port,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        wait_timeout_s=120.0,
    )
    try:
        print(f"Server start: OK (port {port})")

        print("== [4/4] Sending minimal /act request ==")
        payload = {
            "proprio": json_numpy.dumps(np.zeros(20, dtype=np.float32)),
            "language_instruction": "Move the gripper to target position",
            "image0": json_numpy.dumps(np.zeros((256, 256, 3), dtype=np.uint8)),
            "domain_id": 0,
            "steps": 2,
        }
        r = requests.post(f"http://127.0.0.1:{port}/act", json=payload, timeout=90)
        if r.status_code >= 400:
            raise RuntimeError(f"/act failed {r.status_code}: {r.text[:500]}")
        data = r.json()
        if "action" not in data:
            raise RuntimeError(f"/act response has no 'action' field: {json.dumps(data)[:300]}")
        print("Inference request: OK")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> None:
    print("X-VLA one-click test started")
    print(f"Model directory: {MODEL_DIR}")
    check_model_files()
    check_local_loading()
    infer_once()
    print("\nPASS: X-VLA model is complete and runnable on this machine.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFAIL: {exc}")
        sys.exit(1)
