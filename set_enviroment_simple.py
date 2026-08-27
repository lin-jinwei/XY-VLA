import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_NAME = "xy-vla"
PYTHON_VERSION = "3.10"

def run_cmd(cmd, desc):
    print(f"\n[RUN] {desc}")
    print("      " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")

def env_exists(conda_exe, env_name):
    result = subprocess.run(
        [conda_exe, "env", "list"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        raise RuntimeError("Unable to query conda environments.")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return any(line.split()[0] == env_name for line in lines if not line.startswith("#"))

def main():
    parser = argparse.ArgumentParser(
        description="One-click setup for X-VLA conda environment."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the environment if it already exists.",
    )
    args = parser.parse_args()

    conda_exe = shutil.which("conda")
    if not conda_exe:
        raise RuntimeError("conda not found in PATH. Please install/enable Anaconda first.")

    req_file = PROJECT_ROOT / "X-VLA" / "requirements.txt"
    if not req_file.exists():
        alt_req = PROJECT_ROOT / "requirements.txt"
        if alt_req.exists():
            req_file = alt_req
        else:
            raise RuntimeError("requirements.txt not found in project.")

    print("=== X-VLA Environment Setup ===")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Conda env    : {ENV_NAME}")
    print(f"Python       : {PYTHON_VERSION}")
    print(f"Requirements : {req_file}")

    exists = env_exists(conda_exe, ENV_NAME)
    if exists and args.recreate:
        run_cmd([conda_exe, "env", "remove", "-n", ENV_NAME, "-y"], "Remove old env")
        exists = False

    if not exists:
        run_cmd(
            [conda_exe, "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"],
            "Create conda env",
        )
    else:
        print(f"\n[SKIP] Environment '{ENV_NAME}' already exists.")

    run_cmd(
        [
            conda_exe,
            "run",
            "-n",
            ENV_NAME,
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
        ],
        "Install PyTorch (CPU)",
    )

    run_cmd(
        [conda_exe, "run", "-n", ENV_NAME, "pip", "install", "-r", str(req_file)],
        "Install project requirements",
    )

    run_cmd(
        [
            conda_exe,
            "run",
            "-n",
            ENV_NAME,
            "python",
            "-c",
            "import sys,torch; print('python', sys.version.split()[0]); print('torch', torch.__version__)",
        ],
        "Verify environment",
    )

    print("\nDONE: Environment is ready.")
    print(f"Use it with: conda activate {ENV_NAME}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
