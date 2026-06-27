#!/usr/bin/env python3
"""Detect the host NVIDIA GPU + CUDA Toolkit and install the matching CuPy wheel.

Selection logic (see pyproject.toml comment):
  - No NVIDIA GPU          -> install nothing (base package uses NumPy fallback).
  - GPU + CUDA Toolkit     -> cupy-cuda{MAJOR}x          (uses system CUDA, no bundle)
  - GPU, no CUDA Toolkit   -> cupy-cuda{MAJOR}x[ctk]      (bundles CUDA libs via PyPI)

`{MAJOR}` comes from `nvcc --version` when a toolkit is present, otherwise from
the max CUDA version advertised by `nvidia-smi` (the driver's supported runtime).

Run from anywhere, e.g.:
    uv run python scripts/install_cupy.py
    python scripts/install_cupy.py

Installs into this project's .venv (auto-detected). Uses `uv pip` when available,
otherwise falls back to the venv's own pip. Stdlib-only; no third-party imports.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"
VENV_DIR = REPO_ROOT / ".venv"
VENV_BIN = VENV_DIR / ("Scripts" if IS_WINDOWS else "bin")
VENV_PYTHON = VENV_BIN / ("python.exe" if IS_WINDOWS else "python")

# CUDA major version -> CuPy wheel suffix.
WHEEL_SUFFIX = {11: "cuda11x", 12: "cuda12x", 13: "cuda13x"}
# All CuPy distribution names that provide the `cupy` import package; only one
# may be installed at a time, so we scrub them before installing the chosen one.
CONFLICT_DISTS = ["cupy", "cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"]


def run(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str] | None:
    """Run a command, returning (rc, out, err). None if the executable is absent."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=cwd)
    except FileNotFoundError:
        return None
    return proc.returncode, proc.stdout, proc.stderr


def detect_gpu() -> dict | None:
    """Return {name, driver, cuda_max} from nvidia-smi, or None if no NVIDIA GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    name = driver = "?"
    q = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if q:
        parts = [p.strip() for p in q[1].splitlines()[0].split(",")] if q[1].strip() else []
        if len(parts) >= 2:
            name, driver = parts[0], parts[1]
    cuda_max = None
    full = run(["nvidia-smi"])
    if full:
        m = re.search(r"CUDA Version:\s*(\d+(?:\.\d+)?)", full[1])
        if m:
            cuda_max = m.group(1)
    return {"name": name, "driver": driver, "cuda_max": cuda_max}


def detect_toolkit() -> dict | None:
    """Return {major, version, nvcc} from nvcc, or None if no CUDA Toolkit."""
    nvcc = shutil.which("nvcc")
    if nvcc is None and os.environ.get("CUDA_PATH"):
        # nvcc not on PATH but CUDA_PATH suggests a toolkit install; try it.
        candidate = Path(os.environ["CUDA_PATH"]) / ("bin/nvcc.exe" if IS_WINDOWS else "bin/nvcc")
        if candidate.exists():
            nvcc = str(candidate)
    if nvcc is None:
        return None
    r = run([nvcc, "--version"])
    if not r:
        return None
    m = re.search(r"release\s+(\d+)\.(\d+)", r[1])
    if not m:
        return None
    return {"major": int(m.group(1)), "version": f"{m.group(1)}.{m.group(2)}", "nvcc": nvcc}


def pick_wheel(gpu: dict | None, toolkit: dict | None) -> tuple[str, str]:
    """Return (package_spec, reason) or raise SystemExit on unsupported CUDA."""
    if gpu is None:
        sys.exit(0)  # No GPU -> nothing to do (printed by caller).

    if toolkit:
        major = toolkit["major"]
        suffix = WHEEL_SUFFIX.get(major)
        if suffix is None:
            _fail_unsupported(major)
        return f"cupy-{suffix}", f"CUDA Toolkit {toolkit['version']} present -> {suffix} (uses system CUDA, no bundle)"

    # GPU but no toolkit: infer CUDA major from the driver's max supported CUDA.
    major = None
    if gpu.get("cuda_max"):
        major = int(gpu["cuda_max"].split(".")[0])
    if major is None or major not in WHEEL_SUFFIX:
        major = 12  # safe default; the [ctk] bundle will bring its own CUDA 12 libs
    suffix = WHEEL_SUFFIX[major]
    return f"cupy-{suffix}[ctk]", f"no CUDA Toolkit -> {suffix}[ctk] (bundles CUDA via PyPI)"


def _fail_unsupported(major: int) -> None:
    print(f"\n[!] Detected CUDA major {major}, which has no matching CuPy wheel.")
    print(f"    Supported: {', '.join(f'cuda{m}x' for m in sorted(WHEEL_SUFFIX))}.")
    print("    Install a supported CUDA Toolkit or a matching wheel manually.")
    sys.exit(2)


def install(pkg: str) -> None:
    if not VENV_PYTHON.exists():
        print(f"[!] No project venv found at {VENV_DIR}.")
        print("    Create one first with:  uv venv && uv sync")
        sys.exit(1)
    uv = shutil.which("uv")
    # uv pip targets the .venv in the current directory, so run from REPO_ROOT.
    cwd = str(REPO_ROOT) if uv else None
    if uv:
        print(f"[*] Using uv pip (target {VENV_DIR})")
        uninstall_cmd = [uv, "pip", "uninstall", *CONFLICT_DISTS]
        install_cmd = [uv, "pip", "install", pkg]
    else:
        print(f"[*] uv not found; using venv pip ({VENV_PYTHON})")
        uninstall_cmd = [str(VENV_PYTHON), "-m", "pip", "uninstall", "-y", *CONFLICT_DISTS]
        install_cmd = [str(VENV_PYTHON), "-m", "pip", "install", pkg]

    run(uninstall_cmd, cwd=cwd)  # scrub conflicting CuPy dists (ignore errors)

    print(f"[*] $ {' '.join(install_cmd)}")
    r = run(install_cmd, cwd=cwd)
    if not r or r[0] != 0:
        sys.stderr.write((r[2] if r else "install failed") + "\n")
        print("[!] Installation failed.")
        sys.exit(1)


def verify() -> None:
    r = run([str(VENV_PYTHON), "-c",
             "import cupy; print(cupy.__version__); "
             "print('devices:', cupy.cuda.runtime.getDeviceCount())"])
    if r and r[0] == 0:
        ver, devs = r[1].strip().splitlines()
        print(f"[+] cupy {ver} installed; {devs}")
    else:
        print("[!] cupy import failed after install:")
        sys.stderr.write((r[2] if r else "no result") + "\n")
        sys.exit(1)


def main() -> None:
    print("=" * 72)
    print("CuPy wheel installer")
    print(f"  repo : {REPO_ROOT}")
    print(f"  venv : {VENV_DIR}")
    print("=" * 72)

    gpu = detect_gpu()
    toolkit = detect_toolkit()

    print("\n--- detection ---")
    if gpu:
        print(f"  GPU          : {gpu['name']}")
        print(f"  driver       : {gpu['driver']}")
        print(f"  driver CUDA  : {gpu['cuda_max']}")
    else:
        print("  GPU          : none (nvidia-smi not found)")
    if toolkit:
        print(f"  CUDA Toolkit : {toolkit['version']}  ({toolkit['nvcc']})")
    else:
        print("  CUDA Toolkit : none (nvcc not found)")

    print("\n--- decision ---")
    if gpu is None:
        print("  No NVIDIA GPU detected -> nothing to install.")
        print("  The base package runs on the NumPy fallback; no action needed.")
        return

    pkg, reason = pick_wheel(gpu, toolkit)
    print(f"  {reason}")
    print(f"  install      : {pkg}")

    print("\n--- install ---")
    install(pkg)

    print("\n--- verify ---")
    verify()
    print("\n[done] CuPy is ready.")


if __name__ == "__main__":
    main()