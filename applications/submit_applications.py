"""Submit a private, offline Kaggle run with a frozen copy of every test input."""
import argparse
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from puzzles.submit_puzzles import remote_run
from submit import ACCELERATOR, KAGGLE, OWNER, cli_executable

NAME = "practical-applications"
KERNEL = f"{OWNER}/loopedwsp-practical-applications"
TASKS = ("inventory", "scheduling", "telemetry")
OUTPUTS = ["application-results.json", "predictions.npz"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    folder = KAGGLE / "packages" / NAME
    folder.mkdir(parents=True, exist_ok=True)
    files = ["experiment.py", "corpus-lock.json", "cases.npz"] + [f"{task}.py" for task in TASKS]
    files += [f"models/{task}/{item}" for task in TASKS for item in ("network.json", "weights.npz", "freeze.json")]
    contents = {name: (HERE / name).read_bytes() for name in files}
    contents["frame.py"] = (ROOT / "lengths/frame.py").read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    archive = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in contents.items()}
    source = inspect.getsource(remote_run) + f"\nremote_run({archive!r}, {hashes!r}, {OUTPUTS!r}, 'application')\n"
    compile(source, "main.py", "exec")
    script = folder / "main.py"
    script.write_bytes(source.encode("ascii"))
    metadata = {"id": KERNEL, "title": "LoopedWSP Practical Applications", "code_file": script.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": [], "kernel_sources": [], "competition_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Packaged {len(hashes)} inputs in {script.stat().st_size:,} bytes", flush=True)
    if not args.push:
        return
    timeout = 3600
    proc = subprocess.run([cli_executable(), "kernels", "push", "-p", str(folder),
                           "--accelerator", ACCELERATOR, "-t", str(timeout)],
                          cwd=KAGGLE, env={**os.environ, "PYTHONUTF8":"1"},
                          capture_output=True, text=True, errors="replace")
    message = proc.stdout + proc.stderr
    print(message, flush=True)
    match = re.search(r"Kernel version (\d+) successfully pushed", message)
    if proc.returncode or not match:
        raise SystemExit(proc.returncode or "Kaggle did not confirm submission")
    version = int(match.group(1))
    run = KAGGLE / "runs" / NAME / f"v{version}"
    run.mkdir(parents=True, exist_ok=True)
    for file in (script, folder / "kernel-metadata.json"):
        shutil.copy2(file, run / file.name)
    receipt = {"kernel": KERNEL, "version": version, "requested_gpu": ACCELERATOR,
               "requested_private": True, "requested_internet": False, "input_sha256": hashes,
               "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(), "timeout_seconds": timeout,
               "url": f"https://www.kaggle.com/code/{KERNEL}/versions/{version}"}
    (run / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
