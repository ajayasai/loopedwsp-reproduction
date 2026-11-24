"""Package, optionally submit, and record the private Kaggle GPU experiment."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KAGGLE = ROOT / "kaggle"
OWNER = os.environ.get("LOOPEDWSP_KAGGLE_OWNER", "ajayasaithambi")
ACCELERATOR = "NvidiaTeslaT4"


def build(split):
    folder = KAGGLE / "packages" / split
    folder.mkdir(parents=True, exist_ok=True)
    sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "src/maze").glob("*.py"))
    }
    if "evaluate.py" not in sources:
        raise FileNotFoundError("The experiment requires src/maze/evaluate.py")
    bundle = json.dumps(sources, ensure_ascii=True, sort_keys=True)
    digest = hashlib.sha256(bundle.encode("ascii")).hexdigest()
    runner = (KAGGLE / "runner.py").read_text(encoding="utf-8")
    script = runner + f"\nrun({split!r}, json.loads({bundle!r}), {digest!r})\n"
    compile(script, "main.py", "exec")
    (folder / "main.py").write_text(script, encoding="ascii", newline="\n")
    title = f"LoopedWSP Independent Maze {split.title()}"
    metadata = {
        "id": f"{OWNER}/{title.lower().replace(' ', '-')}", "title": title,
        "code_file": "main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )
    return folder, metadata, digest


def cli_executable():
    cli = shutil.which("kaggle")
    if cli:
        return cli
    roaming = os.environ.get("APPDATA")
    if roaming:
        version = f"Python{sys.version_info.major}{sys.version_info.minor}"
        candidate = Path(roaming) / "Python" / version / "Scripts/kaggle.exe"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("Install the authenticated Kaggle CLI or add it to PATH")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["dev", "test"])
    parser.add_argument("--push", action="store_true", help="Submit and run this bundle")
    parser.add_argument("--timeout", type=int, default=3600, help="Kaggle session timeout in seconds")
    args = parser.parse_args()
    folder, metadata, digest = build(args.split)
    print(f"Package {folder}\nSource SHA-256 {digest}", flush=True)
    if not args.push:
        return
    command = [cli_executable(), "kernels", "push", "-p", str(folder),
               "--accelerator", ACCELERATOR, "-t", str(args.timeout)]
    environment = {**os.environ, "PYTHONUTF8": "1"}
    pushed = subprocess.run(command, cwd=KAGGLE, env=environment,
                            capture_output=True, text=True, errors="replace")
    message = pushed.stdout + pushed.stderr
    print(message, flush=True)
    match = re.search(r"Kernel version (\d+) successfully pushed", message)
    if pushed.returncode or not match:
        raise SystemExit(pushed.returncode or "Kaggle did not confirm the submission")
    version = int(match.group(1))
    run_dir = KAGGLE / "runs" / args.split / f"v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "kernel": metadata["id"], "version": version,
        "url": f"https://www.kaggle.com/code/{metadata['id']}/versions/{version}",
        "requested_private": metadata["is_private"], "accelerator": ACCELERATOR,
        "timeout_seconds": args.timeout, "source_bundle_sha256": digest,
        "script_sha256": hashlib.sha256((folder / "main.py").read_bytes()).hexdigest(),
    }
    (run_dir / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(folder / "kernel-metadata.json", run_dir)
    shutil.copy2(folder / "main.py", run_dir)
    print(f"Saved submission receipt at {run_dir}", flush=True)


if __name__ == "__main__":
    main()
