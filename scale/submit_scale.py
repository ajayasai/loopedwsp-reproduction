"""Bundle the frozen transformer and scale benchmark for an offline private Kaggle GPU."""

import argparse
import base64
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submit import ACCELERATOR, KAGGLE, OWNER, cli_executable


def remote_run(payload, hashes):
    import base64
    import hashlib
    import json
    import os
    import subprocess
    import sys
    import tempfile
    import traceback
    import zlib
    from pathlib import Path

    out = Path("/kaggle/working")
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path("/kaggle/temp")
    scratch.mkdir(parents=True, exist_ok=True)
    for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[option] = "4"
    runtime = {"input_sha256": hashes}
    try:
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("Scale evaluation requires CUDA")
        runtime.update(device=torch.cuda.get_device_name(0), pytorch=str(torch.__version__),
                       cuda=torch.version.cuda)
        with tempfile.TemporaryDirectory(dir=scratch) as temp_dir:
            stage = Path(temp_dir)
            for name, encoded in payload.items():
                data = zlib.decompress(base64.b64decode(encoded))
                if hashlib.sha256(data).hexdigest() != hashes[name]:
                    raise ValueError(f"Bundle checksum mismatch for {name}")
                (stage / name).write_bytes(data)
            print(f"Running on {runtime['device']} with {len(hashes)} verified inputs", flush=True)
            with (out / "scale-driver.log").open("w", encoding="utf-8") as log:
                command = [sys.executable, "-u", str(stage / "run_scale.py")]
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      cwd=out, text=True, encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    exit_code = proc.wait()
                if exit_code:
                    raise subprocess.CalledProcessError(exit_code, command)
        for side in (60, 100):
            json.loads((out / f"scale-{side}.json").read_text(encoding="utf-8"))
        runtime["status"] = "complete"
    except BaseException:
        runtime.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (out / "scale-runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")


def package(extra_files):
    folder = KAGGLE / "packages/scale"
    folder.mkdir(parents=True, exist_ok=True)
    src = ROOT / "src/maze"
    results = ROOT / "results/maze"
    freeze = json.loads((results / "freeze.json").read_text(encoding="utf-8"))
    all_sources = {p.name: p.read_text(encoding="utf-8") for p in src.glob("*.py")}
    digest = hashlib.sha256(json.dumps(all_sources, sort_keys=True).encode("ascii")).hexdigest()
    if digest != freeze["source_bundle_sha256"]:
        raise ValueError("Maze source bundle differs from the model freeze")
    files = [src / name for name in ("looped.py", "weights.py", "evaluate.py")]
    files += [ROOT / "scale" / name for name in ("corpus.py", "run_scale.py", "corpus-lock.json")]
    files += [results / name for name in ("weights.npz", "freeze.json")]
    files += extra_files
    if len({p.name for p in files}) != len(files):
        raise ValueError("Each bundled file must have a distinct name")
    contents = {p.name: p.read_bytes() for p in files}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    encoded = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in contents.items()}
    script = inspect.getsource(remote_run) + f"\nremote_run({encoded!r}, {hashes!r})\n"
    compile(script, "main.py", "exec")
    (folder / "main.py").write_text(script, encoding="ascii", newline="\n")
    meta = {
        "id": f"{OWNER}/loopedwsp-independent-maze-scale", "title": "LoopedWSP Independent Maze Scale",
        "code_file": "main.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")
    return folder, meta, hashes, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--corpus-file", action="append", type=Path, default=[])
    parser.add_argument("--timeout", type=int, default=7200, help="Run limit in seconds")
    args = parser.parse_args()
    folder, meta, hashes, frozen_digest = package(args.corpus_file)
    print(json.dumps(hashes, indent=2), flush=True)
    if not args.push:
        return
    command = [cli_executable(), "kernels", "push", "-p", str(folder),
               "--accelerator", ACCELERATOR, "-t", str(args.timeout)]
    pushed = subprocess.run(command, cwd=KAGGLE, env={**os.environ, "PYTHONUTF8": "1"},
                            capture_output=True, text=True, errors="replace")
    output = pushed.stdout + pushed.stderr
    print(output, flush=True)
    match = re.search(r"Kernel version (\d+) successfully pushed", output)
    if pushed.returncode or not match:
        raise SystemExit(pushed.returncode or "Kaggle did not confirm the submission")
    version = int(match.group(1))
    run_dir = KAGGLE / "runs/scale" / f"v{version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "kernel-metadata.json"):
        shutil.copy2(folder / name, run_dir / name)
    receipt = {
        "kernel": meta["id"], "version": version, "requested_private": meta["is_private"],
        "requested_internet": meta["enable_internet"], "requested_gpu": ACCELERATOR,
        "url": f"https://www.kaggle.com/code/{meta['id']}/versions/{version}",
        "maze_source_bundle_sha256": frozen_digest, "input_sha256": hashes,
        "script_sha256": hashlib.sha256((folder / "main.py").read_bytes()).hexdigest(),
        "timeout_seconds": args.timeout,
    }
    (run_dir / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Saved submission receipt at {run_dir}", flush=True)


if __name__ == "__main__":
    main()
