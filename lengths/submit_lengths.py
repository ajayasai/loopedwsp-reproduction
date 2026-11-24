"""Package frozen length-generalization models for a private offline Kaggle GPU run."""

import argparse
import ast
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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from submit import ACCELERATOR, KAGGLE, OWNER, cli_executable


TASKS = ("graph", "parity", "chains")
EXPERIMENT = "length-generalization"
INPUTS = [f"{p}.py" for p in (*TASKS, "frame", "experiment")] + ["corpus-lock.json"]
INPUTS += [f"models/{task}/{name}" for task in TASKS
           for name in ("network.json", "weights.npz", "freeze.json")]


def remote_run(payload, checksums):
    import base64
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import tempfile
    import traceback
    import zlib

    output = Path("/kaggle/working")
    scratch = Path("/kaggle/temp")
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[option] = "4"
    manifest = {"input_sha256": checksums, "status": "starting"}
    try:
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("The length-generalization experiment requires CUDA")
        manifest.update(device=torch.cuda.get_device_name(0), pytorch=str(torch.__version__),
                        cuda=torch.version.cuda, available_devices=torch.cuda.device_count())
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            stage = Path(tmp)
            for name, encoded in payload.items():
                path = stage / name
                data = zlib.decompress(base64.b64decode(encoded, validate=True))
                if hashlib.sha256(data).hexdigest() != checksums[name]:
                    raise ValueError(f"Bundle checksum mismatch for {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            print(f"Running on {manifest['device']} with {len(checksums)} verified inputs", flush=True)
            command = [sys.executable, "-u", str(stage / "experiment.py"), "gpu"]
            with (output / "length-driver.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=output, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    rc = proc.wait()
                if rc:
                    raise subprocess.CalledProcessError(rc, command)
        results = {}
        for task in ("graph", "parity", "chains"):
            filename = f"length-{task}.json"
            data = (output / filename).read_bytes()
            json.loads(data)
            results[filename] = hashlib.sha256(data).hexdigest()
        manifest.update(status="complete", output_sha256=results)
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (output / "length-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def package():
    source = ROOT / "lengths"
    contents = {name: (source / name).read_bytes() for name in INPUTS}
    for name, data in contents.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        elif name.endswith(".json"):
            json.loads(data)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    payload = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in contents.items()}
    for name, encoded in payload.items():
        if zlib.decompress(base64.b64decode(encoded, validate=True)) != contents[name]:
            raise ValueError(f"Cannot round-trip {name}")
    script = inspect.getsource(remote_run) + f"\nremote_run({payload!r}, {hashes!r})\n"
    compile(script, "main.py", "exec")
    folder = KAGGLE / "packages" / EXPERIMENT
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": f"{OWNER}/loopedwsp-length-generalization", "title": "LoopedWSP Length Generalization",
        "code_file": "main.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    }
    (folder / "main.py").write_text(script, encoding="ascii", newline="\n")
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    return folder, metadata, hashes


def verify_outputs(version):
    run = KAGGLE / "runs" / EXPERIMENT / f"v{version}"
    read = lambda path: json.loads(path.read_bytes())
    digest = lambda data: hashlib.sha256(data).hexdigest()
    receipt = read(run / "submission.json")
    meta = read(run / "remote/kernel-metadata.json")
    local = (run / "main.py").read_bytes()
    remote = (run / "remote" / meta["code_file"]).read_bytes()
    if digest(local) != receipt["script_sha256"] or remote.replace(b"\r\n", b"\n") != local:
        raise ValueError("The saved Kaggle script differs from the pushed version")
    if (meta["id"] != receipt["kernel"] or meta["is_private"] is not True or
            meta["enable_gpu"] is not True or meta["enable_internet"] is not False or
            meta["machine_shape"] != receipt["requested_gpu"]):
        raise ValueError("Remote Kaggle metadata differs from the requested configuration")
    # Read the frozen bytes directly from the submitted script for versioned replay.
    call = ast.parse(local).body[-1].value
    data = {k: zlib.decompress(base64.b64decode(v, validate=True)) for k, v in ast.literal_eval(call.args[0]).items()}
    if {n: digest(x) for n, x in data.items()} != receipt["input_sha256"]:
        raise ValueError("Submitted input hashes differ from the local receipt")
    out = run / "output"
    runtime = read(out / "length-runtime.json")
    if (runtime["status"] != "complete" or "T4" not in runtime["device"] or
            runtime["input_sha256"] != receipt["input_sha256"]):
        raise ValueError("Kaggle runtime manifest did not confirm the expected successful GPU execution")
    lock = json.loads(data["corpus-lock.json"])
    counts = {}
    for task in TASKS:
        name = f"length-{task}.json"
        raw = (out / name).read_bytes()
        report = json.loads(raw)
        freeze = json.loads(data[f"models/{task}/freeze.json"])
        by_size = {str(row["size"]): row["corpus"] for row in report["rows"]}
        if (digest(raw) != runtime["output_sha256"][name] or by_size != lock[task] or
                len(by_size) != len(report["rows"]) or report["gpu"] != runtime["device"] or
                not report["cpu_gpu_dev_parity"] or report["model_sha256"] != freeze["model_sha256"]):
            raise ValueError(f"Frozen model, corpus, CPU parity or result hashes failed verification for {task}")
        counts[task] = {k: sum(row[k] for row in report["rows"]) for k in ("correct", "count")}
    summary = {"version": version, "kernel": meta["id"], "server_private": meta["is_private"],
               "server_internet_enabled": meta["enable_internet"], "server_machine_shape": meta["machine_shape"],
               "actual_device": runtime["device"], "script_sha256_lf": digest(local),
               "remote_script_sha256_as_downloaded": digest(remote), "runtime_sha256": digest((out / "length-runtime.json").read_bytes()),
               "verified_input_files": len(data), "output_sha256": runtime["output_sha256"], "scores": counts}
    (run / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--verify", metavar="VERSION", type=int, help="Verify a downloaded private run")
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum remote run time in seconds")
    args = parser.parse_args()
    if args.verify is not None:
        if args.push:
            parser.error("Choose one of --push and --verify")
        verify_outputs(args.verify)
        return
    folder, metadata, hashes = package()
    script_hash = hashlib.sha256((folder / "main.py").read_bytes()).hexdigest()
    print(f"Verified {len(hashes)} inputs and prepared {folder}; script SHA-256 {script_hash}", flush=True)
    if not args.push:
        return
    command = [cli_executable(), "kernels", "push", "-p", str(folder),
               "--accelerator", ACCELERATOR, "-t", str(args.timeout)]
    proc = subprocess.run(command, cwd=KAGGLE, env={**os.environ, "PYTHONUTF8": "1"},
                          capture_output=True, text=True, errors="replace")
    response = proc.stdout + proc.stderr
    print(response, flush=True)
    match = re.search(r"Kernel version (\d+) successfully pushed", response)
    if proc.returncode or not match:
        raise SystemExit(proc.returncode or "Kaggle did not confirm the submission")
    version = int(match.group(1))
    run = KAGGLE / "runs" / EXPERIMENT / f"v{version}"
    run.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "kernel-metadata.json"):
        shutil.copy2(folder / name, run / name)
    receipt = {"kernel": metadata["id"], "version": version, "requested_private": True,
               "requested_internet": False, "requested_gpu": ACCELERATOR, "input_sha256": hashes,
               "script_sha256": script_hash, "timeout_seconds": args.timeout,
               "url": f"https://www.kaggle.com/code/{metadata['id']}/versions/{version}"}
    (run / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {run / 'submission.json'}", flush=True)


if __name__ == "__main__":
    main()
