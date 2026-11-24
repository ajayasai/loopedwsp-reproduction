"""Package and verify the private, offline Kaggle sequential-puzzle experiment."""

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


NAME = "sequential-puzzles"
TASKS = ("hanoi", "checkers")
INPUTS = ["frame.py", "experiment.py", "verify.py"] + [
    f"models/{task}/{name}" for task in TASKS for name in ("network.json", "weights.npz", "freeze.json")]
OUTPUTS = [f"puzzle-{task}.json" for task in TASKS] + [f"emitted-{task}.npz" for task in TASKS]


def remote_run(payload, checksums, expected, stage_name="puzzle"):
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

    out = Path("/kaggle/working")
    scratch = Path("/kaggle/temp")
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    for option in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[option] = "4"
    manifest = {"input_sha256": checksums, "status": "starting"}
    try:
        import torch
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError(f"The {stage_name} evaluation requires CUDA")
        manifest.update(device=torch.cuda.get_device_name(0), available_devices=torch.cuda.device_count(),
                        pytorch=str(torch.__version__), cuda=torch.version.cuda)
        with tempfile.TemporaryDirectory(dir=scratch) as name:
            stage = Path(name)
            for rel, encoded in payload.items():
                data = zlib.decompress(base64.b64decode(encoded, validate=True))
                if hashlib.sha256(data).hexdigest() != checksums[rel]:
                    raise ValueError(f"Embedded checksum mismatch for {rel}")
                dest = stage / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
            print(f"Running on {manifest['device']} with {len(checksums)} frozen inputs", flush=True)
            command = [sys.executable, "-u", str(stage / "experiment.py"), "gpu"]
            with (out / f"{stage_name}-driver.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace") as proc:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    rc = proc.wait()
                if rc:
                    raise subprocess.CalledProcessError(rc, command)
        hashes = {}
        for name in expected:
            data = (out / name).read_bytes()
            if name.endswith(".json"):
                json.loads(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        manifest.update(status="complete", output_sha256=hashes)
    except BaseException:
        manifest.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        (out / f"{stage_name}-runtime.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package():
    contents = {name: (ROOT / "puzzles" / name).read_bytes() for name in INPUTS}
    for name, data in contents.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        elif name.endswith(".json"):
            json.loads(data)
    hashes = {name: sha(data) for name, data in contents.items()}
    payload = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in contents.items()}
    if {name: zlib.decompress(base64.b64decode(x, validate=True)) for name, x in payload.items()} != contents:
        raise ValueError("Compressed package failed local round-trip validation")
    source = inspect.getsource(remote_run) + f"\nremote_run({payload!r}, {hashes!r}, {OUTPUTS!r})\n"
    compile(source, "main.py", "exec")
    folder = KAGGLE / "packages" / NAME
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main.py").write_text(source, encoding="ascii", newline="\n")
    meta = {"id": f"{OWNER}/loopedwsp-sequential-puzzles", "title": "LoopedWSP Sequential Puzzles",
            "code_file": "main.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [],
            "kernel_sources": [], "competition_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")
    print(f"Verified {len(contents)} inputs; packaged {folder}", flush=True)
    return folder, hashes, meta


def verify(version):
    import numpy as np

    run = KAGGLE / "runs" / NAME / f"v{version}"
    out = run / "output"
    receipt = json.loads((run / "submission.json").read_bytes())
    meta = json.loads((run / "remote/kernel-metadata.json").read_bytes())
    local = (run / "main.py").read_bytes()
    remote = (run / "remote" / meta["code_file"]).read_bytes()
    if (sha(local) != receipt["script_sha256"] or remote.replace(b"\r\n", b"\n") != local or
            meta["id"] != receipt["kernel"] or meta["is_private"] is not True or
            meta["enable_gpu"] is not True or meta["enable_internet"] is not False or
            meta["machine_shape"] != receipt["requested_gpu"]):
        raise ValueError("Pulled remote metadata or source mismatch")
    call = ast.parse(local).body[-1].value
    payload = {k: zlib.decompress(base64.b64decode(v, validate=True)) for k, v in ast.literal_eval(call.args[0]).items()}
    hashes = {k: sha(v) for k, v in payload.items()}
    manifest = json.loads((out / "puzzle-runtime.json").read_bytes())
    if (hashes != receipt["input_sha256"] or hashes != manifest["input_sha256"] or
            manifest["status"] != "complete" or "T4" not in manifest["device"]):
        raise ValueError("Input hashes, CUDA device or completion manifest mismatch")
    result_hashes = {name: sha((out / name).read_bytes()) for name in OUTPUTS}
    if result_hashes != manifest["output_sha256"]:
        raise ValueError("Downloaded files differ from recorded Kaggle output hashes")
    array_shapes, scores = {}, {}
    for task in TASKS:
        report = json.loads((out / f"puzzle-{task}.json").read_bytes())
        freeze = json.loads(payload[f"models/{task}/freeze.json"])
        rows = report["size_rows"]
        if (report["model"]["model_sha256"] != freeze["model_sha256"] or
                report["device"] != manifest["device"] or not report["cpu_literal_gpu_dev_parity"] or
                report["emitted_archive_sha256"] != result_hashes[f"emitted-{task}.npz"]):
            raise ValueError(f"{task} report disagrees with model freeze or runtime evidence")
        with np.load(out / f"emitted-{task}.npz", allow_pickle=False) as z:
            expected = {f"n{r['size']}" for r in rows}
            if expected != set(z.files) or len(expected) != len(rows):
                raise ValueError(f"{task} size rows differ from token archive")
            array_shapes[task] = {}
            for row in rows:
                key = f"n{row['size']}"
                array = z[key]
                digest = sha(np.asarray(array, dtype="<i4").tobytes())
                if array.shape != (row["moves"],) or digest != row["move_ids_sha256"]:
                    raise ValueError(f"{task} token digest or move count mismatch for {key}")
                array_shapes[task][key] = list(array.shape)
        scores[task] = {"sizes": len(rows), "valid_optimal_sizes": sum(r["optimal"] and r["goal"] for r in rows),
                        "sizes_with_eos": sum(bool(r["ended_with_eos"]) for r in rows),
                        "changed_at_extra_checkpoints": sum(r["predictions_changed_at_2R_or_4R"] for r in rows)}
    summary = {"version": version, "kernel": meta["id"], "server_private": meta["is_private"],
               "server_internet_enabled": meta["enable_internet"], "server_machine_shape": meta["machine_shape"],
               "actual_device": manifest["device"], "script_sha256_lf": sha(local),
               "remote_script_sha256_as_downloaded": sha(remote), "runtime_sha256": sha((out / "puzzle-runtime.json").read_bytes()),
               "verified_input_files": len(payload), "output_sha256": result_hashes, "scores": scores,
               "emitted_array_shapes": array_shapes}
    (run / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(payload)} bundled inputs, {len(result_hashes)} outputs, and actual {manifest['device']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--push", action="store_true")
    action.add_argument("--verify", type=int, metavar="VERSION")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.verify is not None:
        verify(args.verify)
        return
    folder, hashes, meta = package()
    if not args.push:
        return
    command = [cli_executable(), "kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR,
               "-t", str(args.timeout)]
    proc = subprocess.run(command, cwd=KAGGLE, env={**os.environ, "PYTHONUTF8": "1"},
                          capture_output=True, text=True, errors="replace")
    message = proc.stdout + proc.stderr
    print(message, flush=True)
    match = re.search(r"Kernel version (\d+) successfully pushed", message)
    if proc.returncode or not match:
        raise SystemExit(proc.returncode or "Kaggle did not confirm the submission")
    version = int(match.group(1))
    run = KAGGLE / "runs" / NAME / f"v{version}"
    run.mkdir(parents=True, exist_ok=True)
    for name in ("main.py", "kernel-metadata.json"):
        shutil.copy2(folder / name, run / name)
    receipt = {"kernel": meta["id"], "version": version, "requested_private": True, "requested_internet": False,
               "requested_gpu": ACCELERATOR, "input_sha256": hashes, "script_sha256": sha((folder / "main.py").read_bytes()),
               "timeout_seconds": args.timeout, "url": f"https://www.kaggle.com/code/{meta['id']}/versions/{version}"}
    (run / "submission.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {run / 'submission.json'}", flush=True)


if __name__ == "__main__":
    main()
