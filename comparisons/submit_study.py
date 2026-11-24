"""Package, run and verify the private Kaggle GPU loop comparison."""

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zlib


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from puzzles.submit_puzzles import remote_run
from submit import ACCELERATOR, KAGGLE, OWNER, cli_executable


NAME = "loop-comparisons"
TITLE = "LoopedWSP Loop Comparisons"
KERNEL = f"{OWNER}/loopedwsp-loop-comparisons"
TASKS = ("inventory", "scheduling", "telemetry")
OUTPUTS = ["comparison-results.json", "comparison-predictions.npz"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def build():
    paths = {name: HERE/name for name in
             ("experiment.py", "baselines.py", "recurrent.py", "challenge.npz", "protocol.json")}
    application_files = ["cases.npz", "corpus-lock.json"]
    application_files.extend(f"models/{task}/{name}" for task in TASKS for name in
                             ("network.json", "weights.npz", "freeze.json"))
    paths.update({f"applications/{name}": ROOT/"applications"/name for name in application_files})
    contents = {name: path.read_bytes() for name, path in paths.items()}
    for name, data in contents.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        elif name.endswith(".json"):
            json.loads(data)
    checksums = {name: sha(data) for name, data in contents.items()}
    payload = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in contents.items()}
    code = inspect.getsource(remote_run) + f"\nremote_run({payload!r}, {checksums!r}, {OUTPUTS!r}, 'comparison')\n"
    compile(code, "main.py", "exec")
    folder = KAGGLE/"packages"/NAME
    folder.mkdir(parents=True, exist_ok=True)
    script = code.encode("ascii")
    (folder/"main.py").write_bytes(script)
    meta = {"id": KERNEL, "title": TITLE, "code_file": "main.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "kernel_sources": [], "competition_sources": []}
    write_json(folder/"kernel-metadata.json", meta)
    print(f"Packaged {len(checksums)} inputs; script SHA-256 {sha(script)}", flush=True)
    return folder, checksums, sha(script)


def cli(args):
    cmd = [cli_executable(), "kernels", *args]
    result = subprocess.run(cmd, cwd=KAGGLE, env={**os.environ, "PYTHONUTF8": "1"},
                            capture_output=True, text=True, errors="replace")
    output = result.stdout + result.stderr
    print(output, flush=True)
    result.check_returncode()
    return output


def submit(folder, inputs, script_sha, timeout):
    output = cli(["push", "-p", str(folder), "--accelerator", ACCELERATOR, "-t", str(timeout)])
    match = re.search(r"Kernel version (\d+) successfully pushed", output)
    if not match:
        raise RuntimeError("Kaggle did not acknowledge a version. Check server status before another submission.")
    version = int(match.group(1))
    run = KAGGLE/"runs"/NAME/f"v{version}"
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder/"kernel-metadata.json", run/"kernel-metadata.json")
    receipt = {"version": version, "kernel": KERNEL, "submitted_utc": datetime.now(timezone.utc).isoformat(),
               "requested_gpu": ACCELERATOR, "requested_private": True,
               "requested_internet": False, "input_sha256": inputs, "script_sha256": script_sha,
               "timeout_seconds": timeout, "url": f"https://www.kaggle.com/code/{KERNEL}/versions/{version}"}
    write_json(run/"submission.json", receipt)
    print(f"Saved submission receipt at {run}", flush=True)


def pull_source(receipt):
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = receipt["kernel"].split("/")
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(req)
    meta = response.metadata
    if (meta.current_version_number != receipt["version"] or meta.ref != receipt["kernel"]
            or not meta.is_private or not meta.enable_gpu or meta.enable_tpu or meta.enable_internet
            or meta.machine_shape != receipt["requested_gpu"]):
        raise ValueError("Server version or kernel settings differ from the saved receipt")
    source = response.blob.source.encode("utf-8")
    if sha(source) != receipt["script_sha256"]:
        raise ValueError("The downloaded source differs from the submitted script")
    call = ast.parse(source).body[-1].value
    payload, hashes, outputs, stage = (ast.literal_eval(arg) for arg in call.args)
    checked = {name: sha(zlib.decompress(base64.b64decode(data, validate=True)))
               for name, data in payload.items()}
    if checked != hashes or hashes != receipt["input_sha256"] or outputs != OUTPUTS or stage != "comparison":
        raise ValueError("The server-side payload differs from the expected bundle")
    return {"id": meta.ref, "id_no": meta.id, "version": meta.current_version_number, "title": meta.title,
            "is_private": meta.is_private, "enable_gpu": meta.enable_gpu, "enable_internet": meta.enable_internet,
            "enable_tpu": meta.enable_tpu, "machine_shape": meta.machine_shape,
            "docker_image": meta.docker_image, "downloaded_source_sha256": sha(source)}


def collect(version):
    from kaggle import api
    import numpy as np

    run = KAGGLE/"runs"/NAME/f"v{version}"
    receipt = json.loads((run/"submission.json").read_bytes())
    if receipt["version"] != version or receipt["kernel"] != KERNEL:
        raise ValueError("Version folder and receipt disagree")
    status = str(api.kernels_status(KERNEL).status).split(".")[-1]
    if status != "COMPLETE":
        raise RuntimeError(f"Kaggle status is {status}")
    meta = pull_source(receipt)
    with tempfile.TemporaryDirectory(dir=run) as temp:
        download = Path(temp)
        cli(["output", KERNEL, "-p", str(download)])
        runtime = json.loads((download/"comparison-runtime.json").read_bytes())
        hashes = {name: sha((download/name).read_bytes()) for name in OUTPUTS}
        if (runtime["status"] != "complete" or "T4" not in runtime["device"]
                or runtime["input_sha256"] != receipt["input_sha256"] or runtime["output_sha256"] != hashes):
            raise ValueError("GPU runtime or output files differ from the receipt")
        json.loads((download/OUTPUTS[0]).read_bytes())
        with np.load(download/OUTPUTS[1], allow_pickle=False) as archive:
            shapes = {k: list(archive[k].shape) for k in archive.files}
        if not shapes:
            raise ValueError("Empty prediction archive")
        output_dir = run/"output"
        output_dir.mkdir(exist_ok=True)
        for file in download.iterdir():
            if file.is_file():
                shutil.copy2(file, output_dir/file.name)
    remote = run/"remote"
    remote.mkdir(exist_ok=True)
    write_json(remote/"kernel-metadata.json", meta)
    summary = {"kernel": KERNEL, "version": version, "status": status,
               "device": runtime["device"], "cuda": runtime["cuda"], "pytorch": runtime["pytorch"],
               "server_private": meta["is_private"], "server_internet_enabled": meta["enable_internet"],
               "script_sha256": meta["downloaded_source_sha256"],
               "input_files_verified": len(receipt["input_sha256"]), "output_sha256": hashes,
               "prediction_arrays": shapes}
    write_json(run/"verification.json", summary)
    print(f"Verified {len(shapes)} prediction arrays and archived {run}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    actions = p.add_mutually_exclusive_group()
    actions.add_argument("--push", action="store_true", help="Upload and run on a private Kaggle GPU")
    actions.add_argument("--collect", type=int, metavar="VERSION", help="Download and verify a completed version")
    p.add_argument("--timeout", type=int, default=7200, help="GPU session limit in seconds")
    args = p.parse_args()
    if args.collect is not None:
        collect(args.collect)
        return
    folder, checksums, digest = build()
    if args.push:
        submit(folder, checksums, digest, args.timeout)


if __name__ == "__main__":
    main()
