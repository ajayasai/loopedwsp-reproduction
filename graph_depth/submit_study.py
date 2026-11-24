"""Package the frozen graph-depth study or submit and collect its private Kaggle run."""

import argparse
import ast
import base64
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zlib


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from comparisons.submit_study import cli, sha, write_json
from puzzles.submit_puzzles import remote_run
from submit import ACCELERATOR, KAGGLE, OWNER


STAGE = "graph-depth"
TITLE = "LoopedWSP Pointer Reachability"
KERNEL = f"{OWNER}/loopedwsp-pointer-reachability"
INPUTS = ("experiment.py", "pointer.py", "reachability.py", "corpus.py", "protocol.json", "cases-manifest.json")
OUTPUTS = ["graph-depth-results.json", "graph-depth-predictions.npz"]


def build():
    contents = {name: (HERE / name).read_bytes() for name in INPUTS}
    for name, data in contents.items():
        if name.endswith(".py"):
            compile(data, name, "exec")
        elif name.endswith(".json"):
            json.loads(data)
    hashes = {name: sha(data) for name, data in contents.items()}
    payload = {name: base64.b64encode(zlib.compress(data, 9)).decode("ascii")
               for name, data in contents.items()}
    body = f"\nremote_run({payload!r}, {hashes!r}, {OUTPUTS!r}, {STAGE!r})\n"
    source = (inspect.getsource(remote_run) + body).encode("ascii")
    compile(source, "main.py", "exec")
    if len(source) >= 1_000_000:
        raise ValueError(f"The {len(source):,}-byte script exceeds the 1 MB Kaggle source-size preflight.")
    folder = KAGGLE / "packages" / STAGE
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "main.py").write_bytes(source)
    metadata = {"id": KERNEL, "title": TITLE, "code_file": "main.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": False, "dataset_sources": [], "kernel_sources": [], "competition_sources": []}
    write_json(folder / "kernel-metadata.json", metadata)
    print(f"Packaged {len(hashes)} frozen inputs in {len(source):,} bytes; script SHA-256 {sha(source)}", flush=True)
    return folder, hashes


def push(folder, hashes, timeout):
    source = (folder / "main.py").read_bytes()
    started = datetime.now(timezone.utc).isoformat()
    output = cli(["push", "-p", str(folder), "--accelerator", ACCELERATOR, "-t", str(timeout)])
    match = re.search(r"Kernel version (\d+) successfully pushed", output)
    if not match:
        raise RuntimeError("Kaggle did not acknowledge a version. Check the server before submitting again.")
    version = int(match.group(1))
    run = KAGGLE / "runs" / STAGE / f"v{version}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "main.py").write_bytes(source)
    shutil.copyfile(folder / "kernel-metadata.json", run / "kernel-metadata.json")
    receipt = {"kernel": KERNEL, "version": version, "submitted_utc": started, "requested_gpu": ACCELERATOR,
               "requested_private": True, "requested_internet": False, "input_sha256": hashes,
               "script_sha256": sha(source), "script_bytes": len(source), "timeout_seconds": timeout,
               "url": f"https://www.kaggle.com/code/{KERNEL}/versions/{version}"}
    write_json(run / "submission.json", receipt)
    print(f"Archived the submitted script and receipt in {run}", flush=True)


def pull_metadata(receipt, submitted):
    from kaggle import api
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = KERNEL.split("/")
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(req)
    meta = response.metadata
    if (meta.ref != KERNEL or meta.current_version_number != receipt["version"] or not meta.is_private
            or not meta.enable_gpu or meta.enable_internet or meta.enable_tpu or meta.machine_shape != ACCELERATOR):
        raise ValueError("Kaggle settings or version differ from the saved receipt")
    source = response.blob.source.encode("utf-8")
    if sha(submitted) != receipt["script_sha256"]:
        raise ValueError("The archived submission differs from the source named in its receipt")
    exact = submitted == source
    if not exact and submitted.replace(b"\r\n", b"\n") != source.replace(b"\r\n", b"\n"):
        raise ValueError("Kaggle's downloaded source differs from the submitted script")
    call = ast.parse(source).body[-1].value
    payload, hashes, outputs, stage = (ast.literal_eval(arg) for arg in call.args)
    decoded = {name: sha(zlib.decompress(base64.b64decode(blob, validate=True))) for name, blob in payload.items()}
    if decoded != hashes or hashes != receipt["input_sha256"] or outputs != OUTPUTS or stage != STAGE:
        raise ValueError("Kaggle's embedded files differ from the frozen bundle")
    return {"id": meta.ref, "id_no": meta.id, "version": meta.current_version_number, "title": meta.title,
            "is_private": meta.is_private, "enable_gpu": meta.enable_gpu, "enable_tpu": meta.enable_tpu,
            "enable_internet": meta.enable_internet, "machine_shape": meta.machine_shape, "docker_image": meta.docker_image,
            "downloaded_source_sha256": sha(source), "source_match": "exact" if exact else "line_endings"}


def collect(version):
    from kaggle import api
    import numpy as np

    run = KAGGLE / "runs" / STAGE / f"v{version}"
    receipt = json.loads((run / "submission.json").read_bytes())
    if receipt["kernel"] != KERNEL or receipt["version"] != version:
        raise ValueError("The receipt does not match the requested run")
    status = str(api.kernels_status(KERNEL).status).split(".")[-1]
    if status != "COMPLETE":
        raise RuntimeError(f"Kaggle status is {status}")
    submitted = (run / "main.py").read_bytes()
    with tempfile.TemporaryDirectory(dir=run) as name:
        temp = Path(name)
        cli(["output", KERNEL, "-p", str(temp)])
        meta = pull_metadata(receipt, submitted)
        runtime = json.loads((temp / f"{STAGE}-runtime.json").read_bytes())
        output_hashes = {key: sha((temp / key).read_bytes()) for key in OUTPUTS}
        if (runtime["status"] != "complete" or "T4" not in runtime["device"]
                or runtime["input_sha256"] != receipt["input_sha256"] or runtime["output_sha256"] != output_hashes):
            raise ValueError("Kaggle's GPU runtime or input/output digest does not match")
        results = json.loads((temp / OUTPUTS[0]).read_bytes())
        with np.load(temp / OUTPUTS[1], allow_pickle=False) as archive:
            shapes = {key: list(archive[key].shape) for key in archive.files}
        if not results or not shapes:
            raise ValueError("The results or prediction archive is empty")
        output = run / "output"
        output.mkdir(exist_ok=True)
        for path in temp.iterdir():
            if path.is_file():
                shutil.copyfile(path, output / path.name)
    remote = run / "remote"
    remote.mkdir(exist_ok=True)
    write_json(remote / "kernel-metadata.json", meta)
    manifest = {"kernel": KERNEL, "version": version, "status": status, "device": runtime["device"],
                "cuda": runtime["cuda"], "pytorch": runtime["pytorch"], "server_private": meta["is_private"],
                "server_internet_enabled": meta["enable_internet"], "script_sha256": receipt["script_sha256"],
                "remote_script_sha256": meta["downloaded_source_sha256"], "source_match": meta["source_match"],
                "input_files_verified": len(receipt["input_sha256"]), "output_sha256": output_hashes,
                "prediction_arrays": shapes}
    write_json(run / "verification.json", manifest)
    print(f"Verified {len(shapes)} prediction arrays and archived {run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--push", action="store_true", help="Submit one private GPU run")
    actions.add_argument("--collect", type=int, metavar="VERSION", help="Download and verify a completed version")
    parser.add_argument("--timeout", type=int, default=7200, help="Kaggle GPU session limit in seconds")
    args = parser.parse_args()
    if args.collect is not None:
        collect(args.collect)
        return
    folder, hashes = build()
    if args.push:
        push(folder, hashes, args.timeout)


if __name__ == "__main__":
    main()
