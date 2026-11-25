"""Authenticate the archived script, server metadata and outputs of a private Kaggle study."""
import ast
import base64
from hashlib import sha256
import json
import zlib


def require(condition, issue):
    if not condition:
        raise ValueError(issue)


def read(path):
    return json.loads(path.read_bytes())


def digest(blob):
    return sha256(blob).hexdigest()


def authenticate_run(run, results, *, folder, inputs, outputs, stage_name):
    receipt = read(run / "submission.json")
    requested = read(run / "kernel-metadata.json")
    remote = read(run / "remote/kernel-metadata.json")
    verified = read(run / "verification.json")
    runtime = read(run / "output" / f"{stage_name}-runtime.json")
    kernel, version = receipt["kernel"], receipt["version"]
    require(kernel == requested["id"] == remote["id"] == verified["kernel"]
            and version == remote["version"] == verified["version"] and run.name == f"v{version}"
            and receipt["url"] == f"https://www.kaggle.com/code/{kernel}/versions/{version}", "Kaggle identity differs")
    require(receipt["requested_private"] is verified["server_private"] is True
            and receipt["requested_internet"] is verified["server_internet_enabled"] is False
            and all(meta["is_private"] is meta["enable_gpu"] is True
                    and meta["enable_tpu"] is meta["enable_internet"] is False for meta in [requested, remote])
            and requested["kernel_type"] == "script" and requested["code_file"] == "main.py"
            and receipt["requested_gpu"] == remote["machine_shape"] == "NvidiaTeslaT4", "Kaggle settings differ")
    require(runtime["status"] == "complete" and verified["status"] == "COMPLETE"
            and results["split"] == "test" and runtime["available_devices"] >= 1
            and runtime["device"] == verified["device"] == results["device"] and "T4" in runtime["device"]
            and runtime["pytorch"] == verified["pytorch"] == results["torch"]
            and runtime["cuda"] == verified["cuda"] == results["cuda"], "Reported device or completion differs")
    script = (run / "main.py").read_bytes()
    script_hash = digest(script)
    require(script_hash == receipt["script_sha256"] == verified["script_sha256"] and len(script) == receipt["script_bytes"],
            "Archived script differs from the submission")
    server_hash = remote["downloaded_source_sha256"]
    require(server_hash == verified["remote_script_sha256"] and remote["source_match"] == verified["source_match"],
            "Server source/verification mismatch")
    if verified["source_match"] == "exact":
        require(script_hash == server_hash, "Server source differs from submitted script")
    else:
        normalized = script.replace(b"\r\n", b"\n")
        require(verified["source_match"] == "line_endings" and server_hash != script_hash and
                server_hash in {digest(normalized), digest(normalized.replace(b"\n", b"\r\n"))},
                "Server source differs beyond line endings")
    last = ast.parse(script).body[-1].value
    require(isinstance(last, ast.Call) and isinstance(last.func, ast.Name) and last.func.id == "remote_run"
            and len(last.args) == 4 and not last.keywords, "Unexpected Kaggle entry point")
    payload, declared, output_names, stage = (ast.literal_eval(a) for a in last.args)
    hashes = {name: digest((folder / name).read_bytes()) for name in inputs}
    embedded = {name: digest(zlib.decompress(base64.b64decode(blob, validate=True))) for name, blob in payload.items()}
    require(stage == stage_name and output_names == list(outputs) and verified["input_files_verified"] == len(inputs)
            and embedded == declared == hashes == receipt["input_sha256"] == runtime["input_sha256"],
            "Submitted input bytes or output declaration differ")
    output_hashes = {name: digest((run / "output" / name).read_bytes()) for name in outputs}
    require(output_hashes == verified["output_sha256"] == runtime["output_sha256"], "GPU output hash mismatch")
    return {"kernel": kernel, "version": version, "url": receipt["url"], "device": runtime["device"],
            "pytorch": runtime["pytorch"], "cuda": runtime["cuda"], "status": "COMPLETE",
            "source_match": remote["source_match"], "submitted_sha256": script_hash, "server_sha256": server_hash,
            "input_sha256": hashes, "output_sha256": output_hashes}, verified["prediction_arrays"]
