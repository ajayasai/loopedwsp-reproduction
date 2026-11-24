"""Stage a single benchmark split and run the bundled code on a Kaggle GPU."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def run_logged(command, cwd, log_file):
    with log_file.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def run(split, sources, source_digest):
    output = Path("/kaggle/working/results/maze")
    output.mkdir(parents=True, exist_ok=True)
    run_record = {
        "split": split,
        "is_private_requested": True,
        "accelerator_requested": "NvidiaTeslaT4",
        "dataset": "sapientinc/maze-30x30-hard-1k",
        "revision": "549754de3d67eebe57b6b22c7d226b9412d55b07",
        "source_bundle_sha256": source_digest,
    }
    scratch = Path("/kaggle/temp")
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        root = Path(directory)
        results = root / "results/maze"
        results.mkdir(parents=True)
        try:
            import torch
            from huggingface_hub import hf_hub_download

            torch.set_num_threads(4)
            if not torch.cuda.is_available():
                raise RuntimeError("The requested CUDA GPU is unavailable")
            run_record["gpu"] = torch.cuda.get_device_name(0)
            run_record["cuda"] = torch.version.cuda
            run_record["pytorch"] = str(torch.__version__)
            print(json.dumps(run_record), flush=True)
            for name, content in sources.items():
                target = root / "src/maze" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8", newline="\n")
            backend_log = output / "backend.log"
            run_logged([sys.executable, "-u", "src/maze/check_backend.py"], root, backend_log)
            reports = [
                line for line in backend_log.read_text(encoding="utf-8").splitlines()
                if line.startswith("{") and '"backend_check"' in line
            ]
            if not reports or json.loads(reports[-1]).get("backend_check") != "passed":
                raise RuntimeError("Backend comparison did not report a passing result")
            (output / "backend.json").write_text(reports[-1] + "\n", encoding="utf-8")
            filename = {"dev": "train.csv", "test": "test.csv"}[split]
            dataset_file = Path(hf_hub_download(
                repo_id=run_record["dataset"], repo_type="dataset",
                filename=filename, revision=run_record["revision"],
                cache_dir=str(root / "hf-cache"),
            ))
            data = root / "data/maze" / filename
            data.parent.mkdir(parents=True)
            shutil.copyfile(dataset_file, data)
            run_record["dataset_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
            command = [sys.executable, "-u", "src/maze/evaluate.py", "--split", split]
            run_logged(command, root, output / "evaluation.log")
            if not any(path.is_file() for path in results.rglob("*")):
                raise RuntimeError("Evaluation exited successfully but saved no result files")
            run_record["status"] = "complete"
        except BaseException:
            run_record["status"] = "failed"
            (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            shutil.copytree(results, output, dirs_exist_ok=True)
            (output / "runtime.json").write_text(
                json.dumps(run_record, indent=2) + "\n", encoding="utf-8"
            )


os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
for setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[setting] = "4"
