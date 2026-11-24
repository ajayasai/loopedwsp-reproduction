"""Evaluate one official Maze-Hard split in FP64 on a CUDA device."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time


SIDE = 30
DATASET = "sapientinc/maze-30x30-hard-1k"
REVISION = "549754de3d67eebe57b6b22c7d226b9412d55b07"
TRAIN_SHA256 = "25f9e39b5aee16967da41ba63b2d1d75cf608c2360d8a1419d827f9859dbafa0"


def tensor_items(value, name=""):
    """Yield named weight tensors in a stable order across processes."""
    import numpy as np

    if isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_items(value[key], f"{name}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_items(item, f"{name}/{index}")
    elif isinstance(value, np.ndarray):
        yield name, value


def weight_summary(weights):
    import numpy as np

    digest = hashlib.sha256()
    total = nonzero = 0
    for name, value in tensor_items(weights):
        arr = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        header = json.dumps([name, arr.dtype.str, arr.shape], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(arr.tobytes())
        total += arr.size
        nonzero += np.count_nonzero(arr)
    if not total:
        raise ValueError("The model contains no weight tensors")
    return {"tensor_sha256": digest.hexdigest(), "parameters": int(total), "nonzero_parameters": int(nonzero)}


def source_path(root, split):
    filename = "train.csv" if split == "dev" else "test.csv"
    path = root / "data" / "maze" / filename
    if path.is_file():
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=DATASET, repo_type="dataset", revision=REVISION, filename=filename))


def load_split(path, chars, outputs):
    import numpy as np

    def lookup(alphabet):
        table = np.full(256, -1, dtype=np.int64)
        for index, symbol in enumerate(alphabet):
            table[ord(symbol)] = index
        return table

    input_table, output_table = lookup(chars), lookup(outputs)
    questions, answers = [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            q, a = row["question"], row["answer"]
            if len(q) != SIDE**2 or len(a) != SIDE**2:
                raise ValueError("Unexpected official grid dimensions")
            questions.append(input_table[np.frombuffer(q.encode("ascii"), dtype=np.uint8)])
            answers.append(output_table[np.frombuffer(a.encode("ascii"), dtype=np.uint8)])
    if len(questions) != 1000:
        raise ValueError("Expected 1,000 official mazes")
    inputs, labels = np.stack(questions), np.stack(answers)
    if np.any(inputs < 0) or np.any(labels < 0):
        raise ValueError("Unexpected character in the official dataset")
    return inputs, labels


def grade(predictions, targets, checkpoints):
    import numpy as np

    reference = predictions[checkpoints[0]]
    all_stable = np.ones(len(targets), dtype=bool)
    summary = {}
    for rounds in checkpoints:
        result = np.asarray(predictions[rounds])
        if result.shape != targets.shape or result.dtype != np.int64:
            raise ValueError("The runner must return one integer prediction per grid cell")
        correct = np.all(result == targets, axis=1)
        stable = np.all(result == reference, axis=1)
        all_stable &= stable
        summary[str(rounds)] = {
            "solved": int(correct.sum()),
            "exact_grid_accuracy": float(correct.mean()),
            "same_grid_as_first_checkpoint": int(stable.sum()),
        }
    return {
        "grids": len(targets),
        "by_checkpoint": summary,
        "identical_grids_across_all_checkpoints": int(all_stable.sum()),
        "all_readouts_fixed": bool(all_stable.all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("dev", "test"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing data/ and results/.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("Maze-Hard evaluation requires an active CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if __package__:
        from .looped import gpu_forward
        from .weights import build_weights
    else:
        from looped import gpu_forward
        from weights import build_weights

    weights = build_weights(SIDE)
    model = weight_summary(weights)
    output_dir = args.root / "results" / "maze"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{args.split}.json"
    if args.split == "test":
        if destination.exists():
            raise FileExistsError("Official TEST results already exist in this run directory")
        # Obtain the one-shot marker before looking up or opening TEST. A failed
        # run keeps its marker so that the reason can be reviewed before a retry.
        with (output_dir / "test.started").open("x", encoding="utf-8") as marker:
            marker.write(model["tensor_sha256"] + "\n")

    data = source_path(args.root, args.split)
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    if args.split == "dev" and data_hash != TRAIN_SHA256:
        raise ValueError("The development CSV differs from the audited split")
    ids, labels = load_split(data, weights["chars"], weights["outputs"])
    n = ids.shape[1]
    checkpoints = (2 * n, 4 * n, 8 * n)
    torch.cuda.synchronize()
    start = time.perf_counter()
    info = {}
    with torch.inference_mode():
        predictions = gpu_forward(weights, ids, checkpoints, dtype="float64", batch_size=args.batch_size,
                                  diagnostics=info)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if args.split == "test":
        batches = info.pop("batches")
        info.update({
            "batch_count": len(batches),
            "batches_with_exact_full_state_fixed_point": sum(b["full_state_fixed"] for b in batches),
            "sum_of_evaluated_blocks_across_batches": sum(b["evaluated_blocks"] for b in batches),
            "max_blocks_evaluated_in_a_batch": max(b["evaluated_blocks"] for b in batches),
        })
    report = {
        "split": args.split,
        "dataset": DATASET,
        "revision": REVISION,
        "source_sha256": data_hash,
        "model": model,
        "loop_checkpoints": list(checkpoints),
        "dtype": "float64",
        "allow_tf32": False,
        "device": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "inference_seconds": elapsed,
        "diagnostics": info,
        **grade(predictions, labels, checkpoints),
    }
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    tmp = destination.with_suffix(".json.tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(destination)
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
