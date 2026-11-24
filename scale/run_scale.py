"""Evaluate the archived maze transformer at two larger sizes on one CUDA GPU."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from corpus import get_corpus
from evaluate import weight_summary
from looped import gpu_forward
from weights import build_weights


SIZES = {60: 50, 100: 40}
HERE = Path(__file__).resolve().parent


def load_frozen(path, freeze):
    """Read the actual evaluated tensors and verify the archive before extending positions."""
    with np.load(path, allow_pickle=False) as archive:
        frozen = {name: archive[name] for name in ("token", "row", "col", "bias", "readout")}
        for name in ("chars", "outputs"):
            frozen[name] = archive[name].item()
        frozen["layers"] = []
        for i in range(2):
            prefix = f"layers/{i}/"
            frozen["layers"].append({key[len(prefix):]: archive[key] for key in archive if key.startswith(prefix)})
    if weight_summary(frozen)["tensor_sha256"] != freeze["tensor_sha256"]:
        raise ValueError("Frozen tensor archive does not match the reviewed model")
    return frozen


def extend_positions(frozen, side):
    expanded = dict(frozen)
    formula = build_weights(side)
    for key in ("row", "col"):
        original = frozen[key]
        table = formula[key]
        np.testing.assert_array_equal(table[:len(original)], original, strict=True)
        expanded[key] = table
    return expanded


def digest_grids(grids):
    return hashlib.sha256("".join(grid + "\n" for grid in grids).encode("ascii")).hexdigest()


def encode(grids, alphabet):
    lut = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(alphabet):
        lut[ord(ch)] = i
    rows = [np.frombuffer(grid.encode("ascii"), dtype=np.uint8) for grid in grids]
    ids = lut[np.stack(rows)]
    if (ids < 0).any():
        raise ValueError("Corpus contains an unknown symbol")
    return ids


def evaluate_size(side, frozen, freeze, lock):
    import torch

    count = SIZES[side]
    questions, targets, corpus_info = get_corpus(side)
    if len(questions) != count or len(targets) != count or len(set(questions)) != count:
        raise ValueError("Unexpected corpus count or duplicated inputs")
    if any(len(q) != side**2 or len(a) != side**2 for q, a in zip(questions, targets)):
        raise ValueError("Unexpected grid dimensions")
    hashes = {"questions_sha256": digest_grids(questions), "targets_sha256": digest_grids(targets)}
    if hashes != lock[str(side)]:
        raise ValueError("Corpus differs from pre-submission hashes")

    w = extend_positions(frozen, side)
    ids = encode(questions, w["chars"])
    labels = encode(targets, w["outputs"])
    budget = 2 * side**2
    batch_size = 8
    diagnostics = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    predictions = gpu_forward(w, ids, [budget], dtype="float64", batch_size=batch_size, diagnostics=diagnostics)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    actual = predictions[budget]
    if actual.dtype != np.int64 or actual.shape != labels.shape:
        raise ValueError("Expected one predicted symbol per input cell")
    correct = np.all(actual == labels, axis=1)
    return {
        "side": side, "count": count, "exact_grid_matches": int(correct.sum()),
        "exact_grid_accuracy": float(correct.mean()), "inference_seconds": elapsed,
        "amortized_seconds_per_maze": elapsed / count,
        "device": torch.cuda.get_device_name(), "dtype": "float64", "batch_size": batch_size,
        "loop_budget": budget, "diagnostics": diagnostics,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda": torch.version.cuda, "torch": str(torch.__version__),
        "frozen_model_sha256": freeze["tensor_sha256"], "expanded_model": weight_summary(w),
        "extended_arrays": ["row", "col"], "corpus": corpus_info, "locked_hashes": hashes,
    }


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze = json.loads((HERE / "freeze.json").read_text(encoding="utf-8"))
    lock = json.loads((HERE / "corpus-lock.json").read_text(encoding="utf-8"))
    frozen = load_frozen(HERE / "weights.npz", freeze)
    for side in SIZES:
        print(f"Evaluating {SIZES[side]} grids of size {side} on {torch.cuda.get_device_name()}", flush=True)
        report = evaluate_size(side, frozen, freeze, lock)
        path = Path.cwd() / f"scale-{side}.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{side}x{side}: {report['exact_grid_matches']}/{report['count']}, "
              f"{report['inference_seconds']:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
