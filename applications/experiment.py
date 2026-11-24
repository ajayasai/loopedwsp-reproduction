"""Freeze application models, build the benchmark, and run the offline GPU test."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lengths"))
from frame import Transformer, inspect, rounds

TASKS = ("inventory", "scheduling", "telemetry")
DEV = (8, 16)
SIZES = (8, 16, 32, 64, 128)
BATCH = 16


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_bytes())


def digest(array):
    arr = np.ascontiguousarray(array)
    descriptor = json.dumps([arr.shape, arr.dtype.str]).encode()
    return hashlib.sha256(descriptor + arr.tobytes()).hexdigest()


def draw(module, task, size, split, count, used):
    seed = [20261124, TASKS.index(task), size, int(split == "test")]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    inputs, labels = [], []
    for index in range(count):
        for _ in range(1000):
            x, y = module.sample(size, index, rng)
            if (x.dtype != np.int64 or x.ndim != 2 or len(x) != size or y.shape != (size,) or
                    not np.isin(y, (0, 1)).all() or not np.array_equal(y, module.oracle(x))):
                raise ValueError(f"Invalid {task} generator/oracle pair for {split}/{size}/{index}")
            raw = digest(x)
            if raw not in used:
                used.add(raw)
                break
        else:
            raise ValueError(f"Unable to draw a distinct {task} record")
        inputs.append(x)
        labels.append(y)
    return np.stack(inputs), np.stack(labels).astype(np.uint8)


def prepare():
    bank, lock = {}, {}
    for task in TASKS:
        module = importlib.import_module(task)
        spec, weights = module.build()
        info = inspect(spec, weights)
        info["nonzero_parameters"] = sum(int(np.count_nonzero(w)) for w in weights.values())
        target = HERE / "models" / task
        target.mkdir(parents=True, exist_ok=True)
        write(target / "network.json", spec)
        np.savez_compressed(target / "weights.npz", **weights)
        model = Transformer(spec, weights, literal=True)
        lock[task] = {}
        for n in SIZES:
            used = set()
            splits = [("dev", 12), ("test", 64)] if n in DEV else [("test", 64)]
            for split, count in splits:
                key = f"{task}_{split}_{n}"
                x, y = draw(module, task, n, split, count, used)
                bank[key + "_x"], bank[key + "_y"] = x, y
                lock[task][key] = {"inputs": digest(x), "labels": digest(y), "count": count}
                if split == "dev":
                    pred, _ = model.evaluate_batch(x, per_token=True)
                    if not all(np.array_equal(a, y) for a in pred.values()):
                        raise ValueError(f"{task}/{n} failed literal development check")
                    lock[task][key]["literal_predictions"] = digest(pred[1].astype(np.uint8))
        write(target / "freeze.json", info)
        print(f"Frozen {task} with {info['parameters']} parameters", flush=True)
    np.savez_compressed(HERE / "cases.npz", **bank)
    write(HERE / "corpus-lock.json", lock)


def gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name(0)
    lock = read(HERE / "corpus-lock.json")
    result = {"gpu": device, "dtype": "float64", "batch_size": BATCH, "tasks": {}}
    all_predictions = {}
    with np.load(HERE / "cases.npz", allow_pickle=False) as data, torch.inference_mode():
        for task in TASKS:
            module = importlib.import_module(task)
            folder = HERE / "models" / task
            spec = read(folder / "network.json")
            info = read(folder / "freeze.json")
            with np.load(folder / "weights.npz", allow_pickle=False) as archive:
                weights = {name: archive[name] for name in archive.files}
            if inspect(spec, weights)["model_sha256"] != info["model_sha256"]:
                raise ValueError(f"Changed {task} model")
            model = Transformer(spec, weights, gpu=True)
            report = {"model": info, "name": module.NAME, "unit": module.UNIT, "rows": []}
            result["tasks"][task] = report
            for key, expected in lock[task].items():
                x, y = data[key + "_x"], data[key + "_y"]
                if digest(x) != expected["inputs"] or digest(y) != expected["labels"]:
                    raise ValueError(f"Changed input or target for {key}")
                start = time.perf_counter()
                reference = np.stack([module.oracle(record) for record in x])
                replay = time.perf_counter() - start
                if not np.array_equal(y, reference):
                    raise ValueError(f"Reference labels failed CPU replay for {key}")
                predictions, stable = [], 0
                seconds = 0.0
                for i in range(0, len(x), BATCH):
                    p, timing = model.evaluate_batch(x[i:i+BATCH], per_token=True)
                    predictions.append(p[1].astype(np.uint8))
                    stable += int(((p[2] == p[1]) & (p[4] == p[1])).all(axis=1).sum())
                    seconds += timing[1]
                pred = np.concatenate(predictions)
                if "_dev_" in key:
                    if digest(pred) != expected["literal_predictions"] or stable != len(x):
                        raise ValueError(f"GPU and development reference differ at {key}")
                    continue
                all_predictions[key] = pred
                per_token = pred == y
                size = x.shape[1]
                row = {"size": size, "records": len(x), "exact": int(per_token.all(axis=1).sum()),
                       "decisions": y.size, "correct": int(per_token.sum()), "stable_records": stable,
                       "positive_labels": int(y.sum()), "false_positives": int(((pred == 1) & (y == 0)).sum()),
                       "false_negatives": int(((pred == 0) & (y == 1)).sum()), "rounds": rounds(spec, size),
                       "gpu_seconds": seconds, "oracle_seconds": replay, "predictions_sha256": digest(pred)}
                report["rows"].append(row)
                print(task, size, row["exact"], "/", len(x), f"{seconds:.6f} s", flush=True)
    np.savez_compressed("predictions.npz", **all_predictions)
    write(Path("application-results.json"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "gpu"))
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else gpu)()
