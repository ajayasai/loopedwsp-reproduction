"""Run the registered depth, precision and parameter comparison on one CUDA device."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from baselines import build_baselines
from recurrent import build_recurrent


HERE = Path(__file__).resolve().parent
ASSETS = (HERE if (HERE / "applications").is_dir() else HERE.parent) / "applications"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_digest(array):
    descr = json.dumps([array.shape, array.dtype.str]).encode()
    return hashlib.sha256(descr + np.ascontiguousarray(array).tobytes()).hexdigest()


def original(task):
    folder = ASSETS / "models" / task
    spec, freeze = read(folder / "network.json"), read(folder / "freeze.json")
    with np.load(folder / "weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    for key, w in sorted(weights.items()):
        digest.update(json.dumps([key, w.shape, w.dtype.str]).encode())
        digest.update(np.ascontiguousarray(w).tobytes())
    if digest.hexdigest() != freeze["model_sha256"]:
        raise ValueError(f"Frozen weights changed: {task}")
    return spec, weights, freeze


def get_steps(method, rule, n):
    if method == "original_full":
        return {"tokens": n, "log2_plus_one": (n-1).bit_length() + 1}[rule]
    return 4 if method == "original_4" else 1


def classify(scores):
    """Two denotes a numerical failure, which is never a correct binary prediction."""
    return torch.where(torch.isfinite(scores), (scores > 0).to(torch.uint8), 2)


def grade(pred, y, pairs=False):
    correct = pred == y
    result = {"n_records": len(y), "exact": int(correct.all(1).sum()),
              "n_labels": y.size, "correct": int(correct.sum()), "invalid": int((pred == 2).sum()),
              "false_positives": int(((pred == 1) & (y == 0)).sum()),
              "false_negatives": int(((pred == 0) & (y == 1)).sum()),
              "final_correct": int(correct[:, -1].sum())}
    if pairs:
        result["pairs_both_final_correct"] = int((correct[::2, -1] & correct[1::2, -1]).sum())
    return result


def score_batch(model, batch, method, steps, dtype):
    if method.startswith("original_"):
        screened, raw = model(batch, steps, return_raw=True)
    else:
        screened = raw = model(batch)
    if screened.dtype != dtype or raw.dtype != dtype or screened.shape != batch.shape[:2]:
        raise ValueError(f"Unexpected output precision or shape from {method}")
    return screened, raw


def measure(model, x, method, steps, dtype, batch_size, repeats):
    batches = x.split(batch_size)
    score_batch(model, batches[0], method, steps, dtype)
    times, saved = [], None
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = [score_batch(model, batch, method, steps, dtype) for batch in batches]
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        ids = tuple(classify(torch.cat(scores)).cpu().numpy() for scores in zip(*outputs))
        if saved is not None and not all(np.array_equal(i, j) for i, j in zip(ids, saved)):
            raise RuntimeError(f"Non-reproducible predictions across repeats: {method}")
        saved = ids
    return *saved, times


def counts(model):
    return (sum(w.numel() for w in model.parameters()),
            sum(torch.count_nonzero(w).item() for w in model.parameters()))


def run(dev=False):
    p = read(HERE / "protocol.json")
    for path, name in [(ASSETS/"cases.npz", "cases"), (ASSETS/"corpus-lock.json", "corpus_lock"),
                       (HERE/"challenge.npz", "challenge")]:
        if sha(path) != p[name + "_sha256"]:
            raise ValueError(f"Changed registered {name}")
    device = "cpu" if dev else "cuda"
    if not dev and not torch.cuda.is_available():
        raise RuntimeError("A Kaggle CUDA GPU is required")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    result = {"gpu": "CPU developer smoke test" if dev else torch.cuda.get_device_name(0),
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "protocol_sha256": sha(HERE / "protocol.json"), "rows": []}
    stored = {}
    lock = read(ASSETS / "corpus-lock.json")
    rng = random.Random(p["method_order_seed"])
    with np.load(ASSETS/"cases.npz", allow_pickle=False) as corpus, np.load(
            HERE/"challenge.npz", allow_pickle=False) as challenges, torch.inference_mode():
        for task in p["tasks"]:
            spec, w, frozen = original(task)
            for n in ([8, 16] if dev else p["sizes"]):
                key = f"{task}_{'dev' if dev else 'test'}_{n}"
                x, y = corpus[key+"_x"], corpus[key+"_y"]
                if tensor_digest(x) != lock[task][key]["inputs"] or tensor_digest(y) != lock[task][key]["labels"]:
                    raise ValueError(f"Archived corpus mismatch: {key}")
                pub = len(y)
                if not dev:
                    challenge = f"{task}_{n}"
                    x = np.concatenate((x, challenges[challenge+"_x"]))
                    y = np.concatenate((y, challenges[challenge+"_y"]))
                x = torch.as_tensor(x, device=device)
                for precision in p["precisions"]:
                    dtype = getattr(torch, precision)
                    recur = build_recurrent(spec, w)
                    models = build_baselines(task)
                    models.update({name: recur for name in p["methods"] if name.startswith("original_")})
                    if set(models) != set(p["methods"]):
                        raise ValueError("Methods disagree with registration")
                    param_counts = {name: counts(m) for name, m in models.items()}
                    if param_counts["original_full"] != (frozen["parameters"], frozen["nonzero_parameters"]):
                        raise ValueError("Original parameter count mismatch")
                    order = list(p["methods"])
                    rng.shuffle(order)
                    for name in order:
                        model = models[name].to(device=device, dtype=dtype).eval()
                        steps = get_steps(name, spec["round_rule"], n)
                        if dev:
                            scores, raw = score_batch(model, x, name, steps, dtype)
                            labels = classify(scores).numpy()
                            if name == "original_full" and precision == "float64" and not np.array_equal(labels, y):
                                raise ValueError(f"Development reference mismatch for {task}")
                            continue
                        pred, raw, times = measure(model, x, name, steps, dtype, p["batch_size"], p["timed_repeats"])
                        pk = f"{task}_{n}_{precision}_{name}"
                        raw_key = pk + "_raw" if name.startswith("original_") else pk
                        stored[pk], stored[raw_key] = pred, raw
                        dense, nonzero = param_counts[name]
                        result["rows"].append({
                            "task": task, "size": n, "precision": precision, "method": name,
                            "parameters": dense, "nonzero_parameters": nonzero, "steps": steps,
                            "seconds": times, "median_seconds": statistics.median(times),
                            "prediction_key": pk, "raw_prediction_key": raw_key,
                            "published": grade(pred[:pub], y[:pub]), "challenge": grade(pred[pub:], y[pub:], True),
                            "raw_published": grade(raw[:pub], y[:pub]), "raw_challenge": grade(raw[pub:], y[pub:], True)})
                print(task, n, "complete", flush=True)
    if not dev:
        Path("comparison-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        np.savez_compressed("comparison-predictions.npz", **stored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gpu", "dev"])
    run(parser.parse_args().mode == "dev")
