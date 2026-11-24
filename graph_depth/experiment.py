"""Run the frozen pointer-chasing and directed-reachability comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
from statistics import median
import tempfile
from time import perf_counter

import numpy as np
import torch

import corpus
import pointer
import reachability


HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_bytes())
TASKS = {"pointer": pointer, "reachability": reachability}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensor_ledger(model):
    entries = {}
    for kind, tensors in [("parameter", model.named_parameters()), ("buffer", model.named_buffers())]:
        for name, tensor in tensors:
            a = tensor.detach().cpu().contiguous().numpy()
            entries[name] = {"kind": kind, "shape": list(a.shape), "size": a.size,
                             "nonzero": int(np.count_nonzero(a)), "dtype": str(a.dtype),
                             "sha256": digest(a.tobytes())}
    parameters = [entry for entry in entries.values() if entry["kind"] == "parameter"]
    return {"stored": sum(x["size"] for x in parameters), "nonzero": sum(x["nonzero"] for x in parameters),
            "buffer_scalars": sum(x["size"] for x in entries.values() if x["kind"] == "buffer"),
            "tensor_manifest_sha256": digest(json.dumps(entries, sort_keys=True).encode()), "tensors": entries}


def inputs(arrays, task, split, hops, device):
    prefix = f"{task}_{split}_{hops}_"
    def tensor(name):
        return torch.as_tensor(arrays[prefix + name], device=device)
    source = tensor("source").long()
    if task == "reachability":
        arguments = (tensor("adj").bool(), source, tensor("target").long())
    else:
        table = tensor("table").long()
        # Remove serialization order only. Every architecture sees the same single-hop pointer table.
        ordered = torch.empty_like(table[:, :, 0]).scatter_(1, table[:, :, 0], table[:, :, 1])
        arguments = (ordered, source)
    return arguments, arrays[prefix + "label"].astype(np.int64)


def predict(model, task, hops, arguments, batch_size, repeats, device):
    batches = [tuple(a[start:start + batch_size] for a in arguments)
               for start in range(0, len(arguments[0]), batch_size)]
    def forward(batch):
        return model(*batch, hops) if task == "pointer" else model(*batch)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    forward(batches[0])
    sync()
    seconds = []
    first = None
    for _ in range(repeats):
        sync()
        start = perf_counter()
        logits = [forward(batch) for batch in batches]
        sync()
        seconds.append(perf_counter() - start)
        values = torch.cat(logits)
        classes = PROTOCOL["nodes"] if task == "pointer" else 2
        if tuple(values.shape) != (len(arguments[0]), classes) or values.dtype != torch.float32:
            raise ValueError(f"Unexpected logits: {values.shape}, {values.dtype}")
        pred = torch.where(values.isfinite().all(1), values.argmax(1), -1).cpu().numpy().astype(np.int16)
        if first is not None and not np.array_equal(first, pred):
            raise ValueError("Predictions changed across identical timing repeats")
        first = pred
    return first, seconds


def score(task, pred, gold):
    right = pred == gold
    result = {"correct": int(right.sum()), "queries": len(gold), "pairs": len(gold) // 2,
              "both_pair_members_correct": int(right.reshape(-1, 2).all(1).sum()),
              "invalid": int((pred < 0).sum())}
    if task == "reachability":
        for label, name in [(0, "unreachable"), (1, "reachable")]:
            result[name + "_correct"] = int(right[gold == label].sum())
            result[name + "_queries"] = int((gold == label).sum())
    return result


@torch.inference_mode()
def run(split, cases_path, output, device):
    cfg = PROTOCOL["test" if split == "test" else "development"]
    timing = PROTOCOL["timing"]
    rng = random.Random(timing["order_seed"])
    results = {"split": split, "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
               "pytorch": str(torch.__version__), "cuda": torch.version.cuda, "numpy": np.__version__,
               "python": platform.python_version(), "precision": "float32", "ledgers": {}, "conditions": []}
    predictions = {}
    with np.load(cases_path, allow_pickle=False) as arrays:
        for task, module in TASKS.items():
            design = PROTOCOL["tasks"][task]
            models = {}
            for i, budget in enumerate(design["budgets"]):
                family = module.build_models(budget, nodes=PROTOCOL["nodes"])
                if set(family) != set(design["depths"]):
                    raise ValueError(f"Methods differ from protocol: {list(family)}")
                for name, model in family.items():
                    model = model.to(dtype=torch.float32)
                    ledger = tensor_ledger(model)
                    ledger["architecture"] = model.metadata()
                    expected = budget if name in design["matched"] else design["unrolled_parameters"][i]
                    if ledger["stored"] != expected:
                        raise ValueError(f"{task}/{budget}/{name}: {ledger['stored']} vs {expected}")
                    results["ledgers"][f"{task}/{budget}/{name}"] = ledger
                    models[budget, name] = model.eval().to(device)
            for hops in cfg["hops"]:
                args, gold = inputs(arrays, task, split, hops, device)
                if len(gold) != cfg["pairs_per_bin"] * 2:
                    raise ValueError("Unexpected cohort size")
                order = list(models)
                rng.shuffle(order)
                for budget, name in order:
                    model = models[budget, name]
                    depth = design["depths"][name]
                    if task == "pointer" and model.executed_depth(hops) != depth:
                        raise ValueError("Pointer depth differs from the frozen protocol")
                    pred, seconds = predict(model, task, hops, args, timing["batch_size"],
                                            timing["full_cohort_repeats"] if split == "test" else 1, device)
                    key = f"{task}_{budget}_{name}_{hops}"
                    predictions[key] = pred
                    results["conditions"].append({"task": task, "budget": budget, "method": name,
                                                  "hops": hops, "stages": depth, "prediction_key": key,
                                                  "stored": results["ledgers"][f"{task}/{budget}/{name}"]["stored"],
                                                  "seconds": seconds, "median_seconds": median(seconds),
                                                  **score(task, pred, gold)})
                print(f"{task} {split} {hops} hops: {len(order)} settings complete", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "graph-depth-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "graph-depth-predictions.npz", **predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["dev","gpu"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    gpu = args.mode == "gpu"
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("The test evaluation requires a CUDA GPU")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output or (Path.cwd() if gpu else HERE.parent / "audit-output" / "graph-depth-dev")
    with tempfile.TemporaryDirectory() as temp:
        archive = Path(temp) / "cases.npz"
        manifest = corpus.write_cases(archive)
        frozen = json.loads((HERE / "cases-manifest.json").read_bytes())
        if manifest["arrays"] != frozen["arrays"]:
            raise ValueError("Regenerated inputs differ from the frozen case manifest")
        run("test" if gpu else "dev", archive, output, torch.device("cuda" if gpu else "cpu"))


if __name__ == "__main__":
    main()
