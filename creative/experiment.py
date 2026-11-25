"""Freeze, run, or try three hand-programmed attention models."""
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import torch

import ballots
import folds
import typeset


HERE = Path(__file__).resolve().parent
MODULES = {"folds": folds, "typeset": typeset, "ballots": ballots}
CLASSES = {"folds": 8, "typeset": 2, "ballots": 5}
PLAN = {"seed": 2026112501, "dev": {"sizes": [4, 8], "records": 8},
        "test": {"sizes": [4, 8, 16, 32], "records": 64}, "batch_size": 8, "repeats": 3}


def packed(obj):
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def fingerprint(blob):
    return sha256(blob).hexdigest()


def load_json(path):
    return json.loads(path.read_bytes())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def freeze():
    cohorts, tensors, ledgers = {}, {}, {}
    for task_id, (task, module) in enumerate(MODULES.items()):
        model = module.build_model().eval()
        entries = {}
        parameter_names = dict(model.named_parameters())
        for name, tensor in model.state_dict().items():
            a = tensor.detach().cpu().numpy().copy(order="C")
            tensors[task + "/" + name] = a
            entries[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": fingerprint(a.tobytes()),
                             "parameter": name in parameter_names, "count": a.size, "nonzero": int(np.count_nonzero(a))}
        if not set(parameter_names) <= set(entries):
            raise ValueError(f"{task} has unarchived parameters")
        ledgers[task] = {"tensors": entries, "parameters": sum(p.numel() for p in model.parameters())}
        used = set()
        for split_id, split in enumerate(("dev", "test")):
            cfg = PLAN[split]
            for size in cfg["sizes"]:
                rng = np.random.default_rng(np.random.SeedSequence([PLAN["seed"], task_id, split_id, size]))
                rows = []
                for attempt in range(20000):
                    record = module.generate(rng, size, attempt)
                    encoded = np.ascontiguousarray(module.encode(record), dtype="<i8")
                    key = (encoded.shape, fingerprint(encoded.tobytes()))
                    if key in used:
                        continue
                    gold = np.asarray(module.reference(record))
                    if gold.ndim != 1 or not gold.size or gold.min() < 0 or gold.max() >= CLASSES[task]:
                        raise ValueError(f"Invalid oracle output for {task}")
                    used.add(key)
                    rows.append({"input": record, "gold": gold.astype(int).tolist()})
                    if len(rows) == cfg["records"]:
                        break
                if len(rows) != cfg["records"]:
                    raise ValueError(f"Insufficient distinct {task} inputs for {size}")
                cohorts[f"{split}/{task}/{size}"] = rows
    cases = gzip.compress(packed(cohorts), compresslevel=9, mtime=0)
    (HERE / "cases.json.gz").write_bytes(cases)
    np.savez_compressed(HERE / "weights.npz", **tensors)
    write_json(HERE / "freeze.json", {"plan": PLAN, "models": ledgers, "cases_sha256": fingerprint(cases),
               "weights_sha256": fingerprint((HERE / "weights.npz").read_bytes()),
               "cohorts": {k: fingerprint(packed(v)) for k, v in cohorts.items()}})
    print(f"Froze {len(cohorts)} cohorts and {len(tensors)} model tensors.")


def load_model(task, lock, device):
    module = MODULES[task]
    model = module.build_model().eval()
    ledger = lock["models"][task]
    if sum(p.numel() for p in model.parameters()) != ledger["parameters"] or set(model.state_dict()) != set(ledger["tensors"]):
        raise ValueError(f"Incompatible model structure for {task}")
    state = {}
    with np.load(HERE / "weights.npz", allow_pickle=False) as archive:
        for name, entry in ledger["tensors"].items():
            value = archive[task + "/" + name]
            if (list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]
                    or fingerprint(value.tobytes()) != entry["sha256"]):
                raise ValueError(f"Frozen tensor mismatch for {task}/{name}")
            state[name] = torch.from_numpy(value)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def inputs(module, records, device):
    encoded = np.stack([module.encode(r) for r in records])
    if encoded.ndim != 3 or encoded.dtype.kind not in "iu":
        raise ValueError("Expected an integer token matrix for each record")
    return torch.as_tensor(encoded, dtype=torch.long, device=device)


def decode(logits, task):
    if logits.ndim != 3 or logits.shape[-1] != CLASSES[task] or not logits.is_floating_point():
        raise ValueError(f"Invalid logits for {task}: {logits.shape}")
    return torch.where(logits.isfinite().all(-1), logits.argmax(-1), -1).cpu().numpy().astype(np.int16)


@torch.inference_mode()
def run(split, output, device):
    lock = load_json(HERE / "freeze.json")
    cases_blob = (HERE / "cases.json.gz").read_bytes()
    if (lock["plan"] != PLAN or fingerprint(cases_blob) != lock["cases_sha256"]
            or fingerprint((HERE / "weights.npz").read_bytes()) != lock["weights_sha256"]):
        raise ValueError("Corpus, weight archive or protocol differs from freeze")
    cases = json.loads(gzip.decompress(cases_blob))
    predictions, conditions = {}, []
    repeats = PLAN["repeats"] if split == "test" else 1
    for task, module in MODULES.items():
        model = load_model(task, lock, device)
        for size in PLAN[split]["sizes"]:
            key = f"{split}/{task}/{size}"
            cohort = cases[key]
            if fingerprint(packed(cohort)) != lock["cohorts"][key]:
                raise ValueError(f"Unexpected cohort {key}")
            gold = np.stack([r["gold"] for r in cohort])
            tensors = inputs(module, [r["input"] for r in cohort], device).split(PLAN["batch_size"])
            model(tensors[0])
            ref_pred, seconds = None, []
            for _ in range(repeats):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = perf_counter()
                logits = [model(batch) for batch in tensors]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                seconds.append(perf_counter() - start)
                pred = decode(torch.cat(logits), task)
                if pred.shape != gold.shape or (ref_pred is not None and not np.array_equal(pred, ref_pred)):
                    raise ValueError(f"Unexpected output shape or unstable repeats in {key}")
                ref_pred = pred
            predictions[f"{task}_{size}"] = pred
            correct = gold == pred
            conditions.append({"task": task, "size": size, "records": len(gold), "exact_records": int(correct.all(1).sum()),
                               "labels": gold.size, "correct_labels": int(correct.sum()), "nonfinite": int((pred < 0).sum()),
                               "parameters": lock["models"][task]["parameters"], "median_seconds": median(seconds),
                               "seconds": seconds, "class_counts": dict(sorted(Counter(map(int, gold.ravel())).items()))})
            print(f"{key}: {conditions[-1]['exact_records']}/{len(gold)} exact", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "creative-results.json", {"split": split, "device": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
               "torch": str(torch.__version__), "cuda": torch.version.cuda, "weights_sha256": lock["weights_sha256"],
               "cases_sha256": lock["cases_sha256"], "batch_size": PLAN["batch_size"], "conditions": conditions})
    np.savez_compressed(output / "creative-predictions.npz", **predictions)


@torch.inference_mode()
def predict(task, record):
    module = MODULES[task]
    device = torch.device("cpu")
    model = load_model(task, load_json(HERE / "freeze.json"), device)
    pred = decode(model(inputs(module, [record], device)), task)[0]
    if (pred < 0).any():
        raise ValueError("Nonfinite model output")
    return module.format_output(record, pred.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("freeze", help="Rebuild frozen inputs, oracle targets and stored model tensors")
    for mode in ("dev", "gpu"):
        actions.add_parser(mode).add_argument("--output", type=Path)
    sample = actions.add_parser("predict")
    sample.add_argument("task", choices=MODULES)
    source = sample.add_mutually_exclusive_group()
    source.add_argument("--example", action="store_true", help="Show a built-in record and its model prediction")
    source.add_argument("--input", type=Path, help="Read JSON from a file (defaults to stdin)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
    elif args.command == "predict":
        record = MODULES[args.task].example() if args.example else load_json(args.input) if args.input else json.load(sys.stdin)
        prediction = predict(args.task, record)
        print(json.dumps({"input": record, "output": prediction} if args.example else prediction, indent=2, ensure_ascii=False))
    else:
        gpu = args.command == "gpu"
        if gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU evaluation requires CUDA")
        output = args.output or (Path.cwd() if gpu else HERE.parent / "audit-output" / "creative-dev")
        run("test" if gpu else "dev", output, torch.device("cuda" if gpu else "cpu"))


if __name__ == "__main__":
    main()
