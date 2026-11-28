"""Evaluate a frozen protocol or apply a programmed transformer to a JSON record."""
import argparse
import gzip
import importlib
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "protocol.json").read_bytes())
MODELS = {name: importlib.import_module(name) for name in PROTOCOL["tasks"]}
STUDY = PROTOCOL["study"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(blob):
    return sha256(blob).hexdigest()


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n").encode())


def input_id(module, record):
    encoded = np.ascontiguousarray(module.encode(record), dtype="<i8")
    return encoded.shape, digest(encoded.tobytes())


def freeze():
    cases, tensors, ledgers = {}, {}, {}
    for task_id, (task, module) in enumerate(MODELS.items()):
        model = module.build_model().eval()
        entries = {}
        parameters = dict(model.named_parameters())
        for name, t in model.state_dict().items():
            value = t.detach().cpu().numpy().copy(order="C")
            entries[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                             "parameter": name in parameters, "nonzero": int(np.count_nonzero(value)),
                             "sha256": digest(value.tobytes())}
            tensors[f"{task}/{name}"] = value
        ledgers[task] = {"parameters": sum(t.numel() for t in parameters.values()), "tensors": entries}
        seen = {input_id(module, module.example())}
        checks = json.loads((HERE / "checks.json").read_bytes())
        seen.update(input_id(module, c["input"]) for c in checks if c["task"] == task)
        for split_id, (split, count) in enumerate(PROTOCOL["per_size"].items()):
            for size in PROTOCOL["tasks"][task][split]:
                rng = np.random.default_rng(np.random.SeedSequence([PROTOCOL["seed"], task_id, split_id, size]))
                rows = []
                for index in range(20000):
                    record = module.generate(rng, size, index)
                    key = input_id(module, record)
                    if key in seen:
                        continue
                    labels = np.asarray(module.reference(record))
                    if (labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu"
                            or labels.min() < 0 or labels.max() >= PROTOCOL["tasks"][task]["classes"]):
                        raise ValueError(f"Invalid oracle result for {task}")
                    seen.add(key)
                    rows.append({"input": record, "gold": labels.tolist()})
                    if len(rows) == count:
                        break
                if len(rows) != count:
                    raise ValueError(f"Insufficient distinct inputs for {task}/{size}")
                cases[f"{split}/{task}/{size}"] = rows
    examples = HERE / "examples"
    examples.mkdir(exist_ok=True)
    for task, module in MODELS.items():
        write_json(examples / f"{task}.json", module.example())
    np.savez_compressed(HERE / "weights.npz", **tensors)
    data = gzip.compress(canonical(cases), compresslevel=9, mtime=0)
    (HERE / "cases.json.gz").write_bytes(data)
    write_json(HERE / "freeze.json", {"models": ledgers, "cases_sha256": digest(data),
               "weights_sha256": digest((HERE / "weights.npz").read_bytes()),
               "protocol_sha256": digest((HERE / "protocol.json").read_bytes()),
               "checks_sha256": digest((HERE / "checks.json").read_bytes()),
               "cohorts": {k: digest(canonical(v)) for k, v in cases.items()}})
    print(f"Froze {len(tensors)} tensors and {sum(map(len, cases.values()))} records")


def load(task, lock, device):
    path = HERE / "weights.npz"
    if digest(path.read_bytes()) != lock["weights_sha256"]:
        raise ValueError("Frozen weight archive changed")
    model = MODELS[task].build_model().eval()
    ledger = lock["models"][task]
    if (set(model.state_dict()) != set(ledger["tensors"])
            or sum(p.numel() for p in model.parameters()) != ledger["parameters"]):
        raise ValueError(f"Unexpected {task} model structure")
    states = {}
    with np.load(path, allow_pickle=False) as archive:
        for name, info in ledger["tensors"].items():
            value = archive[f"{task}/{name}"]
            if (list(value.shape) != info["shape"] or str(value.dtype) != info["dtype"]
                    or digest(value.tobytes()) != info["sha256"]):
                raise ValueError(f"Frozen tensor mismatch at {task}/{name}")
            states[name] = torch.from_numpy(value)
    model.load_state_dict(states, strict=True)
    return model.to(device)


def inputs(task, records, device):
    values = np.stack([MODELS[task].encode(record) for record in records])
    if values.ndim != 3 or values.dtype.kind not in "iu":
        raise ValueError("Expected integer tokens")
    return torch.as_tensor(values, dtype=torch.long, device=device)


def decode(task, logits):
    if logits.ndim != 3 or logits.shape[-1] != PROTOCOL["tasks"][task]["classes"] or logits.dtype != torch.float32:
        raise ValueError(f"Unexpected {task} logits with shape {logits.shape} and dtype {logits.dtype}")
    return torch.where(logits.isfinite().all(-1), logits.argmax(-1), -1).cpu().numpy().astype(np.int16)


def exhaustive_calendar(model, device):
    """Visit each legal (number, name, month, day) tuple exactly once."""
    task = "calendar_round"
    module = MODELS[task]
    dates = [{"number": number, "name": name, "month": month, "day": day}
             for number in range(1, 14) for name in module.NAMES
             for month in module.MONTHS for day in range(5 if month == "Wayeb'" else 20)]
    records = [{"dates": dates[i:i+25]} for i in range(0, len(dates), 25)]
    batches = inputs(task, records, device).split(PROTOCOL["batch"])
    predictions = np.concatenate([decode(task, model(batch)).reshape(-1, 2) for batch in batches])
    gold = np.stack([module.reference(record) for record in records]).reshape(-1, 2)
    correct = (predictions == gold).all(axis=1)
    compatible = gold[:, 0] != module.NO_DATE
    scores = {"pairs": len(dates), "exact": int(correct.sum()),
              "compatible": int(compatible.sum()), "compatible_exact": int(correct[compatible].sum()),
              "incompatible": int((~compatible).sum()), "incompatible_exact": int(correct[~compatible].sum())}
    return predictions, scores


@torch.inference_mode()
def evaluate(split, dest, device):
    lock = json.loads((HERE / "freeze.json").read_bytes())
    blob = (HERE / "cases.json.gz").read_bytes()
    if digest(blob) != lock["cases_sha256"] or digest((HERE / "protocol.json").read_bytes()) != lock["protocol_sha256"]:
        raise ValueError("The cases or protocol differ from the freeze")
    cases = json.loads(gzip.decompress(blob))
    rows, preds, worked = [], {}, {}
    checks_blob = (HERE / "checks.json").read_bytes()
    if digest(checks_blob) != lock["checks_sha256"]:
        raise ValueError("Worked examples differ from the freeze")
    checks = json.loads(checks_blob)
    for task in MODELS:
        model = load(task, lock, device)
        for c in (c for c in checks if c["task"] == task):
            name = c["name"]
            if name in worked:
                raise ValueError(f"Duplicate worked example {name}")
            raw_output = decode(task, model(inputs(task, [c["input"]], device)))[0]
            preds["check_" + name] = raw_output
            if "forms" in c:
                surface = MODELS[task].format_output(c["input"], raw_output.tolist())["forms"]
                worked[name] = {"task": task, "correct": surface == c["forms"], "forms": surface}
            else:
                worked[name] = {"task": task, "correct": raw_output.tolist() == c["labels"]}
        for size in PROTOCOL["tasks"][task][split]:
            key = f"{split}/{task}/{size}"
            cohort = cases[key]
            if digest(canonical(cohort)) != lock["cohorts"][key]:
                raise ValueError(f"Unexpected cohort {key}")
            batches = inputs(task, [r["input"] for r in cohort], device).split(PROTOCOL["batch"])
            gold = np.asarray([r["gold"] for r in cohort], dtype=np.int16)
            model(batches[0])
            previous, times = None, []
            for _ in range(PROTOCOL["repeats"] if split == "test" else 1):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = perf_counter()
                logits = [model(batch) for batch in batches]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(perf_counter() - start)
                predicted = decode(task, torch.cat(logits))
                if predicted.shape != gold.shape or (previous is not None and not np.array_equal(previous, predicted)):
                    raise ValueError(f"Unstable or invalid predictions for {key}")
                previous = predicted
            correct = predicted == gold
            preds[f"{task}_{size}"] = predicted
            rows.append({"task": task, "size": size, "parameters": lock["models"][task]["parameters"],
                         "records": len(gold), "exact_records": int(correct.all(1).sum()),
                         "labels": gold.size, "correct_labels": int(correct.sum()), "nonfinite": int((predicted < 0).sum()),
                         "seconds": times, "median_seconds": median(times)})
            print(f"{key}: {rows[-1]['exact_records']}/{len(gold)} correct", flush=True)
    extra = {}
    if split == "test":
        preds["calendar_all"], extra["exhaustive_calendar"] = exhaustive_calendar(load("calendar_round", lock, device), device)
        print(f"Exhaustive calendar: {extra['exhaustive_calendar']['exact']}/{extra['exhaustive_calendar']['pairs']} correct", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    write_json(dest / f"{STUDY}-results.json", {"split": split, "torch": str(torch.__version__), "cuda": torch.version.cuda,
               "device": torch.cuda.get_device_name() if device.type == "cuda" else "CPU", "conditions": rows, "checks": worked,
               "cases_sha256": lock["cases_sha256"], "weights_sha256": lock["weights_sha256"], **extra})
    np.savez_compressed(dest / f"{STUDY}-predictions.npz", **preds)


@torch.inference_mode()
def predict(task, record):
    device = torch.device("cpu")
    model = load(task, json.loads((HERE / "freeze.json").read_bytes()), device)
    pred = decode(task, model(inputs(task, [record], device)))[0]
    if (pred < 0).any():
        raise ValueError("Nonfinite result")
    return MODELS[task].format_output(record, pred.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("freeze", help="Rebuild the saved inputs, labels and weights")
    for mode in ("dev", "gpu"):
        actions.add_parser(mode).add_argument("--output", type=Path)
    demo = actions.add_parser("predict")
    demo.add_argument("task", choices=MODELS)
    src = demo.add_mutually_exclusive_group()
    src.add_argument("--example", action="store_true")
    src.add_argument("--input", type=Path, help="JSON record; defaults to stdin")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
    elif args.command == "predict":
        if args.example:
            record = MODELS[args.task].example()
        elif args.input:
            record = json.loads(args.input.read_bytes())
        else:
            record = json.load(sys.stdin)
        answer = predict(args.task, record)
        print(json.dumps({"input": record, "output": answer} if args.example else answer, indent=2, ensure_ascii=False))
    else:
        gpu = args.command == "gpu"
        if gpu and not torch.cuda.is_available():
            raise RuntimeError("The GPU test requires CUDA")
        out = args.output or (Path.cwd() if gpu else HERE.parent / "audit-output" / f"{STUDY}-dev")
        evaluate("test" if gpu else "dev", out, torch.device("cuda" if gpu else "cpu"))


if __name__ == "__main__":
    main()
