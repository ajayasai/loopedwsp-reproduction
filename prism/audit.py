"""Check the frozen inputs and recompute every public Kaggle prediction using NumPy."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
from oracles import DAY_NAMES, MONTHS, ORACLES, calendar_cycle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_evidence import authenticate_run, digest, read, require

INPUTS = (
    "experiment.py",
    "protocol.json",
    "checks.json",
    "isotopes.py",
    "calendar_round.py",
    "arm.py",
    "cases.json.gz",
    "weights.npz",
    "freeze.json",
)
OUTPUTS = ("prism-results.json", "prism-predictions.npz")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def check_weights(lock):
    counts = {}
    with np.load(HERE / "weights.npz", allow_pickle=False) as archive:
        expected = {f"{task}/{name}" for task, spec in lock["models"].items() for name in spec["tensors"]}
        require(expected == set(archive.files), "Weight names differ from freeze")
        for task, spec in lock["models"].items():
            total = nonzero = 0
            for name, metadata in spec["tensors"].items():
                tensor = archive[f"{task}/{name}"]
                nz = int(np.count_nonzero(tensor))
                require(
                    list(tensor.shape) == metadata["shape"]
                    and str(tensor.dtype) == metadata["dtype"]
                    and digest(tensor.tobytes()) == metadata["sha256"]
                    and nz == metadata["nonzero"]
                    and np.isfinite(tensor).all(),
                    f"Invalid {task} tensor {name}",
                )
                if metadata["parameter"]:
                    require(tensor.dtype == np.float32, "Unexpected parameter precision")
                    total += tensor.size
                    nonzero += nz
            require(total == spec["parameters"], f"Parameter count differs for {task}")
            counts[task] = {"parameters": total, "nonzero": nonzero}
    return counts


def test_size(task, record):
    key = {"isotopes": "ions", "calendar_round": "dates", "arm": "targets"}[task]
    return len(record[key])


def check_prediction(prediction, target, classes, shape, name):
    require(
        prediction.dtype == np.int16
        and list(prediction.shape) == shape
        and prediction.shape == np.shape(target)
        and (prediction >= -1).all()
        and (prediction < classes).all(),
        f"Invalid prediction array for {name}",
    )
    return prediction == np.asarray(target)


def check_all_dates(predictions):
    cycle = calendar_cycle()
    targets = []
    for number in range(1, 14):
        for name in DAY_NAMES:
            for month in MONTHS:
                for day in range(5 if month == "Wayeb'" else 20):
                    offset = cycle.get((number, name, day, month))
                    targets.append((365, 365) if offset is None else divmod(offset, 365))
    gold = np.asarray(targets)
    require(gold.shape == (94900, 2), "Calendar domain coverage differs")
    correct = check_prediction(predictions, gold, 366, [94900, 2], "calendar_all").all(axis=1)
    valid = gold[:, 0] < 365
    return {
        "pairs": len(targets),
        "exact": int(correct.sum()),
        "compatible": int(valid.sum()),
        "compatible_exact": int(correct[valid].sum()),
        "incompatible": int((~valid).sum()),
        "incompatible_exact": int(correct[~valid].sum()),
    }


def evaluate(run):
    reported = read(run / "output" / OUTPUTS[0])
    receipt, shapes = authenticate_run(
        run, reported, folder=HERE, inputs=INPUTS, outputs=OUTPUTS, stage_name="prism"
    )
    protocol = read(HERE / "protocol.json")
    tasks = protocol["tasks"]
    checks = read(HERE / "checks.json")
    lock = read(HERE / "freeze.json")
    for stem, ext in (("weights", "npz"), ("cases", "json.gz"), ("protocol", "json"), ("checks", "json")):
        require(
            digest((HERE / f"{stem}.{ext}").read_bytes()) == lock[stem + "_sha256"],
            f"Invalid freeze for {stem}",
        )
    require(
        all(lock[k] == reported[k] for k in ("weights_sha256", "cases_sha256")),
        "Run used different inputs",
    )
    models = check_weights(lock)
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))
    keys = {f"{s}/{t}/{n}" for t, cfg in tasks.items() for s in ("dev", "test") for n in cfg[s]}
    require(set(cases) == set(lock["cohorts"]) == keys, "Unexpected benchmark cohorts")
    reported_rows = {(r["task"], r["size"]): r for r in reported["conditions"]}
    pairs = [(t, n) for t, cfg in tasks.items() for n in cfg["test"]]
    require(
        len(reported["conditions"]) == len(pairs) and set(reported_rows) == set(pairs),
        "GPU size coverage differs",
    )
    names = {c["name"] for c in checks}
    require(
        len(names) == len(checks) and set(reported["checks"]) == names,
        "Worked checks missing or repeated",
    )
    seen = {t: {canonical(read(HERE / "examples" / f"{t}.json"))} for t in tasks}
    counts = {t: Counter() for t in tasks}
    correct_counts = {t: Counter() for t in tasks}
    verified, rows = {}, []
    with np.load(run / "output" / OUTPUTS[1], allow_pickle=False) as archive:
        expected = {f"{t}_{n}" for t, n in pairs} | {"check_" + name for name in names} | {"calendar_all"}
        require(set(archive.files) == set(shapes) == expected, "Prediction archive coverage differs")
        exhaustive = check_all_dates(archive["calendar_all"])
        require(
            shapes["calendar_all"] == [94900, 2] and exhaustive == reported["exhaustive_calendar"],
            "Independent exhaustive calendar score differs",
        )
        for check in checks:
            task, name = check["task"], check["name"]
            seen[task].add(canonical(check["input"]))
            target, _ = ORACLES[task](check["input"])
            require(target == check["labels"], f"Worked answer differs for {name}")
            key = "check_" + name
            ok = check_prediction(archive[key], target, tasks[task]["classes"], shapes[key], name)
            worked = {"task": task, "correct": bool(ok.all())}
            require(worked == reported["checks"][name], f"Worked score differs for {name}")
            verified[name] = {**worked, "correct_labels": int(ok.sum()), "labels": len(target)}
        for split in ("dev", "test"):
            for task, cfg in tasks.items():
                for size in cfg[split]:
                    cohort = f"{split}/{task}/{size}"
                    batch = cases[cohort]
                    require(
                        digest(canonical(batch)) == lock["cohorts"][cohort]
                        and len(batch) == protocol["per_size"][split],
                        f"Corpus changed in {cohort}",
                    )
                    targets, groups = [], []
                    for case in batch:
                        record = case["input"]
                        marker = canonical(record)
                        require(
                            marker not in seen[task] and test_size(task, record) == size,
                            f"Duplicate or wrong size in {cohort}",
                        )
                        seen[task].add(marker)
                        target, categories = ORACLES[task](record)
                        require(
                            target == case["gold"]
                            and len(categories) * (1 if task == "isotopes" else 2) == len(target),
                            f"Reference differs in {cohort}",
                        )
                        targets.append(target)
                        groups.append(categories)
                    if split == "dev":
                        continue
                    key = f"{task}_{size}"
                    pred = archive[key]
                    ok = check_prediction(pred, targets, cfg["classes"], shapes[key], key)
                    scores = {
                        "records": len(batch),
                        "exact_records": int(ok.all(axis=1).sum()),
                        "labels": ok.size,
                        "correct_labels": int(ok.sum()),
                        "parameters": models[task]["parameters"],
                        "nonfinite": int((pred < 0).sum()),
                    }
                    row = reported_rows[task, size]
                    require(
                        all(row[k] == value for k, value in scores.items()),
                        f"Reported score differs for {key}",
                    )
                    times = row["seconds"]
                    require(
                        len(times) == protocol["repeats"]
                        and all(t > 0 and np.isfinite(t) for t in times)
                        and row["median_seconds"] == median(times),
                        f"Invalid timing for {key}",
                    )
                    # A calendar offset or a pose count with its witness is one
                    # answer. Both integer labels must match to earn credit.
                    width = 1 if task == "isotopes" else 2
                    answers = ok.reshape(len(batch), -1, width).all(axis=-1)
                    for group_row, flags in zip(groups, answers, strict=True):
                        for categories, correct in zip(group_row, flags, strict=True):
                            counts[task].update(categories)
                            if correct:
                                correct_counts[task].update(categories)
                    row = {**row, "answers": int(answers.size), "correct_answers": int(answers.sum())}
                    rows.append(dict(row))
    return {
        "run": receipt,
        "exhaustive_calendar": exhaustive,
        "models": models,
        "conditions": rows,
        "worked_checks": verified,
        "answers_by_category": counts,
        "correct_answers_by_category": correct_counts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit = evaluate(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(audit, indent=2, allow_nan=False) + "\n").encode())
    print(f"Regraded {sum(r['records'] for r in audit['conditions'])} held-out records")
