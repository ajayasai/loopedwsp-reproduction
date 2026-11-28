"""Check the frozen inputs and recompute every public Kaggle prediction using NumPy."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np
from oracles import LETTERS, ORACLES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_evidence import authenticate_run, digest, read, require

INPUTS = (
    "experiment.py",
    "protocol.json",
    "checks.json",
    "quake.py",
    "braille.py",
    "bridge.py",
    "cases.json.gz",
    "context_pairs.json.gz",
    "weights.npz",
    "freeze.json",
)
OUTPUTS = ("mosaic-results.json", "mosaic-predictions.npz")


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
    return len(record[{"quake": "stations", "braille": "cells", "bridge": "deals"}[task]])


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


def check_context_pairs(archive, shapes, reported, protocol, lock, seen):
    blob = (HERE / "context_pairs.json.gz").read_bytes()
    require(digest(blob) == lock["context_pairs_sha256"], "Paired inputs differ from the freeze")
    cases = json.loads(gzip.decompress(blob))
    spec = protocol["context_pairs"]
    require(set(cases) == {str(n) for n in spec["sizes"]}, "Paired test sizes differ")
    rows = []
    for size in spec["sizes"]:
        pairs = cases[str(size)]
        require(len(pairs) == spec["pairs_per_size"], "Paired case count differs")
        targets, terminal = [], Counter()
        for numeric, capital in pairs:
            n, c = numeric["cells"], capital["cells"]
            require(
                len(n) == len(c) == size
                and n[0] == "3456"
                and c[0] == "6"
                and n[1:] == c[1:]
                and all(dot in LETTERS[:10] for dot in n[1:]),
                "Invalid matched pair",
            )
            for line in (numeric, capital):
                fingerprint = canonical(line)
                require(fingerprint not in seen, "Paired line overlaps another input")
                seen.add(fingerprint)
            # Decode expected values directly from the structure of this test.
            ids = [LETTERS.index(dot) for dot in n[1:]]
            nd = [0] + [54 + (i + 1) % 10 for i in ids]
            cd = [0, 28 + ids[0], *[2 + i for i in ids[1:]]]
            require(
                nd == ORACLES["braille"](numeric)[0] and cd == ORACLES["braille"](capital)[0],
                "Paired targets disagree with sequential reader",
            )
            targets.append((nd, cd))
            terminal[ids[-1]] += 1
        require(terminal == {i: len(pairs) // 10 for i in range(10)}, "Unbalanced final patterns")
        key = "context_" + str(size)
        ok = check_prediction(archive[key], targets, 64, shapes[key], key)
        rows.append(
            {
                "size": size,
                "pairs": len(pairs),
                "exact_pairs": int(ok.all(axis=(1, 2)).sum()),
                "both_final_correct": int(ok[:, :, -1].all(axis=1).sum()),
            }
        )
    require(rows == reported["context_pairs"], "Paired GPU scores differ")
    return rows


def evaluate(run):
    reported = read(run / "output" / OUTPUTS[0])
    receipt, shapes = authenticate_run(
        run, reported, folder=HERE, inputs=INPUTS, outputs=OUTPUTS, stage_name="mosaic"
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
    quake_counts, quake_exact = Counter(), Counter()
    verified, rows = {}, []
    with np.load(run / "output" / OUTPUTS[1], allow_pickle=False) as archive:
        expected = {f"{t}_{n}" for t, n in pairs} | {"check_" + name for name in names}
        expected.update(f"context_{size}" for size in protocol["context_pairs"]["sizes"])
        require(set(archive.files) == set(shapes) == expected, "Prediction archive coverage differs")
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
                            target == case["gold"] and len(categories) == len(target),
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
                    if task == "quake":
                        for target, correct in zip(targets, ok.all(axis=1), strict=True):
                            count = sum(label > 0 for label in target)
                            category = "none" if count == 0 else "unique" if count == 1 else "multiple"
                            quake_counts[category] += 1
                            quake_exact[category] += int(correct)
                    for labels, flags in zip(groups, ok, strict=True):
                        for categories, correct in zip(labels, flags, strict=True):
                            counts[task].update(categories)
                            if correct:
                                correct_counts[task].update(categories)
                    rows.append(dict(row))
        context = check_context_pairs(archive, shapes, reported, protocol, lock, seen["braille"])
    return {
        "run": receipt,
        "context_pairs": context,
        "quake_records": quake_counts,
        "exact_quake_records": quake_exact,
        "models": models,
        "conditions": rows,
        "worked_checks": verified,
        "labels_by_category": counts,
        "correct_labels_by_category": correct_counts,
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
