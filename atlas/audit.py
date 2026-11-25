"""Recompute labels from raw inputs and check the archived Kaggle run without PyTorch."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_evidence import authenticate_run, digest, read, require

INPUTS = ("experiment.py", "protocol.json", "checks.json", "rings.py", "loom.py", "braid.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("atlas-results.json", "atlas-predictions.npz")


def rings(record):
    # Cross-multiply integer totals to compare two fractions, including exact
    # ties with different denominators. Mark a start if no eligible rival wins.
    def changes(widths):
        return [0 if a is None or b is None else int(b > a)-int(b < a) for a,b in zip(widths,widths[1:])]

    sample = np.asarray(changes(record["sample"]))
    windows = np.lib.stride_tricks.sliding_window_view(changes(record["reference"]), len(sample))
    paired = windows * sample
    usable = np.count_nonzero(paired, axis=1)
    agree = (paired == 1).sum(axis=1)
    eligible = usable >= 4
    advantage = agree[:,None]*usable - usable[:,None]*agree
    beaten = (advantage[eligible] > 0).any(axis=0)
    labels = (eligible.astype(int) + (eligible & ~beaten)).tolist()
    winners = labels.count(2)
    return labels, ("insufficient" if winners == 0 else "unique" if winners == 1 else "tied")


def loom(record):
    # Expand the tie-up into individual (treadle, shaft) connections.
    ties = {(treadle, shaft) for treadle, shafts in enumerate(record["tie_up"], 1) for shaft in shafts}
    labels = [int(any((treadle, shaft) in ties for treadle in pick))
              for pick in record["treadling"] for shaft in record["threading"]]
    return labels, ("compound" if any(len(pick) > 1 for pick in record["treadling"]) else "simple")


def braid(record):
    # Give each new under-arc its own variable. Impose one Fox equation per
    # crossing plus three equations identifying bottom endpoints with the top.
    word = record["word"]
    dimension = 3 + len(word)
    system = np.zeros((dimension, dimension), dtype=np.int64)
    positions = [0, 1, 2]
    for row, crossing in enumerate(word):
        left = abs(crossing)-1
        over, under = (left, left+1) if crossing > 0 else (left+1, left)
        old_over, old_under = positions[over], positions[under]
        new_arc = 3 + row
        system[row, old_over] += 2
        system[row, old_under] -= 1
        system[row, new_arc] -= 1
        positions[over], positions[under] = new_arc, old_over
    for i, arc in enumerate(positions):
        system[len(word)+i, arc] += 1
        system[len(word)+i, i] -= 1

    system %= 3
    rank = 0
    for col in range(dimension):
        candidates = np.flatnonzero(system[rank:, col])
        if not candidates.size:
            continue
        pivot = rank + candidates[0]
        system[[rank,pivot]] = system[[pivot,rank]]
        system[rank] = system[rank] * system[rank,col] % 3
        system[rank+1:] = (system[rank+1:] - system[rank+1:,col,None] * system[rank]) % 3
        rank += 1
    colorings = 3 ** (dimension-rank)
    return [colorings], str(colorings)


ORACLES = {"rings": rings, "loom": loom, "braid": braid}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fingerprint(task, record):
    if task != "loom":
        return canonical(record)
    # The loom input encodes each row as a set of switches with eight treadle slots.
    ties = [sorted(row) for row in record["tie_up"]]
    return canonical([record["threading"], ties + [[]]*(8-len(ties)), [sorted(row) for row in record["treadling"]]])


def unique_affine_start(record):
    sample, ref = record["sample"], record["reference"]
    if None in sample:
        return None
    matches = []
    for start in range(len(ref)-len(sample)+1):
        window = ref[start:start+len(sample)]
        if None in window:
            continue
        for scale in (1,2):
            shift = sample[0] - scale*window[0]
            if 1 <= shift <= 29 and all(a == scale*b+shift for a,b in zip(sample,window,strict=True)):
                matches.append(start)
                break
    return matches[0] if len(matches) == 1 else None


def size_of(task, record):
    if task == "loom":
        require(len(record["threading"]) == len(record["treadling"]), "Benchmark draft must be square")
    return len(record[{"loom":"threading", "rings":"reference", "braid":"word"}[task]])


def check_weights(lock):
    counts = {}
    with np.load(HERE / "weights.npz", allow_pickle=False) as weights:
        wanted = {f"{task}/{name}" for task, spec in lock["models"].items() for name in spec["tensors"]}
        require(set(weights.files) == wanted, "Tensor list differs from the freeze")
        for task, spec in lock["models"].items():
            total = nonzero = 0
            for name, entry in spec["tensors"].items():
                t = weights[f"{task}/{name}"]
                nz = int(np.count_nonzero(t))
                require(list(t.shape) == entry["shape"] and str(t.dtype) == entry["dtype"]
                        and digest(t.tobytes()) == entry["sha256"] and nz == entry["nonzero"]
                        and np.isfinite(t).all(), f"Invalid tensor {task}/{name}")
                if entry["parameter"]:
                    require(t.dtype == np.float32, "Unexpected parameter precision")
                    total += t.size
                    nonzero += nz
            require(total == spec["parameters"], f"Parameter count differs for {task}")
            counts[task] = {"parameters": total, "nonzero": nonzero}
    return counts


def evaluate(run):
    reported = read(run / "output" / OUTPUTS[0])
    identity, shapes = authenticate_run(run, reported, folder=HERE, inputs=INPUTS, outputs=OUTPUTS, stage_name="atlas")
    protocol, lock, checks = (read(HERE / name) for name in ("protocol.json", "freeze.json", "checks.json"))
    for stem, ext in (("weights","npz"), ("cases","json.gz"), ("checks","json"), ("protocol","json")):
        require(digest((HERE / f"{stem}.{ext}").read_bytes()) == lock[stem+"_sha256"], f"Freeze differs for {stem}")
    require(all(reported[k] == lock[k] for k in ("weights_sha256","cases_sha256")), "Result inputs differ")
    models = check_weights(lock)
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))
    tasks = protocol["tasks"]
    pairs = [(t, n) for t, cfg in tasks.items() for n in cfg["test"]]
    expected = {f"{split}/{t}/{n}" for t,cfg in tasks.items() for split in ("dev","test") for n in cfg[split]}
    require(set(cases) == set(lock["cohorts"]) == expected, "Unexpected corpus coverage")
    reports = {(r["task"],r["size"]): r for r in reported["conditions"]}
    require(len(reported["conditions"]) == len(pairs) and set(reports) == set(pairs), "Unexpected GPU coverage")
    seen = {t:{fingerprint(t, read(HERE / "examples" / f"{t}.json"))} for t in tasks}
    names = {c["name"] for c in checks}
    require(len(names) == len(checks) and set(reported["checks"]) == names, "Worked check coverage differs")
    categories = {t:Counter() for t in tasks}
    categories_exact = {t:Counter() for t in tasks}
    diagnostics = dict.fromkeys(("unique_affine_windows", "affine_offset_among_best", "affine_offset_sole_best",
                                 "first_treadle_changes_draft", "changed_drafts_exact"), 0)
    worked, conditions = {}, []
    with np.load(run / "output" / OUTPUTS[1], allow_pickle=False) as raw:
        keys = {f"{t}_{n}" for t,n in pairs} | {"check_"+name for name in names}
        require(set(raw.files) == set(shapes) == keys, "Prediction arrays differ from manifest")
        def prediction(key, task, target):
            a = raw[key]
            require(a.dtype == np.int16 and list(a.shape) == shapes[key] and a.shape == np.shape(target)
                    and (a >= -1).all() and (a < tasks[task]["classes"]).all(), f"Invalid predictions for {key}")
            return a
        for c in checks:
            task, name = c["task"], c["name"]
            labels, category = ORACLES[task](c["input"])
            require(labels == c["labels"], f"Worked oracle disagrees for {name}")
            seen[task].add(fingerprint(task, c["input"]))
            a = prediction("check_"+name, task, labels)
            correct = bool(np.array_equal(a, labels))
            require(reported["checks"][name] == {"task":task, "correct":correct}, f"Worked score differs for {name}")
            worked[name] = {"task": task, "correct": correct, "category": category, "labels":len(labels)}

        for split in ("dev", "test"):
            for task, cfg in tasks.items():
                for size in cfg[split]:
                    key = f"{split}/{task}/{size}"
                    batch = cases[key]
                    require(len(batch) == protocol["per_size"][split] and digest(canonical(batch)) == lock["cohorts"][key],
                            f"Frozen cases differ for {key}")
                    targets, groups = [], []
                    for item in batch:
                        record = item["input"]
                        marker = fingerprint(task, record)
                        require(marker not in seen[task] and size_of(task,record) == size, f"Repeated or mis-sized {key}")
                        seen[task].add(marker)
                        target, group = ORACLES[task](record)
                        require(item["gold"] == target, f"Independent reference disagrees for {key}")
                        targets.append(target)
                        groups.append(group)
                    if split == "dev":
                        continue
                    a = prediction(f"{task}_{size}", task, targets)
                    correct = a == np.asarray(targets)
                    exact = correct.all(axis=1)
                    row = reports[task,size]
                    scores = {"parameters":models[task]["parameters"], "records":len(a), "exact_records":int(exact.sum()),
                              "labels":a.size, "correct_labels":int(correct.sum()), "nonfinite":int((a < 0).sum())}
                    require(all(row[k] == v for k,v in scores.items()), f"Reported scores differ for {key}")
                    times = row["seconds"]
                    require(len(times) == protocol["repeats"] and all(np.isfinite(t) and t>0 for t in times)
                            and row["median_seconds"] == median(times), f"Invalid timing for {key}")
                    categories[task].update(groups)
                    categories_exact[task].update(g for g,passed in zip(groups, exact, strict=True) if passed)
                    for item, pred, target, passed in zip(batch, a, targets, exact, strict=True):
                        record = item["input"]
                        if task == "rings":
                            start = unique_affine_start(record)
                            if start is not None:
                                best = np.flatnonzero(pred == 2).tolist()
                                diagnostics["unique_affine_windows"] += 1
                                diagnostics["affine_offset_among_best"] += start in best
                                diagnostics["affine_offset_sole_best"] += best == [start]
                        elif task == "loom":
                            single = {**record, "treadling":[pick[:1] for pick in record["treadling"]]}
                            if loom(single)[0] != target:
                                diagnostics["first_treadle_changes_draft"] += 1
                                diagnostics["changed_drafts_exact"] += int(passed)
                    conditions.append(dict(row))
    return {"run":identity, "models":models, "conditions":conditions, "worked_checks":worked,
            "records_by_category":categories, "exact_records_by_category":categories_exact, "diagnostics":diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    evidence = evaluate(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(evidence, indent=2, allow_nan=False)+"\n").encode())
    rows = evidence["conditions"]
    print(f"Regraded {sum(r['records'] for r in rows)} records and {sum(r['labels'] for r in rows):,} labels")
