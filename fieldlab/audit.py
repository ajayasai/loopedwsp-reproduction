"""Independently verify and regrade the frozen field applications on a CPU."""
import argparse
from collections import Counter, deque
import gzip
import json
from pathlib import Path
from statistics import median
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_evidence import authenticate_run, digest, read, require

INPUTS = ("experiment.py", "protocol.json", "basin.py", "transit.py", "bowling.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("fieldlab-results.json", "fieldlab-predictions.npz")
SIZES = {"basin": [4, 6, 8, 12], "transit": [24, 48, 96, 192], "bowling": [3, 5, 8, 10]}
CLASSES = {"basin": 10, "transit": 29, "bowling": 302}
TEMPLATES = [(p, phase) for p in (3, 4, 5, 7, 9) for phase in range(p)]
PENDING = 301


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def solve_basin(record, size):
    grid = np.asarray(record["heights"])
    require(set(record) == {"heights"} and grid.shape == (size, size) and grid.dtype.kind in "iu"
            and grid.min() >= 0 and grid.max() <= 9, "Invalid elevation raster")
    depth = np.zeros_like(grid)
    # At water elevation h, any cell at or below h without a path to the rim
    # at or below h contributes one unit to its final water column.
    for h in range(9):
        wettable = grid <= h
        seen = np.zeros_like(wettable)
        seen[0, :], seen[-1, :] = wettable[0, :], wettable[-1, :]
        seen[:, 0], seen[:, -1] = wettable[:, 0], wettable[:, -1]
        frontier = deque(zip(*np.nonzero(seen)))
        while frontier:
            y, x = frontier.popleft()
            for r, c in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= r < size and 0 <= c < size and wettable[r, c] and not seen[r, c]:
                    seen[r, c] = True
                    frontier.append((r, c))
        depth += wettable & ~seen
    local = np.zeros_like(grid)
    neighboring_min = np.minimum.reduce([grid[:-2, 1:-1], grid[2:, 1:-1], grid[1:-1, :-2], grid[1:-1, 2:]])
    local[1:-1, 1:-1] = np.maximum(0, neighboring_min - grid[1:-1, 1:-1])
    wet = bool(depth.any())
    return depth.ravel(), {"wet": wet, "dry": not wet, "local_heuristic_wrong": not np.array_equal(local, depth)}


def solve_transit(record, size):
    base, flux = record["baseline"], record["flux"]
    require(set(record) == {"baseline", "flux"} and type(base) is int and 16 <= base <= 4095
            and isinstance(flux, list) and len(flux) == size and
            all(v is None or (type(v) is int and abs(v - base) <= 16) for v in flux), "Invalid brightness record")
    observed = [(t, value) for t, value in enumerate(flux) if value is not None]
    flat_sse = sum((value - base) ** 2 for _, value in observed)
    best_id, best_drop = 0, 0
    for i, (period, phase) in enumerate(TEMPLATES, 1):
        model_sse = supporters = 0
        for t, value in observed:
            at_transit = (t - phase) % period == 0
            model_flux = base - 8 if at_transit else base
            model_sse += (value - model_flux) ** 2
            supporters += at_transit and value <= base - 5
        drop = flat_sse - model_sse
        if supporters >= 3 and drop >= 12 * 16 and drop > best_drop:
            best_id, best_drop = i, drop
    return np.array([best_id]), {}


def solve_bowling(record, size):
    frames = record["frames"]
    require(set(record) == {"frames"} and isinstance(frames, list) and len(frames) == size, "Invalid card size")
    points = [0] * size
    owed = [0] * size
    partial = [False] * size
    for i, frame in enumerate(frames):
        require(isinstance(frame, list) and 1 <= len(frame) <= 3 and
                all(type(p) is int and 0 <= p <= 10 for p in frame), "Invalid credited pin counts")
        strike = frame[0] == 10
        if len(frame) >= 2 and not strike:
            require(sum(frame[:2]) <= 10, "Invalid second roll")
        spare = not strike and len(frame) >= 2 and sum(frame[:2]) == 10
        needed = (3 if strike or spare else 2) if i == 9 else (1 if strike else 2)
        require(len(frame) <= needed and (i == size-1 or len(frame) == needed), "Invalid frame completion")
        if i == 9 and len(frame) == 3 and strike and frame[1] != 10:
            require(sum(frame[1:]) <= 10, "Invalid fill rack")
        for pins in frame:
            points[i] += pins
            for prior in range(i):
                if owed[prior]:
                    points[prior] += pins
                    owed[prior] -= 1
        partial[i] = len(frame) < needed
        if i != 9:
            owed[i] = 2 if strike else int(spare)
    cumulative = np.cumsum(points)
    unsettled = np.logical_or.accumulate(np.array(partial) | (np.array(owed) > 0))
    answer = np.where(unsettled, PENDING, cumulative)
    pending = bool(unsettled.any())
    return answer, {"pending": pending, "fully_posted": not pending,
                    "complete_ten_frame_games": size == 10 and not pending}


ORACLES = {"basin": solve_basin, "transit": solve_transit, "bowling": solve_bowling}


def model_ledgers(lock):
    expected = {
        "basin": {"block.query": -128., "block.value": 1., "block.residual": [1., -1.], "readout": [1., -1.]},
        "transit": {"address": [256., -128.], "thresholds": [4., 5.], "value": [[1., 0.], [-4., 0.], [0., 1.], [0., -1.]]},
        "bowling": {"position.projection": [128., -64.], "own_pin_weights": [1., 1., 1.], "status_weight": [2., -1.]},
    }
    require(set(lock["models"]) == set(expected), "Unexpected model names")
    ledgers = {}
    with np.load(HERE / "weights.npz", allow_pickle=False) as weights:
        keys = {f"{task}/{name}" for task, tensors in expected.items() for name in tensors}
        require(len(weights.files) == len(keys) and set(weights.files) == keys, "Unexpected weight archive keys")
        for task, tensors in expected.items():
            model = lock["models"][task]
            require(set(model["tensors"]) == set(tensors), f"Unexpected tensors for {task}")
            count = nonzero = 0
            for name, values in tensors.items():
                a = weights[f"{task}/{name}"]
                info = {"shape": list(a.shape), "dtype": str(a.dtype), "parameter": True,
                        "nonzero": int(np.count_nonzero(a)), "sha256": digest(a.tobytes())}
                require(a.dtype == np.float32 and np.array_equal(a, np.asarray(values, dtype=np.float32))
                        and model["tensors"][name] == info, f"Unexpected weight tensor {task}/{name}")
                count += a.size
                nonzero += info["nonzero"]
            require(count == model["parameters"], f"Wrong parameter count in {task}")
            ledgers[task] = {"parameters": count, "nonzero": nonzero, "tensors": len(tensors)}
    return ledgers


def frozen():
    lock = read(HERE / "freeze.json")
    for short, name in (("cases", "cases.json.gz"), ("weights", "weights.npz"), ("protocol", "protocol.json")):
        require(lock[short + "_sha256"] == digest((HERE / name).read_bytes()), f"{name} differs from freeze")
    protocol = read(HERE / "protocol.json")
    require(protocol["seed"] == 2026112503 and protocol["repeats"] == 3 and protocol["batch"] == 8
            and protocol["per_size"] == {"dev": 8, "test": 64} and set(protocol["tasks"]) == set(ORACLES)
            and all(protocol["tasks"][t]["classes"] == CLASSES[t] and protocol["tasks"][t]["test"] == sizes
                    and protocol["tasks"][t]["dev"] == sizes[:2] for t, sizes in SIZES.items()), "Unexpected protocol")
    ledgers = model_ledgers(lock)
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))
    expected = {f"{split}/{t}/{n}" for t, sizes in SIZES.items() for split in ("dev", "test")
                for n in (sizes[:2] if split == "dev" else sizes)}
    require(set(cases) == set(lock["cohorts"]) == expected, "Unexpected cohort set")
    seen = {t: set() for t in ORACLES}
    labels, traits = {}, {}
    for key, rows in cases.items():
        split, task, size = key.split("/")
        require(len(rows) == protocol["per_size"][split] and digest(packed(rows)) == lock["cohorts"][key],
                f"Invalid cohort {key}")
        answers, properties = [], []
        for row in rows:
            require(set(row) == ({"input", "gold", "injected"} if task == "transit" else {"input", "gold"}),
                    f"Unexpected data alongside model input in {key}")
            value = row["input"]
            answer, prop = ORACLES[task](value, int(size))
            gold = np.asarray(row["gold"])
            require(gold.dtype.kind in "iu" and np.array_equal(answer, gold), f"Oracle disagreement in {key}")
            # All three record encodings are injective on the validated raw schema.
            fingerprint = digest(packed(value))
            require(fingerprint not in seen[task], f"Duplicate model input in {key}")
            seen[task].add(fingerprint)
            if task == "transit" and row["injected"] is not None:
                tag = row["injected"]
                require(set(tag) == {"period", "phase"} and all(type(x) is int for x in tag.values())
                        and (tag["period"], tag["phase"]) in TEMPLATES, "Invalid injection metadata")
            answers.append(answer)
            properties.append(prop)
        labels[key] = np.stack(answers)
        traits[key] = {name: np.array([p[name] for p in properties]) for name in properties[0]}
    return lock, ledgers, cases, labels, traits


def science_metrics(rows, predictions):
    counts = dict.fromkeys(("injected", "recovered", "missed", "wrong_template", "indistinguishable_wrong_template",
                            "uninjected", "false_alarms", "invalid", "invalid_injected", "invalid_uninjected"), 0)
    for row, cls in zip(rows, predictions[:, 0], strict=True):
        tag = row["injected"]
        counts["uninjected" if tag is None else "injected"] += 1
        if cls < 0:
            counts["invalid"] += 1
            counts["invalid_uninjected" if tag is None else "invalid_injected"] += 1
        elif tag is None:
            counts["false_alarms"] += int(cls > 0)
        elif cls == 0:
            counts["missed"] += 1
        else:
            p1, phase1 = TEMPLATES[cls-1]
            p2, phase2 = tag["period"], tag["phase"]
            if (p1, phase1) == (p2, phase2):
                counts["recovered"] += 1
            else:
                counts["wrong_template"] += 1
                identical = all((i % p1 == phase1) == (i % p2 == phase2)
                                for i, v in enumerate(row["input"]["flux"]) if v is not None)
                counts["indistinguishable_wrong_template"] += int(identical)
    return counts


def evaluate(run):
    results = read(run / "output/fieldlab-results.json")
    run_info, shapes = authenticate_run(run, results, folder=HERE, inputs=INPUTS, outputs=OUTPUTS, stage_name="fieldlab")
    lock, ledgers, cases, labels, traits = frozen()
    require(all(lock[k] == results[k] for k in ("weights_sha256", "cases_sha256")), "Unexpected run inputs")
    pairs = [(task, n) for task, sizes in SIZES.items() for n in sizes]
    rows = {(r["task"], r["size"]): r for r in results["conditions"]}
    require(len(results["conditions"]) == len(pairs) and set(rows) == set(pairs), "Invalid GPU coverage")
    checked, science = [], []
    totals = {"basin": Counter(), "bowling": Counter()}
    with np.load(run / "output/fieldlab-predictions.npz", allow_pickle=False) as archive:
        names = {f"{t}_{n}" for t, n in pairs}
        require(set(shapes) == set(archive.files) == names and len(archive.files) == len(names), "Prediction names differ")
        for t, n in pairs:
            key, name = f"test/{t}/{n}", f"{t}_{n}"
            gold, pred = labels[key], archive[name]
            require(pred.dtype == np.int16 and pred.shape == gold.shape and list(pred.shape) == shapes[name]
                    and pred.min() >= -1 and pred.max() < CLASSES[t], f"Invalid predicted classes in {name}")
            ok = pred == gold
            exact = ok.all(axis=1)
            row = rows[t, n]
            scores = {"parameters": ledgers[t]["parameters"], "records": len(gold), "exact_records": int(exact.sum()),
                      "labels": gold.size, "correct_labels": int(ok.sum()), "nonfinite": int((pred < 0).sum())}
            require(all(row[k] == v for k, v in scores.items()), f"Wrong score counts in {name}")
            times = row["seconds"]
            require(len(times) == 3 and all(np.isfinite(s) and s > 0 for s in times) and
                    row["median_seconds"] == median(times), f"Unexpected timing in {name}")
            categories = {}
            if t == "transit":
                science.append({"size": n, **science_metrics(cases[key], pred)})
            else:
                for category, mask in traits[key].items():
                    categories[category] = int(mask.sum())
                    categories[category + "_correct"] = int((mask & exact).sum())
                totals[t].update(categories)
            checked.append({"task": t, "size": n, **scores, "seconds": times,
                            "median_seconds": median(times), "categories": categories})
    return {"verified": True, "run": run_info, "model_ledgers": ledgers, "frozen_cohorts_checked": len(labels),
            "development_records_checked": sum(len(v) for k, v in labels.items() if k.startswith("dev/")),
            "test_records_checked": sum(r["records"] for r in checked), "labels_regraded": sum(r["labels"] for r in checked),
            "conditions": checked, "test_categories": {k: dict(v) for k, v in totals.items()}, "science": science}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    audit = evaluate(args.version_folder.resolve())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_bytes((json.dumps(audit, indent=2, allow_nan=False) + "\n").encode())
    print(f"Verified {len(audit['conditions'])} GPU settings and regraded {audit['labels_regraded']:,} labels "
          f"on {audit['test_records_checked']} held-out records.")
