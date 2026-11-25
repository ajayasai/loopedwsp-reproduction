"""Authenticate a collected creative-applications run and regrade it with NumPy."""
import argparse
import ast
import base64
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
INPUTS = ("experiment.py", "folds.py", "typeset.py", "ballots.py", "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("creative-results.json", "creative-predictions.npz")
PLAN = {"seed": 2026112501, "dev": {"sizes": [4, 8], "records": 8},
        "test": {"sizes": [4, 8, 16, 32], "records": 64}, "batch_size": 8, "repeats": 3}
CLASSES = {"folds": 8, "typeset": 2, "ballots": 5}
SIDES = (("left", "right"), ("right", "left"), ("top", "bottom"), ("bottom", "top"))
COMMANDS = [f"{moving}_{motion}_{rest}" for moving, rest in SIDES for motion in ("under", "over")]


def require(condition, issue):
    if not condition:
        raise ValueError(issue)


def read(path):
    return json.loads(path.read_bytes())


def digest(blob):
    return sha256(blob).hexdigest()


def packed(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def authenticate(run, results):
    receipt = read(run / "submission.json")
    requested = read(run / "kernel-metadata.json")
    remote = read(run / "remote/kernel-metadata.json")
    verified = read(run / "verification.json")
    runtime = read(run / "output/creative-runtime.json")
    kernel, version = receipt["kernel"], receipt["version"]
    require(kernel == requested["id"] == remote["id"] == verified["kernel"]
            and version == remote["version"] == verified["version"] and run.name == f"v{version}"
            and receipt["url"] == f"https://www.kaggle.com/code/{kernel}/versions/{version}", "Kaggle identity differs")
    require(receipt["requested_private"] is verified["server_private"] is True
            and receipt["requested_internet"] is verified["server_internet_enabled"] is False
            and all(meta["is_private"] is meta["enable_gpu"] is True
                    and meta["enable_tpu"] is meta["enable_internet"] is False for meta in [requested, remote])
            and requested["kernel_type"] == "script" and requested["code_file"] == "main.py"
            and receipt["requested_gpu"] == remote["machine_shape"] == "NvidiaTeslaT4", "Kaggle settings differ")
    require(runtime["status"] == "complete" and verified["status"] == "COMPLETE"
            and results["split"] == "test" and runtime["available_devices"] >= 1
            and runtime["device"] == verified["device"] == results["device"] and "T4" in runtime["device"]
            and runtime["pytorch"] == verified["pytorch"] == results["torch"]
            and runtime["cuda"] == verified["cuda"] == results["cuda"], "Reported device or completion differs")
    script = (run / "main.py").read_bytes()
    script_hash = digest(script)
    require(script_hash == receipt["script_sha256"] == verified["script_sha256"] and len(script) == receipt["script_bytes"],
            "Archived script differs from the submission")
    server_hash = remote["downloaded_source_sha256"]
    require(server_hash == verified["remote_script_sha256"] and remote["source_match"] == verified["source_match"],
            "Server source/verification mismatch")
    if verified["source_match"] == "exact":
        require(script_hash == server_hash, "Server source differs from submitted script")
    else:
        normalized = script.replace(b"\r\n", b"\n")
        require(verified["source_match"] == "line_endings" and server_hash != script_hash and
                server_hash in {digest(normalized), digest(normalized.replace(b"\n", b"\r\n"))},
                "Server source differs beyond line endings")
    last = ast.parse(script).body[-1].value
    require(isinstance(last, ast.Call) and isinstance(last.func, ast.Name) and last.func.id == "remote_run"
            and len(last.args) == 4 and not last.keywords, "Unexpected Kaggle entry point")
    payload, declared, outputs, stage = (ast.literal_eval(a) for a in last.args)
    hashes = {name: digest((HERE / name).read_bytes()) for name in INPUTS}
    embedded = {name: digest(zlib.decompress(base64.b64decode(blob, validate=True))) for name, blob in payload.items()}
    require(stage == "creative" and outputs == list(OUTPUTS) and verified["input_files_verified"] == len(INPUTS)
            and embedded == declared == hashes == receipt["input_sha256"] == runtime["input_sha256"],
            "Submitted input bytes or output declaration differ")
    output_hashes = {name: digest((run / "output" / name).read_bytes()) for name in OUTPUTS}
    require(output_hashes == verified["output_sha256"] == runtime["output_sha256"], "GPU output hash mismatch")
    return {"kernel": kernel, "version": version, "url": receipt["url"], "device": runtime["device"],
            "pytorch": runtime["pytorch"], "cuda": runtime["cuda"], "status": "COMPLETE",
            "source_match": remote["source_match"], "submitted_sha256": script_hash, "server_sha256": server_hash,
            "input_sha256": hashes, "output_sha256": output_hashes}, verified["prediction_arrays"]


def model_ledger(lock):
    # Reconstruct all stored coefficients from the three analytic designs.
    f32 = lambda x: np.asarray(x, dtype=np.float32)
    expected = {
        "folds": {"block.query": np.diag(f32([2,1])), "block.key": np.diag(f32([16,-16])),
                  "block.surface_routes": np.eye(4, dtype=np.float32)[[0,3,1,2]], "visible": f32([1,0])},
        "typeset": {"block.address": f32([-128,128]), "block.threshold": -np.arange(32, dtype=np.float32),
                    "block.spline": f32([1] + [2]*31)},
        "ballots": {"preference_query": f32([32,160]), "elimination_query": f32([-160,-32,160])},
    }
    summary = {}
    require(set(expected) == set(lock["models"]), "Unexpected task in freeze")
    with np.load(HERE / "weights.npz", allow_pickle=False) as weights:
        all_names = {f"{task}/{name}" for task, tensors in expected.items() for name in tensors}
        require(len(weights.files) == len(all_names) and set(weights.files) == all_names, "Unexpected weight archive keys")
        for task, tensors in expected.items():
            manifest = lock["models"][task]
            require(set(manifest["tensors"]) == set(tensors), f"{task} model tensors differ")
            total = 0
            for name, expected_value in tensors.items():
                arr = weights[f"{task}/{name}"]
                entry = {"shape": list(arr.shape), "dtype": str(arr.dtype), "count": arr.size, "parameter": True,
                         "sha256": digest(arr.tobytes()), "nonzero": int(np.count_nonzero(arr))}
                require(entry == manifest["tensors"][name] and entry["dtype"] == "float32"
                        and arr.shape == expected_value.shape and arr.tobytes() == expected_value.tobytes(),
                        f"Unexpected tensor {task}/{name}")
                total += arr.size
            require(total == manifest["parameters"], f"{task} parameter count differs")
            summary[task] = {"parameters": total, "tensors": len(tensors)}
    return summary


def folded_colors(record, size):
    """Trace each visible pixel backward to its original position and face."""
    front, back = (np.asarray(record[face]) for face in ["front", "back"])
    require(all(grid.shape == (size,size) and grid.dtype.kind in "iu" and grid.min() >= 0 and grid.max() < 8
                for grid in [front, back]), "Invalid paper colors or cohort dimensions")
    folds = record["folds"]
    height, width = front.shape
    history = []
    for cmd in folds:
        require(cmd in COMMANDS, f"Unknown crease {cmd}")
        moving, motion, _ = cmd.split("_")
        vertical = moving in ("top", "bottom")
        length = height if vertical else width
        require(length >= 2 and length % 2 == 0, "Crease does not bisect the current sheet")
        history.append((vertical, length, moving in ("left", "top"), motion == "over"))
        if vertical:
            height //= 2
        else:
            width //= 2
    row, col = np.indices((height, width))
    front_up = True
    for vertical, length, moving_low, over in reversed(history):
        cell = row if vertical else col
        on_stationary = cell + (length // 2 if moving_low else 0)
        on_moving = (over == front_up)
        original = length - 1 - on_stationary if on_moving else on_stationary
        if on_moving:
            front_up = not front_up
        if vertical:
            row = original
        else:
            col = original
    answer = (front if front_up else back)[row,col].ravel()
    identity = [front.tolist(), back.tolist(), folds]  # Recoverable from the encoded token matrix at fixed size.
    return answer, identity, {"original_back_visible": int(not front_up)}


def paragraph_breaks(record, size):
    """Forward DAG relaxation, keeping the lexicographically longest break sequence on ties."""
    words, width = record["words"], record["width"]
    require(type(width) is int and 1 <= width <= 32 and len(words) == size
            and all(isinstance(w,str) and w.isascii() and w.isprintable() and not any(c.isspace() for c in w)
                    and 1 <= len(w) <= width for w in words), "Invalid paragraph or cohort dimensions")
    lengths = [len(word) for word in words]
    # Paths store negative ends so the minimum is the leftmost longest line.
    states = {0: (0, ())}
    ways = {0: 1}
    for end in range(1, size+1):
        for start in range(end):
            used = sum(lengths[start:end]) + end-start-1
            if used > width:
                continue
            previous_cost, previous_path = states[start]
            cost = previous_cost + ((width-used)**2 if end < size else 0)
            option = (cost, previous_path + (-end,))
            if end not in states or cost < states[end][0]:
                ways[end] = ways[start]
                states[end] = option
            elif cost == states[end][0]:
                ways[end] += ways[start]
                states[end] = min(states[end], option)
    best_cost, breaks = states[size]
    flags = np.zeros(size, dtype=np.int64)
    flags[-np.asarray(breaks)-1] = 1
    occupied, greedy_cost = 0, 0
    for length in lengths:
        if occupied and occupied+1+length > width:
            greedy_cost += (width-occupied)**2
            occupied = 0
        occupied = occupied+1+length if occupied else length
    return flags, [width, lengths], {"greedy_cost_above_optimum": int(greedy_cost > best_cost),
                                     "multiple_minimum_cost_layouts": int(ways[size] > 1)}


def elimination_order(record, size):
    """Assign each ballot to its minimum surviving rank; use integer histograms for the tally."""
    films, ballots = record["films"], record["ballots"]
    require(len(films) == len(set(films)) == 5 and all(isinstance(title,str) and title.strip() for title in films)
            and len(ballots) == size and all(len(row) == 5 and set(row) == set(films) for row in ballots),
            "Invalid election or cohort dimensions")
    ids = np.asarray([[films.index(title) for title in row] for row in ballots])
    ranks = np.argsort(ids, axis=1)
    remaining = list(range(5))
    gone = []
    initial_leaders = set()
    had_tie = False
    while len(remaining) > 1:
        choices = ranks[:, remaining].argmin(axis=1)
        counts = np.bincount(choices, minlength=len(remaining))
        if not gone:
            initial_leaders = {remaining[i] for i in np.flatnonzero(counts == counts.max())}
        had_tie |= (counts == counts.min()).sum() > 1
        gone.append(remaining.pop(int(counts.argmin())))
    return np.asarray(gone + remaining), ids.tolist(), {"elimination_tie": int(had_tie),
            "unique_first_choice_leader_lost": int(len(initial_leaders) == 1 and remaining[0] not in initial_leaders)}


ORACLES = {"folds": folded_colors, "typeset": paragraph_breaks, "ballots": elimination_order}


def cases_and_weights():
    lock = read(HERE / "freeze.json")
    blob = (HERE / "cases.json.gz").read_bytes()
    require(lock["plan"] == PLAN and lock["cases_sha256"] == digest(blob)
            and lock["weights_sha256"] == digest((HERE / "weights.npz").read_bytes()), "Freeze or corpus differs")
    models = model_ledger(lock)
    cases = json.loads(gzip.decompress(blob))
    expected = {f"{split}/{task}/{size}" for split in ("dev", "test") for task in ORACLES for size in PLAN[split]["sizes"]}
    require(set(cases) == set(lock["cohorts"]) == expected, "Frozen cohorts differ")
    labels, properties, seen = {}, {}, {task:set() for task in ORACLES}
    for key, rows in cases.items():
        split, task, size = key.split("/")
        require(len(rows) == PLAN[split]["records"] and digest(packed(rows)) == lock["cohorts"][key],
                f"Frozen cohort bytes or size differ for {key}")
        answers = []
        counts = Counter()
        for row in rows:
            answer, feature_key, traits = ORACLES[task](row["input"], int(size))
            fingerprint = digest(packed(feature_key))
            require(fingerprint not in seen[task], f"Repeated model input inside or across splits for {key}")
            seen[task].add(fingerprint)
            gold = np.asarray(row["gold"])
            require(gold.dtype.kind in "iu" and np.array_equal(answer, gold), f"Independent oracle disagrees for {key}")
            answers.append(answer)
            counts.update(traits)
        labels[key] = np.stack(answers)
        properties[key] = dict(sorted(counts.items()))
    return lock, models, labels, properties


def evaluate(run):
    results = read(run / "output/creative-results.json")
    identity, shapes = authenticate(run, results)
    lock, models, labels, properties = cases_and_weights()
    require(all(results[k] == lock[k] for k in ("cases_sha256", "weights_sha256"))
            and results["batch_size"] == PLAN["batch_size"], "Run differs from the frozen study")
    expected = {(task, size) for task in ORACLES for size in PLAN["test"]["sizes"]}
    actual = {(row["task"], row["size"]):row for row in results["conditions"]}
    require(len(results["conditions"]) == len(expected) and actual.keys() == expected, "Missing or duplicate GPU settings")
    audited = []
    with np.load(run / "output/creative-predictions.npz", allow_pickle=False) as preds:
        keys = {f"{task}_{size}" for task, size in expected}
        require(len(preds.files) == len(keys) and keys == set(shapes) == set(preds.files), "Unexpected prediction arrays")
        for task, size in sorted(expected):
            key = f"{task}_{size}"
            gold = labels[f"test/{task}/{size}"]
            pred = preds[key]
            require(pred.dtype == np.int16 and pred.shape == gold.shape and list(pred.shape) == shapes[key]
                    and pred.min() >= -1 and pred.max() < CLASSES[task], f"Invalid prediction shape or ID for {key}")
            ok = gold == pred
            scores = {"records": len(gold), "exact_records": int(ok.all(axis=1).sum()), "labels": gold.size,
                      "correct_labels": int(ok.sum()), "nonfinite": int((pred == -1).sum()),
                      "parameters": models[task]["parameters"],
                      "class_counts": dict(sorted(Counter(str(int(c)) for c in gold.flat).items()))}
            row = actual[task, size]
            require(all(v == row[k] for k,v in scores.items()), f"Regraded metrics disagree for {key}")
            times = row["seconds"]
            require(len(times) == PLAN["repeats"] and all(np.isfinite(t) and t > 0 for t in times)
                    and median(times) == row["median_seconds"], f"Invalid durations or median for {key}")
            audited.append({"task":task, "size":size, **scores, "median_seconds": row["median_seconds"],
                            "independent_case_properties": properties[f"test/{task}/{size}"]})
    return {"verified": True, "run": identity, "model_ledgers": models, "frozen_cohorts_checked": len(labels),
            "development_records_checked": sum(len(v) for k,v in labels.items() if k.startswith("dev/")),
            "test_records_checked": sum(row["records"] for row in audited),
            "labels_regraded": sum(row["labels"] for row in audited), "conditions": audited}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    audit = evaluate(args.version_folder.resolve())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {len(audit['conditions'])} GPU settings and regraded {audit['labels_regraded']:,} labels "
          f"on {audit['test_records_checked']} held-out records.")


if __name__ == "__main__":
    main()
