"""Independently verify, solve and regrade the archived workshop GPU study."""
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

INPUTS = ("experiment.py", "protocol.json", "mobile.py", "cels.py", "roll.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("workshop-results.json", "workshop-predictions.npz")
CLASSES = {"mobile": 17, "cels": 16, "roll": 145}
SPLITS = {"dev": {"sizes": [4, 8], "per_size": 8},
          "test": {"sizes": [4, 8, 16, 32], "per_size": 64}}


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def solve_mobile(record, size):
    """Try all 16 masses and total the raw leaf weights on each side of each rod."""
    masses, arms = record["masses"], record["arms"]
    require(set(record) == {"masses", "arms"} and len(masses) == size and masses.count(None) == 1
            and all(x is None or (type(x) is int and 1 <= x <= 32) for x in masses)
            and len(arms) == size-1 and all(len(arm) == 2 and all(type(x) is int and 1 <= x <= 512
                                                                 for x in arm) for arm in arms),
            "Invalid mobile layout or mass")
    missing = masses.index(None)
    trials = np.tile([x or 0 for x in masses], (16, 1))
    trials[:, missing] = np.arange(1, 17)
    valid = np.ones(16, dtype=bool)
    for idx, (left, right) in enumerate(arms):
        # Nodes at each level cover equally sized contiguous ranges of leaves.
        level = (idx + 1).bit_length() - 1
        span = size >> level
        start = (idx + 1 - (1 << level)) * span
        mid = start + span // 2
        balances = (left * trials[:, start:mid].sum(axis=1) ==
                    right * trials[:, mid:start+span].sum(axis=1))
        valid &= balances
        if idx == 0:
            root_accepts = bool(balances.any())
        if span == 2 and start <= missing < start + span:
            parent_accepts = bool(balances.any())
    candidates = np.flatnonzero(valid) + 1
    require(len(candidates) <= 1, "Multiple feasible counterweights")
    answer = int(candidates[0]) if candidates.size else 0
    flags = {"feasible": bool(answer), "impossible": not answer,
             "root_only_false_positive": root_accepts and not answer,
             "parent_only_false_positive": parent_accepts and not answer}
    return np.array([answer]), flags


def solve_cels(record, size):
    """Follow one ray through the stack for each output pixel, in integer RGB4."""
    bg, layers = record["background"], record["cels"]
    def rgb(values):
        return len(values) == 3 and all(type(x) is int and 0 <= x <= 15 for x in values)
    require(set(record) == {"background", "cels"} and rgb(bg) and len(layers) == size, "Invalid cel record")
    for layer in layers:
        x0, y0, x1, y1 = layer["rect"]
        require(set(layer) == {"rgb", "rect", "opacity"} and all(type(x) is int for x in layer["rect"])
                and 0 <= x0 < x1 <= 8 and 0 <= y0 < y1 <= 8 and rgb(layer["rgb"])
                and type(layer["opacity"]) is int and layer["opacity"] in (0, 1, 2), "Invalid cel")
    def raster(order):
        channels = []
        for y in range(8):
            for x in range(8):
                color = bg
                for layer in order:
                    x0, y0, x1, y1 = layer["rect"]
                    if x0 <= x < x1 and y0 <= y < y1:
                        if layer["opacity"] == 2:
                            color = layer["rgb"]
                        elif layer["opacity"] == 1:
                            color = [(back + front + 1) // 2 for back, front in zip(color, layer["rgb"])]
                channels.extend(color)
        return np.array(channels)
    answer = raster(layers)
    sensitive = not np.array_equal(answer, raster(layers[::-1]))
    return answer, {"order_sensitive": sensitive, "order_insensitive": not sensitive}


def solve_roll(record, size):
    """Locate one flipped track by XORing the indices of present holes."""
    frames = record["frames"]
    require(set(record) == {"frames"} and len(frames) == size, "Invalid roll")
    answers = []
    for row in frames:
        require(isinstance(row, str) and len(row) == 8 and set(row) <= {"0", "1"}, "Invalid frame")
        holes = [i for i, bit in enumerate(row, start=1) if bit == "1"]
        syndrome = 0
        for track in holes:
            syndrome ^= track % 8
        if len(holes) % 2 == 0 and syndrome:
            answers.append(144)
            continue
        bad_track = (syndrome or 8) if len(holes) % 2 else 0
        event = sum((int(row[pos-1]) ^ int(bad_track == pos)) << j
                    for j, pos in enumerate((3, 5, 6, 7)))
        answers.append(16*bad_track + event)
    answer = np.array(answers)
    return answer, {"clean": answer < 16, "corrected": (16 <= answer) & (answer < 144),
                    "double_error": answer == 144}


ORACLES = {"mobile": solve_mobile, "cels": solve_cels, "roll": solve_roll}


def model_ledger(lock):
    data_tracks = (3, 5, 6, 7)
    readout = []
    for row in range(144):
        track, event = divmod(row, 16)
        syndrome = track % 8
        observed = event ^ (1 << data_tracks.index(track)) if track in data_tracks else event
        bits = [(syndrome >> i) & 1 for i in range(3)] + [int(track > 0)] + [(observed >> i) & 1 for i in range(4)]
        readout.append([1 if bit else -1 for bit in bits] + [8-sum(bits)])
    readout.append([0]*8 + [7.5])
    expected = {
        "mobile": {"block.queries": [-80., 80.], "block.torque_sign": [1., -1.],
                   "block.merge": [1., 1.], "impossible_score": -0.5},
        "cels": {"block.address": [512., -256., 256.], "block.bias": -np.arange(30, dtype=np.float32),
                 "block.slope": [1., -1.]*15, "block.gate_margin": 16.},
        "roll": {"membership": [[((t % 8) >> i) & 1 for i in range(3)] + [1] for t in range(1, 9)],
                 "block.position_query": [64., -32.], "block.parity_response": [2., -4., -0.5], "readout": readout},
    }
    models = {}
    require(set(lock["models"]) == set(expected), "Unexpected model names")
    with np.load(HERE / "weights.npz", allow_pickle=False) as archive:
        names = {f"{task}/{name}" for task, entries in expected.items() for name in entries}
        require(len(archive.files) == len(names) and set(archive.files) == names, "Weight archive keys differ")
        for task, tensors in expected.items():
            manifest = lock["models"][task]
            require(set(manifest["tensors"]) == set(tensors), "Weight manifest keys differ")
            stored = nonzero = 0
            for name, expected_value in tensors.items():
                a = archive[f"{task}/{name}"]
                ref = np.asarray(expected_value, dtype=np.float32)
                info = {"shape": list(a.shape), "dtype": str(a.dtype), "parameter": True,
                        "nonzero": int(np.count_nonzero(a)), "sha256": digest(a.tobytes())}
                require(info == manifest["tensors"][name] and a.dtype == ref.dtype and a.shape == ref.shape
                        and a.tobytes() == ref.tobytes(), f"Different tensor {task}/{name}")
                stored += a.size
                nonzero += info["nonzero"]
            require(stored == manifest["parameters"], f"Unexpected parameter count for {task}")
            models[task] = {"parameters": stored, "nonzero": nonzero, "tensors": len(tensors)}
    return models


def frozen():
    protocol = read(HERE / "protocol.json")
    lock = read(HERE / "freeze.json")
    blob = (HERE / "cases.json.gz").read_bytes()
    require(lock["protocol_sha256"] == digest((HERE / "protocol.json").read_bytes())
            and lock["weights_sha256"] == digest((HERE / "weights.npz").read_bytes())
            and lock["cases_sha256"] == digest(blob), "Frozen artifact hash differs")
    require(protocol["splits"] == SPLITS and protocol["seed"] == 2026112502 and protocol["batch"] == 8
            and protocol["repeats"] == 3 and {k: v["classes"] for k, v in protocol["tasks"].items()} == CLASSES,
            "Unexpected experiment protocol")
    models = model_ledger(lock)
    cases = json.loads(gzip.decompress(blob))
    keys = {f"{split}/{task}/{size}" for split, cfg in SPLITS.items() for task in ORACLES for size in cfg["sizes"]}
    require(set(cases) == set(lock["cohorts"]) == keys, "Unexpected frozen cohorts")
    labels, strata = {}, {}
    seen = {t: set() for t in ORACLES}
    for key, records in cases.items():
        split, task, size = key.split("/")
        require(len(records) == SPLITS[split]["per_size"] and digest(packed(records)) == lock["cohorts"][key],
                f"Unexpected cohort size or digest for {key}")
        answers, traits = [], []
        for row in records:
            # The oracles constrain every field and type. Each accepted raw
            # record maps one-to-one to the token features at its input size.
            answer, flags = ORACLES[task](row["input"], int(size))
            fingerprint = digest(packed(row["input"]))
            require(fingerprint not in seen[task], f"Duplicate input inside or across splits in {key}")
            seen[task].add(fingerprint)
            gold = np.asarray(row["gold"])
            require(gold.dtype.kind in "iu" and np.array_equal(gold, answer), f"Oracle disagrees in {key}")
            answers.append(answer)
            traits.append(flags)
        labels[key] = np.stack(answers)
        strata[key] = {name: np.stack([r[name] for r in traits]) for name in traits[0]}
    return lock, models, labels, strata


def evaluate(run):
    results = read(run / "output/workshop-results.json")
    identity, shapes = authenticate_run(run, results, folder=HERE, inputs=INPUTS, outputs=OUTPUTS, stage_name="workshop")
    lock, models, labels, strata = frozen()
    require(all(results[k] == lock[k] for k in ("cases_sha256", "weights_sha256")), "Unexpected run inputs")
    pairs = [(t, n) for t in ORACLES for n in SPLITS["test"]["sizes"]]
    rows = {(r["task"], r["size"]): r for r in results["conditions"]}
    require(len(results["conditions"]) == len(pairs) and set(rows) == set(pairs), "Unexpected GPU conditions")
    checked, totals = [], {t: Counter() for t in ORACLES}
    with np.load(run / "output/workshop-predictions.npz", allow_pickle=False) as archive:
        names = {f"{task}_{size}" for task, size in pairs}
        require(len(archive.files) == len(names) and names == set(shapes) == set(archive.files),
                "Unexpected downloaded prediction arrays")
        for task, size in pairs:
            key, cohort = f"{task}_{size}", f"test/{task}/{size}"
            gold, pred = labels[cohort], archive[key]
            require(pred.dtype == np.int16 and pred.shape == gold.shape and list(pred.shape) == shapes[key]
                    and pred.min() >= -1 and pred.max() < CLASSES[task], f"Invalid predictions for {key}")
            ok = pred == gold
            exact = ok.all(1)
            scores = {"records": len(pred), "exact_records": int(exact.sum()), "parameters": models[task]["parameters"],
                      "labels": gold.size, "correct_labels": int(ok.sum()), "nonfinite": int((pred < 0).sum())}
            row = rows[task, size]
            require(all(v == row[k] for k, v in scores.items()), f"GPU counts disagree for {key}")
            times = row["seconds"]
            require(len(times) == 3 and all(np.isfinite(t) and t > 0 for t in times)
                    and median(times) == row["median_seconds"], f"Unexpected times for {key}")
            categories = {}
            for name, flag in strata[cohort].items():
                correct = ok if task == "roll" else exact
                require(flag.shape == correct.shape, f"Incompatible category {name} in {key}")
                categories[name] = int(flag.sum())
                categories[name + "_correct"] = int((flag & correct).sum())
            checked.append({"task": task, "size": size, **scores, "seconds": times, "median_seconds": median(times),
                            "categories": categories})
            totals[task].update(categories)
    return {"verified": True, "run": identity, "model_ledgers": models, "frozen_cohorts_checked": len(labels),
            "development_records_checked": sum(len(v) for k, v in labels.items() if k.startswith("dev/")),
            "test_records_checked": sum(r["records"] for r in checked),
            "labels_regraded": sum(r["labels"] for r in checked),
            "test_categories": {k: dict(v) for k, v in totals.items()}, "conditions": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    audit = evaluate(args.version_folder.resolve())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {len(audit['conditions'])} GPU settings and regraded {audit['labels_regraded']:,} "
          f"labels on {audit['test_records_checked']} held-out records.")
