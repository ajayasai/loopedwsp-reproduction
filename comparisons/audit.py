"""Authenticate a downloaded GPU comparison and independently regrade its class IDs with NumPy."""

import argparse
import ast
import base64
from collections import Counter
import inspect
from itertools import product
import json
import math
from pathlib import Path
import statistics
import sys
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
APPS = ROOT / "applications"
sys.path.insert(0, str(ROOT))
from applications.audit import read, reference, require, sha, tensor_digest, verify_model, verify_package


TASKS = ("inventory", "scheduling", "telemetry")
SIZES = (8, 16, 32, 64, 128)
PRECISIONS = ("float64", "float32", "float16")
ORIGINALS = ("original_1", "original_4", "original_full")
METHODS = (*ORIGINALS, "full_feedforward", "local_w2", "local_w8", "local_w32")
OUTPUTS = ["comparison-results.json", "comparison-predictions.npz"]
LEGACY_RUN = ROOT / "kaggle/runs/practical-applications/v1"
LEGACY_PREDICTIONS_SHA256 = "f8232382670e4f14cff24fe3690720cc0daec222d592ee7e54081aa0bea187d4"


def input_files():
    paths = {name: HERE / name for name in
             ("experiment.py", "baselines.py", "recurrent.py", "challenge.npz", "protocol.json")}
    names = ["cases.npz", "corpus-lock.json"]
    names.extend(f"models/{task}/{file}" for task in TASKS for file in ("network.json", "weights.npz", "freeze.json"))
    paths.update({f"applications/{name}": APPS / name for name in names})
    return {name: path.read_bytes() for name, path in paths.items()}


def verify_source(run, contents, hashes, expected):
    # Reconstruct the script if a checkout contains just the archived server digest.
    # Extract and hash literal arguments from every available version without executing any payload.
    sources = [p for p in (run / "main.py", run / "remote/main.py") if p.is_file()]
    if sources:
        scripts = [(p.relative_to(run).as_posix(), p.read_bytes()) for p in sources]
    else:
        from puzzles.submit_puzzles import remote_run
        bundle = {name: base64.b64encode(zlib.compress(data)).decode("ascii") for name, data in contents.items()}
        body = inspect.getsource(remote_run) + f"\nremote_run({bundle!r}, {hashes!r}, {OUTPUTS!r}, 'comparison')\n"
        scripts = [("reconstructed from frozen files", body.encode("ascii"))]
    for label, script in scripts:
        require(sha(script) == expected, f"Script digest differs for {label}")
        call = ast.parse(script).body[-1].value
        require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "remote_run"
                and len(call.args) == 4 and not call.keywords, "Unexpected entry point")
        bundle, embedded, outputs, stage = (ast.literal_eval(arg) for arg in call.args)
        require(stage == "comparison" and outputs == OUTPUTS and embedded == hashes and set(bundle) == set(hashes),
                "Unexpected source payload or outputs")
        for name, encoded in bundle.items():
            require(zlib.decompress(base64.b64decode(encoded, validate=True)) == contents[name],
                    f"Source embeds different bytes for {name}")
    return [label for label, _ in scripts]


def authenticate(run, results):
    receipt = read(run / "submission.json")
    request = read(run / "kernel-metadata.json")
    remote = read(run / "remote/kernel-metadata.json")
    runtime = read(run / "output/comparison-runtime.json")
    collected = read(run / "verification.json")
    kernel, version = receipt["kernel"], receipt["version"]
    require(kernel == request["id"] == remote["id"] == collected["kernel"]
            and run.name == f"v{version}" and version == remote["version"] == collected["version"]
            and receipt["url"] == f"https://www.kaggle.com/code/{kernel}/versions/{version}",
            "Version or kernel identity differs")
    require(receipt["requested_private"] is True and receipt["requested_internet"] is False
            and collected["server_private"] is True and collected["server_internet_enabled"] is False
            and all(m["is_private"] is True and m["enable_gpu"] is True and m["enable_internet"] is False
                    and m["enable_tpu"] is False for m in (request, remote)), "GPU or privacy settings differ")
    require(receipt["requested_gpu"] == remote["machine_shape"] == "NvidiaTeslaT4"
            and collected["device"] == results["gpu"] == runtime["device"] and "T4" in runtime["device"]
            and runtime["available_devices"] >= 1 and runtime["status"] == "complete"
            and collected["status"] == "COMPLETE" and collected["cuda"] == runtime["cuda"] == results["cuda"]
            and collected["pytorch"] == runtime["pytorch"] == results["torch"], "Runtime identity differs")
    files = input_files()
    hashes = {name: sha(data) for name, data in files.items()}
    require(hashes == runtime["input_sha256"] == receipt["input_sha256"] and
            len(files) == collected["input_files_verified"] and hashes["protocol.json"] == results["protocol_sha256"],
            "Input hashes or protocol identity differ")
    script_hash = receipt["script_sha256"]
    require(script_hash == remote["downloaded_source_sha256"] == collected["script_sha256"], "Script hashes differ")
    source = verify_source(run, files, hashes, script_hash)
    outputs = {name: sha((run / "output" / name).read_bytes()) for name in OUTPUTS}
    require(outputs == collected["output_sha256"] == runtime["output_sha256"], "GPU output hashes differ")
    return {"kernel": kernel, "version": version, "device": runtime["device"], "source_verified": source,
            "script_sha256": script_hash, "protocol_sha256": hashes["protocol.json"], "inputs_verified": len(files),
            "output_sha256": outputs, "runtime_sha256": sha((run / "output/comparison-runtime.json").read_bytes())}


def load_targets():
    locks = read(APPS / "corpus-lock.json")
    targets, counts = {}, {}
    with np.load(APPS / "cases.npz", allow_pickle=False) as cases, np.load(
            HERE / "challenge.npz", allow_pickle=False) as challenge:
        keys = {f"{t}_{n}_{f}" for t, n, f in product(TASKS, SIZES, ("x", "y"))}
        require(len(challenge.files) == len(keys) and set(challenge.files) == keys, "Challenge archive keys differ")
        for task in TASKS:
            freeze = read(APPS / "models" / task / "freeze.json")
            spec, vocab = verify_model(task, freeze)
            rule = "log2_plus_one" if task == "inventory" else "tokens"
            require(spec["round_rule"] == rule, f"Unexpected frozen depth rule for {task}")
            counts[task] = (freeze["parameters"], freeze["nonzero_parameters"])
            for n in SIZES:
                orig, key = f"{task}_test_{n}", f"{task}_{n}"
                x, y = cases[orig + "_x"], cases[orig + "_y"]
                cx, cy = challenge[key + "_x"], challenge[key + "_y"]
                lock = locks[task][orig]
                require(lock["count"] == 64 and tensor_digest(x) == lock["inputs"]
                        and tensor_digest(y) == lock["labels"], f"Corpus lock differs at {orig}")
                for rows, labels, count in ((x, y, 64), (cx, cy, 32)):
                    require(rows.dtype == np.int64 and rows.shape == (count, n, len(vocab)) and
                            labels.dtype == np.uint8 and labels.shape == (count, n) and
                            all(np.all((rows[..., i] >= 0) & (rows[..., i] < lim)) for i, lim in enumerate(vocab)),
                            f"Malformed inputs for {key}")
                    require(np.array_equal(reference(task, rows), labels), f"Independent labels differ at {key}")
                require(np.array_equal(cx[::2, 1:], cx[1::2, 1:]) and np.all(cy[::2, -1] != cy[1::2, -1]),
                        f"Challenge pair suffix or final label differs at {key}")
                records = list(x) + list(cx)
                if n in (8, 16):
                    records.extend(cases[f"{task}_dev_{n}_x"])
                require(len({r.tobytes() for r in records}) == len(records), f"Duplicate record at {key}")
                targets[task, n] = np.concatenate((y, cy))
    return targets, counts


def expected_counts(task, method, frozen):
    if method in ORIGINALS:
        return frozen[task]
    taps = int(method.removeprefix("local_w")) if method.startswith("local_w") else 0
    # Inventory: two zero-point offsets, one margin. Telemetry: two thresholds,
    # one bonus plus either one global recency gain or one gain per local lag.
    # Scheduling: 12 hidden entries (8 nonzero), 3 output coefficients, one margin.
    if task == "scheduling":
        return 16 + taps, 12 + taps
    count = 4 if task == "telemetry" and not taps else 3 + taps
    return count, count


def grade(ids, y, challenge):
    ok = ids == y
    positives = (ids == 1) & (y == 0)
    negatives = (ids == 0) & (y == 1)
    result = {"n_records": len(ids), "exact": int(ok.all(axis=1).sum()),
              "n_labels": y.size, "correct": int(ok.sum()), "invalid": int((ids == 2).sum()),
              "false_positives": int(positives.sum()), "false_negatives": int(negatives.sum()),
              "final_correct": int(ok[:, -1].sum())}
    if challenge:
        result["pairs_both_final_correct"] = int(ok[:, -1].reshape(-1, 2).all(axis=1).sum())
    return result


def disagreement(old, new, y):
    diff = old != new
    before, now = old == y, new == y
    old_exact, now_exact = before.all(1), now.all(1)
    return {"different_flags": int(diff.sum()), "different_records": int(diff.any(1).sum()),
            "old_correct_flags_lost": int((before & ~now).sum()), "new_correct_flags_gained": int((~before & now).sum()),
            "old_exact_records_lost": int((old_exact & ~now_exact).sum()),
            "new_exact_records_gained": int((~old_exact & now_exact).sum())}


def legacy_predictions():
    identity = verify_package(LEGACY_RUN)
    digest = identity["output_sha256"]["predictions.npz"]
    require(digest == LEGACY_PREDICTIONS_SHA256, "Legacy prediction archive differs from the published archive")
    audited = read(ROOT / "results/applications/verification.json")
    require(audited["verified"] is True and audited["run"] == identity, "Legacy audit run identity differs")
    report = read(LEGACY_RUN / "output/application-results.json")
    require(report["dtype"] == "float64" and report["gpu"] == identity["device"], "Legacy precision or GPU differs")
    predictions = {}
    with np.load(LEGACY_RUN / "output/predictions.npz", allow_pickle=False) as data:
        require(len(data.files) == len(TASKS) * len(SIZES), "Unexpected number of legacy arrays")
        for task in TASKS:
            original = {r["size"]: r for r in report["tasks"][task]["rows"]}
            verified = {r["size"]: r for r in audited["tasks"][task]["sizes"]}
            for n in SIZES:
                pred = data[f"{task}_test_{n}"]
                require(pred.dtype == np.uint8 and pred.shape == (64, n) and np.isin(pred, (0, 1)).all()
                        and tensor_digest(pred) == original[n]["predictions_sha256"] == verified[n]["predictions_sha256"],
                        f"Legacy class IDs or tensor digest differ for {task}/{n}")
                predictions[task, n] = pred
    return digest, predictions


def evaluate(run):
    report = read(run / "output/comparison-results.json")
    identity = authenticate(run, report)
    protocol = read(HERE / "protocol.json")
    require(all(protocol[key] == list(values) for key, values in
                (("tasks", TASKS), ("sizes", SIZES), ("precisions", PRECISIONS), ("methods", METHODS))) and
            protocol["timed_repeats"] == 3 and protocol["batch_size"] == 16, "Unexpected study grid or timing policy")
    for path, name in ((APPS / "cases.npz", "cases"), (APPS / "corpus-lock.json", "corpus_lock"),
                       (HERE / "challenge.npz", "challenge")):
        require(sha(path.read_bytes()) == protocol[name + "_sha256"], f"Protocol hash differs for {name}")
    targets, frozen_counts = load_targets()
    legacy_hash, old = legacy_predictions()
    expected = set(product(TASKS, SIZES, PRECISIONS, METHODS))
    seen, shapes = set(), {}
    totals = {k: Counter() for k in ("published", "challenge", "raw_published", "raw_challenge")}
    screen_effects = {cohort: Counter() for cohort in ("published", "challenge")}
    legacy = []
    with np.load(run / "output/comparison-predictions.npz", allow_pickle=False) as data:
        for row in report["rows"]:
            task, n, dtype, method = setting = tuple(row[k] for k in ("task", "size", "precision", "method"))
            require(setting in expected and setting not in seen, f"Unexpected or repeated configuration {setting}")
            seen.add(setting)
            counts = expected_counts(task, method, frozen_counts)
            steps = 4 if method == "original_4" else 1
            if method == "original_full":
                steps = (n - 1).bit_length() + 1 if task == "inventory" else n
            require(all(type(row[k]) is int for k in ("size", "parameters", "nonzero_parameters", "steps"))
                    and (row["parameters"], row["nonzero_parameters"]) == counts and row["steps"] == steps,
                    f"Parameter count or depth differs at {setting}")
            times = row["seconds"]
            require(isinstance(times, list) and len(times) == 3 and
                    all(type(t) in (float, int) and math.isfinite(t) and t > 0 for t in times)
                    and row["median_seconds"] == statistics.median(times), f"Invalid timings at {setting}")
            strict_key = f"{task}_{n}_{dtype}_{method}"
            raw_key = strict_key + "_raw" if method in ORIGINALS else strict_key
            require(row["prediction_key"] == strict_key and row["raw_prediction_key"] == raw_key,
                    f"Prediction keys differ at {setting}")
            y = targets[task, n]
            strict = data[strict_key]
            raw = data[raw_key] if raw_key != strict_key else strict
            for key, ids in ((strict_key, strict), (raw_key, raw)):
                require(ids.shape == y.shape and ids.dtype == np.uint8 and np.isin(ids, (0, 1, 2)).all(),
                        f"Invalid saved classes at {key}")
                shapes[key] = list(ids.shape)
            require(np.all((strict == 2) | (strict == raw)), f"Screen changes a valid raw classification at {setting}")
            for prefix, ids in (("", strict), ("raw_", raw)):
                for cohort, span in (("published", slice(0, 64)), ("challenge", slice(64, None))):
                    scores = grade(ids[span], y[span], cohort == "challenge")
                    key = prefix + cohort
                    require(row[key] == scores and all(type(v) is int for v in row[key].values()),
                            f"Reported {key} scores disagree at {setting}")
                    totals[key].update(scores)
            for cohort, span in (("published", slice(0, 64)), ("challenge", slice(64, None))):
                screen_effects[cohort].update(disagreement(raw[span], strict[span], y[span]))
            if method == "original_full" and dtype == "float64":
                previous = old[task, n]
                legacy.append({"task": task, "size": n, "old_sha256": tensor_digest(previous),
                               "strict_sha256": tensor_digest(strict[:64]), "raw_sha256": tensor_digest(raw[:64]),
                               "strict": disagreement(previous, strict[:64], y[:64]),
                               "raw": disagreement(previous, raw[:64], y[:64])})
        require(seen == expected and len(data.files) == len(shapes) and set(data.files) == set(shapes),
                "Incomplete configuration grid or unexpected prediction tensors")
    require(read(run / "verification.json")["prediction_arrays"] == shapes, "Collector's prediction shapes differ")
    for count in totals.values():
        count["incorrect_records"] = count["n_records"] - count["exact"]
        count["incorrect_flags"] = count["n_labels"] - count["correct"]
        require(count["incorrect_flags"] == count["invalid"] + count["false_positives"] + count["false_negatives"],
                "Disjoint flag loss categories do not sum")
    legacy_totals = {channel: Counter() for channel in ("strict", "raw")}
    for row in legacy:
        for channel, summed in legacy_totals.items():
            summed.update(row[channel])
    return {"verified": True, "run": identity, "settings_verified": len(seen), "prediction_arrays": len(shapes),
            "timings_per_setting": 3, "published_records_verified": 64 * len(targets),
            "challenge_pairs_verified": 16 * len(targets), "totals_over_settings": totals,
            "grading": "Finite scores > 0 give class 1 and zero ties give 0. Saved class 2 is invalid. Each incorrect flag counts once. A record passes only if every flag is correct. The archives contain class IDs. Timings are checked from the three reported values.",
            "raw_to_strict_over_settings": screen_effects,
            "legacy_fp64": {"prediction_archive_sha256": legacy_hash, "cohort": "published",
                            "method": "original_full", "totals": legacy_totals,
                            "by_size": sorted(legacy, key=lambda r: (r["task"], r["size"]))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = evaluate(args.version_folder.resolve())
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {result['settings_verified']} settings, {result['prediction_arrays']} prediction arrays, "
          f"and {result['challenge_pairs_verified']} challenge pairs.")


if __name__ == "__main__":
    main()
