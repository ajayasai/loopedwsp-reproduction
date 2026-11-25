"""Audit a downloaded graph-depth GPU run independently with NumPy, without running models."""

import argparse
import ast
import base64
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUTS = ("graph-depth-results.json", "graph-depth-predictions.npz")
INPUTS = ("experiment.py", "pointer.py", "reachability.py", "corpus.py", "protocol.json", "cases-manifest.json")
NODES = 64
HOPS = {"dev": (1, 2, 4), "test": (1, 2, 3, 4, 5, 8, 9, 16, 17, 32)}
PAIRS = {"dev": 8, "test": 64}
BUDGETS = {"pointer": (3072, 6144), "reachability": (52, 100)}
STAGES = {"pointer": {"loop_shared": 6, "fixed_2": 2, "fixed_4": 4, "fixed_6": 6, "untied_full": 6},
          "reachability": {"local_tied": 63, "local_untied_1": 1, "local_untied_2": 2,
                          "local_untied_4": 4, "global_untied_4": 4, "global_untied_6": 6, "local_unrolled": 63}}
LOOP = {"pointer": "loop_shared", "reachability": "local_tied"}
UNROLLED = {"pointer": "untied_full", "reachability": "local_unrolled"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def authenticate(run, results):
    receipt = read(run / "submission.json")
    request = read(run / "kernel-metadata.json")
    remote = read(run / "remote/kernel-metadata.json")
    collector = read(run / "verification.json")
    runtime = read(run / "output/graph-depth-runtime.json")
    kernel, version = receipt["kernel"], receipt["version"]
    require(kernel == request["id"] == remote["id"] == collector["kernel"] and
            version == remote["version"] == collector["version"] and run.name == f"v{version}" and
            receipt["url"] == f"https://www.kaggle.com/code/{kernel}/versions/{version}", "Kaggle identity differs")
    require(receipt["requested_private"] is collector["server_private"] is True and
            receipt["requested_internet"] is collector["server_internet_enabled"] is False and
            all(meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_tpu"] is False
                and meta["enable_internet"] is False for meta in (request, remote)), "Private GPU settings differ")
    require(remote["machine_shape"] == receipt["requested_gpu"] == "NvidiaTeslaT4" and
            remote["source_match"] == collector["source_match"] and
            runtime["device"] == collector["device"] == results["device"] and "T4" in runtime["device"] and
            collector["pytorch"] == runtime["pytorch"] == results["pytorch"] and
            collector["cuda"] == runtime["cuda"] == results["cuda"] and runtime["available_devices"] >= 1 and
            collector["status"] == "COMPLETE" and runtime["status"] == "complete", "GPU runtime identity differs")
    script = (run / "main.py").read_bytes()
    require(sha(script) == receipt["script_sha256"] == collector["script_sha256"] and
            len(script) == receipt["script_bytes"], "Submitted source differs from its receipt")
    remote_hash = remote["downloaded_source_sha256"]
    require(collector["remote_script_sha256"] == remote_hash, "Downloaded source digest differs")
    lf = script.replace(b"\r\n", b"\n")
    if remote["source_match"] == "exact":
        require(remote_hash == sha(script), "Exact server source differs")
    else:
        require(remote["source_match"] == "line_endings" and remote_hash != sha(script) and
                remote_hash in (sha(lf), sha(lf.replace(b"\n", b"\r\n"))), "Cannot verify server source line endings")
    call = ast.parse(script).body[-1].value
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "remote_run" and
            len(call.args) == 4 and not call.keywords, "Unexpected GPU entry point")
    payload, declared, outputs, stage = (ast.literal_eval(arg) for arg in call.args)
    require(set(payload) == set(INPUTS) and outputs == list(OUTPUTS) and stage == "graph-depth",
            "Submitted filenames, stage or output declaration differs")
    hashes = {name: sha((HERE / name).read_bytes()) for name in INPUTS}
    embedded = {name: sha(zlib.decompress(base64.b64decode(blob, validate=True))) for name, blob in payload.items()}
    require(embedded == hashes == declared == receipt["input_sha256"] == runtime["input_sha256"] and
            collector["input_files_verified"] == len(INPUTS), "Frozen source, receipt or runtime input differs")
    output_hashes = {name: sha((run / "output" / name).read_bytes()) for name in OUTPUTS}
    require(output_hashes == collector["output_sha256"] == runtime["output_sha256"], "Downloaded outputs differ")
    return {"kernel": kernel, "version": version, "url": receipt["url"], "device": runtime["device"],
            "private": True, "internet_enabled": False, "submitted_script_sha256": sha(script),
            "server_script_sha256": remote_hash, "source_match": remote["source_match"],
            "input_sha256": hashes, "output_sha256": output_hashes,
            "runtime_sha256": sha((run / "output/graph-depth-runtime.json").read_bytes())}


def check_protocol(protocol):
    require(protocol["nodes"] == NODES and protocol["precision"] == "float32" and
            set(protocol["tasks"]) == set(STAGES), "Unexpected task, node count or precision")
    for split, key in (("dev", "development"), ("test", "test")):
        require(protocol[key] == {"hops": list(HOPS[split]), "pairs_per_bin": PAIRS[split]},
                f"Different {key} cohorts")
    require(protocol["timing"] == {"batch_size": 16, "warmup_batches": 1, "full_cohort_repeats": 3,
                                  "order_seed": 803472}, "Timing protocol differs")
    for task, depths in STAGES.items():
        cfg = protocol["tasks"][task]
        require(cfg["depths"] == depths and cfg["reference"] == LOOP[task] and cfg["unrolled"] == UNROLLED[task]
                and cfg["budgets"] == list(BUDGETS[task]) and len(cfg["matched"]) == len(depths) - 1
                and set(cfg["matched"]) == set(depths) - {UNROLLED[task]}, f"Model protocol differs for {task}")
        unrolled = [6*b if task == "pointer" else 4+63*(b-4) for b in BUDGETS[task]]
        require(cfg["unrolled_parameters"] == unrolled, f"Unrolled budgets differ for {task}")


def pointer_targets(tables, sources, k):
    rows = np.arange(len(tables))
    node, dest = tables[:, :, 0], tables[:, :, 1]
    require(np.all(np.sort(node, axis=1) == np.arange(NODES)) and np.all(dest < NODES) and
            np.array_equal(node[::2], node[1::2]) and np.all(sources[::2] == sources[1::2]),
            f"Malformed pointer ID table or paired source at {k}")
    ordered = np.empty_like(node)
    ordered[rows[:, None], node] = dest
    incoming = np.sort(ordered, axis=1)
    permutation = np.all(incoming[::2] == np.arange(NODES), axis=1)
    require(np.array_equal(incoming[::2], incoming[1::2]) and
            np.all(np.count_nonzero(ordered[::2] != ordered[1::2], axis=1) == 2) and
            np.array_equal(permutation, np.arange(len(tables) // 2) % 2 == 0),
            f"Pointer degrees, graph family or two-edge intervention differs at {k}")
    path = [sources]
    for _ in range(k):
        path.append(ordered[rows, path[-1]])
    path = np.stack(path, axis=1)
    require(np.all(np.diff(np.sort(path, axis=1).astype(np.int16), axis=1) > 0) and
            np.array_equal(path[::2, :-1], path[1::2, :-1]) and np.all(path[::2, -1] != path[1::2, -1]),
            f"Pointer paths overlap themselves or do not differ at the exact final hop at {k}")
    return path[:, -1].copy()


def reachability_targets(adj, sources, targets, k):
    require(np.array_equal(sources[::2], sources[1::2]) and np.array_equal(targets[::2], targets[1::2])
            and np.all(sources != targets), f"Reachability query changed within a pair at {k}")
    left, right = adj[::2], adj[1::2]
    require(np.all(np.count_nonzero(left != right, axis=(1, 2)) == 4) and
            np.array_equal(left.sum(axis=1), right.sum(axis=1)) and
            np.array_equal(left.sum(axis=2), right.sum(axis=2)) and
            np.all((adj & adj.transpose(0, 2, 1)).any(axis=(1, 2))) and np.all(adj.sum(axis=2).max(axis=1) > 1),
            f"Reachability degrees, intervention, branch or directed cycle differs at {k}")
    # Expand all rows of a Boolean frontier in parallel, independently of the
    # deque implementation used to generate the frozen corpus.
    rows = np.arange(len(adj))
    frontier = np.arange(NODES) == sources[:, None]
    visited = frontier.copy()
    distance = np.full(len(adj), -1, dtype=np.int16)
    for d in range(1, NODES):
        frontier = (adj & frontier[:, :, None]).any(axis=1) & ~visited
        if not frontier.any():
            break
        distance[frontier[rows, targets]] = d
        visited |= frontier
    require(np.all(np.sort(distance.reshape(-1, 2), axis=1) == [-1, k]),
            f"Reachability pair is not exactly one distance-{k} positive and one unreachable negative")
    return (distance >= 0).astype(np.uint8)


def verify_cases():
    manifest = read(HERE / "cases-manifest.json")
    require(manifest["vertices"] == NODES and manifest["pairs_per_bin"] == PAIRS and
            manifest["hops"] == {k: list(v) for k, v in HOPS.items()}, "Frozen case configuration differs")
    fields = {"pointer": ("table", "source", "label"), "reachability": ("adj", "source", "target", "label")}
    targets = {}
    keys = {f"{task}_{split}_{k}_{field}" for split, hops in HOPS.items() for k in hops
            for task, names in fields.items() for field in names}
    with np.load(HERE / "cases.npz", allow_pickle=False) as archive:
        require(len(archive.files) == len(keys) and set(archive.files) == set(manifest["arrays"]) == keys,
                "Case file inventory differs")
        for split, hops in HOPS.items():
            count = 2 * PAIRS[split]
            for k in hops:
                for task, names in fields.items():
                    arrays = {}
                    for name in names:
                        key = f"{task}_{split}_{k}_{name}"
                        a = archive[key]
                        shape = {"table": (count, NODES, 2), "adj": (count, NODES, NODES)}.get(name, (count,))
                        dtype = np.bool_ if name == "adj" else np.uint8
                        fingerprint = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha(a.tobytes("C"))}
                        require(a.dtype == dtype and a.shape == shape and fingerprint == manifest["arrays"][key],
                                f"Frozen case array differs: {key}")
                        if name in ("source", "target"):
                            require(np.all(a < NODES), f"Invalid endpoint at {key}")
                        arrays[name] = a
                    if task == "pointer":
                        target = pointer_targets(arrays["table"], arrays["source"], k)
                    else:
                        target = reachability_targets(arrays["adj"], arrays["source"], arrays["target"], k)
                    require(np.array_equal(target, arrays["label"]), f"Independent labels differ at {task}/{split}/{k}")
                    targets[task, split, k] = target
    return targets, len(keys)


def describe_tensor(a, kind):
    return {"kind": kind, "shape": list(a.shape), "size": a.size, "nonzero": int(np.count_nonzero(a)),
            "dtype": str(a.dtype), "sha256": sha(a.tobytes("C"))}


def expected_ledger(task, budget, name):
    """Rebuild serialized FP32 tensors and integer/Boolean buffers without importing PyTorch."""
    depth = STAGES[task][name]
    blocks = 1 if name == LOOP[task] else depth
    tensors = {}
    if task == "pointer":
        width = budget // (2 * NODES) // (1 if name == UNROLLED[task] else blocks)
        freq = np.arange(1, width//2 + 1, dtype=np.float64)
        theta = np.arange(NODES)[:, None] * (2 * math.pi) * freq / NODES + (freq + 0.5) / (NODES + 1)
        coeff = np.stack((np.cos(theta), np.sin(theta)), axis=-1).reshape(NODES, width)
        coeff = (coeff * math.sqrt(1024 * math.sqrt(width))).astype("<f4")
        stage = {key: describe_tensor(coeff, "parameter") for key in ("query", "key")}
    else:
        allocation = (budget - 4) // 4
        heads = allocation // (1 if name == UNROLLED[task] else blocks) - 1
        index = np.arange(heads * math.ceil(NODES / heads), dtype="<i8").reshape(-1, heads).T
        live = index < NODES
        group_size = live.sum(1)
        global_heads = name.startswith("global_")
        query = np.full(heads, 2048 * math.sqrt(NODES + 1)) if global_heads else np.log(4 * (group_size + 1))
        values = 2 * group_size if global_heads else np.full(heads, 2)
        coeffs = {"q": query, "k": np.ones(heads), "v": values, "o": np.ones(heads),
                  "w1": 1, "b1": -1, "w2": -1, "b2": 0}
        stage = {key: describe_tensor(np.asarray(v, dtype="<f4"), "parameter") for key, v in coeffs.items()}
        stage["live"] = describe_tensor(live, "buffer")
        stage["indices"] = describe_tensor(np.minimum(index, NODES - 1), "buffer")
        for key, value in (("readout", [-1, 1]), ("bias", [0.5, -0.5])):
            tensors[key] = describe_tensor(np.array(value, dtype="<f4"), "parameter")
    for i in range(blocks):
        tensors.update({f"blocks.{i}.{key}": entry for key, entry in stage.items()})
    return tensors


def verify_ledgers(ledgers):
    expected_keys = {f"{task}/{budget}/{name}" for task, names in STAGES.items()
                     for budget in BUDGETS[task] for name in names}
    require(set(ledgers) == expected_keys, "Coverage of named model ledgers differs")
    overview = {}
    for task, names in STAGES.items():
        for budget in BUDGETS[task]:
            for name, depth in names.items():
                key = f"{task}/{budget}/{name}"
                ledger = ledgers[key]
                tensors = expected_ledger(task, budget, name)
                require(ledger["tensors"] == tensors and
                        ledger["tensor_manifest_sha256"] == sha(json.dumps(tensors, sort_keys=True).encode()),
                        f"Named tensor values, dimensions, dtype, nonzeros or hash differ at {key}")
                params = [v for v in tensors.values() if v["kind"] == "parameter"]
                stored = sum(v["size"] for v in params)
                nonzero = sum(v["nonzero"] for v in params)
                buffers = sum(v["size"] for v in tensors.values() if v["kind"] == "buffer")
                expected = budget if name != UNROLLED[task] else (6*budget if task == "pointer" else 4+63*(budget-4))
                meta = ledger["architecture"]
                require(ledger["stored"] == meta["parameters"] == stored == expected and
                        ledger["nonzero"] == meta["nonzero_parameters"] == nonzero and ledger["buffer_scalars"] == buffers
                        and meta["name"] == name, f"Parameter, buffer or architecture totals differ at {key}")
                distinct = 1 if name == LOOP[task] else depth
                if task == "pointer":
                    require(meta["stages"] == depth and meta["distinct_blocks"] == distinct and
                            meta["address_width"] == tensors["blocks.0.key"]["shape"][1] and meta["nodes"] == NODES and
                            meta["shared"] is (name == LOOP[task]) and meta["hop_capacity"] == 2**depth-1 and
                            meta["core_parameters"] == stored and meta["readout_parameters"] == 0 and
                            meta["embedding_parameters"] == 0, f"Pointer architecture metadata differs at {key}")
                else:
                    require(meta["steps"] == depth and meta["stored_stages"] == distinct and meta["budget"] == budget and
                            meta["heads_per_stage"] == tensors["blocks.0.q"]["size"] and
                            meta["tied"] is (name == LOOP[task]) and meta["exact_budget"] is (stored == budget),
                            f"Reachability architecture metadata differs at {key}")
                overview[key] = {"stored": stored, "nonzero": nonzero, "buffer_scalars": buffers,
                                 "tensors": len(tensors), "tensor_manifest_sha256": ledger["tensor_manifest_sha256"]}
    return overview


def counts(task, prediction, truth):
    correct = prediction == truth
    values = {"correct": int(correct.sum()), "queries": len(truth), "pairs": len(truth)//2,
              "both_pair_members_correct": int(correct.reshape(-1, 2).all(1).sum()),
              "invalid": int(np.count_nonzero(prediction == -1))}
    if task == "reachability":
        for label, group in ((1, "reachable"), (0, "unreachable")):
            selected = truth == label
            values[group + "_correct"] = int((correct & selected).sum())
            values[group + "_queries"] = int(selected.sum())
    return values


def four_outcomes(left, right, prefix=""):
    return {prefix + k: int(v.sum()) for k, v in (("both_correct", left & right), ("loop_only", left & ~right),
                                                  ("comparator_only", ~left & right), ("neither", ~left & ~right))}


def compare_rows(task, budget, name, k, truth, ref_row, row, ref_pred, pred):
    left, right = ref_pred == truth, pred == truth
    pair_left, pair_right = left.reshape(-1, 2).all(1), right.reshape(-1, 2).all(1)
    result = {"task": task, "hops": k, "budget": budget, "reference": LOOP[task], "comparator": name,
              "matched_budget": ref_row["stored"] == row["stored"], "loop_parameters": ref_row["stored"],
              "comparator_parameters": row["stored"], "loop_stages": ref_row["stages"], "comparator_stages": row["stages"],
              "queries": len(truth), **four_outcomes(left, right), "loop_correct": int(left.sum()),
              "comparator_correct": int(right.sum()), "loop_accuracy": float(left.mean()),
              "comparator_accuracy": float(right.mean()), "loop_minus_comparator_accuracy": float(left.mean()-right.mean()),
              "pairs": len(pair_left), **four_outcomes(pair_left, pair_right, "pair_"),
              "loop_exact_pairs": int(pair_left.sum()), "comparator_exact_pairs": int(pair_right.sum()),
              "loop_pair_accuracy": float(pair_left.mean()), "comparator_pair_accuracy": float(pair_right.mean())}
    for label, r in (("loop", ref_row), ("comparator", row)):
        result[label + "_invalid"] = r["invalid"]
        for group in ("reachable", "unreachable"):
            for suffix in ("queries", "correct"):
                field = f"{group}_{suffix}"
                result[f"{label}_{field}"] = r.get(field, "")
        for i, value in enumerate(r["seconds"], 1):
            result[f"{label}_seconds_{i}"] = value
        result[label + "_median_seconds"] = r["median_seconds"]
    return result


def verify_scores(results, archive, targets, ledgers, split):
    require(results["split"] == split and results["precision"] == "float32", "Wrong split or inference precision")
    repeats = 3 if split == "test" else 1
    conditions = {}
    expected = {(t, b, m, k) for t, names in STAGES.items() for b in BUDGETS[t] for m in names for k in HOPS[split]}
    shapes, predictions = {}, {}
    with np.load(archive, allow_pickle=False) as data:
        for row in results["conditions"]:
            task, budget, name, k = setting = tuple(row[v] for v in ("task", "budget", "method", "hops"))
            require(setting in expected and setting not in conditions, f"Unknown or duplicate setting {setting}")
            key = f"{task}_{budget}_{name}_{k}"
            require(key == row["prediction_key"] and row["stored"] == ledgers[f"{task}/{budget}/{name}"]["stored"]
                    and row["stages"] == STAGES[task][name], f"Key, steps or count differs at {key}")
            truth = targets[task, split, k]
            pred = data[key]
            require(pred.dtype == np.int16 and pred.shape == truth.shape and
                    np.all((pred >= -1) & (pred < (NODES if task == "pointer" else 2))), f"Invalid class IDs at {key}")
            metrics = counts(task, pred, truth)
            require(all(type(row[v]) is int and row[v] == val for v, val in metrics.items()),
                    f"Independent scores differ at {key}")
            timings = row["seconds"]
            require(isinstance(timings, list) and len(timings) == repeats and
                    all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in timings) and
                    row["median_seconds"] == median(timings), f"Timing repeats or median differ at {key}")
            conditions[setting], predictions[setting], shapes[key] = row, pred, list(pred.shape)
        require(set(conditions) == expected and set(shapes) == set(data.files) and len(shapes) == len(data.files),
                "Incomplete or extra result rows or prediction arrays")
    paired = []
    for task, names in STAGES.items():
        for budget in BUDGETS[task]:
            for k in HOPS[split]:
                ref = task, budget, LOOP[task], k
                for name in names:
                    if name == LOOP[task]:
                        continue
                    comp = task, budget, name, k
                    paired.append(compare_rows(task, budget, name, k, targets[task, split, k],
                                               conditions[ref], conditions[comp], predictions[ref], predictions[comp]))
    return paired, shapes


def evaluate(run):
    results = read(run / "output" / OUTPUTS[0])
    identity = authenticate(run, results)
    check_protocol(read(HERE / "protocol.json"))
    targets, arrays = verify_cases()
    ledgers = verify_ledgers(results["ledgers"])
    paired, shapes = verify_scores(results, run / "output" / OUTPUTS[1], targets, ledgers, "test")
    require(shapes == read(run / "verification.json")["prediction_arrays"], "Collector's prediction shapes differ")
    summary = {"verified": True, "run": identity, "case_arrays_verified": arrays, "test_hops": list(HOPS["test"]),
               "test_queries_per_task": 2*PAIRS["test"]*len(HOPS["test"]),
               "test_pairs_per_task": PAIRS["test"]*len(HOPS["test"]), "conditions_verified": len(shapes),
               "timings_per_condition": 3, "paired_comparisons": len(paired), "model_ledgers": ledgers,
               "method": "Independent successor traversal and directed Boolean-frontier distances. Input pair edits preserve degrees. Tensor values and named buffers independently rebuilt in NumPy. A nonfinite output has saved class -1, which is incorrect. Kaggle saved one class vector per condition after checking repeat agreement."}
    return summary, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("audit_json", nargs="?", type=Path, help="Write the audit summary here")
    parser.add_argument("comparisons_csv", nargs="?", type=Path, help="Write paired condition comparisons here")
    args = parser.parse_args()
    summary, paired = evaluate(args.version_folder.resolve())
    if args.comparisons_csv or args.audit_json:
        csv_file = args.comparisons_csv or args.audit_json.with_name("comparisons.csv")
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with csv_file.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(paired[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(paired)
        summary["comparisons_csv_sha256"] = sha(csv_file.read_bytes())
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {summary['conditions_verified']} GPU settings, {len(paired)} paired comparisons, "
          f"and {summary['case_arrays_verified']} frozen case arrays.")


if __name__ == "__main__":
    main()
