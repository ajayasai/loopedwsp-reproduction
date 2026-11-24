"""Authenticate downloaded Kaggle artifacts and regrade every application prediction."""

import argparse
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np


HERE = Path(__file__).resolve().parent
TASKS = ("inventory", "scheduling", "telemetry")
SIZES = (8, 16, 32, 64, 128)
SCORE_KEYS = ("records", "exact", "decisions", "correct", "positive_labels",
              "false_positives", "false_negatives")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def tensor_digest(arr):
    arr = np.ascontiguousarray(arr)
    return sha(json.dumps([arr.shape, arr.dtype.str]).encode() + arr.tobytes())


def verify_package(run):
    output = run / "output"
    receipt = read(run / "submission.json")
    meta = read(run / "remote/kernel-metadata.json")
    require(run.name == f"v{receipt['version']}" and meta["id"] == receipt["kernel"],
            "Kaggle version or kernel ID differs from the receipt")
    require(receipt["requested_private"] is True and receipt["requested_internet"] is False
            and meta["is_private"] is True and meta["enable_gpu"] is True
            and meta["enable_internet"] is False and not meta.get("enable_tpu")
            and meta["machine_shape"] == receipt["requested_gpu"] == "NvidiaTeslaT4",
            "Kaggle hardware or privacy settings differ from the submission request")
    filename = meta["code_file"]
    require(Path(filename).name == filename, "Unexpected server script path")
    submitted = (run / "main.py").read_bytes()
    pulled = (run / "remote" / filename).read_bytes()
    require(sha(submitted) == receipt["script_sha256"] and
            pulled.replace(b"\r\n", b"\n") == submitted.replace(b"\r\n", b"\n"),
            "Server script, submission archive, or script receipt differs")
    # Parse literals from the submitted script without executing it or extracting paths.
    call = ast.parse(submitted).body[-1].value
    require(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and
            call.func.id == "remote_run" and len(call.args) == 4, "Unexpected script entry point")
    payload, declared, outputs, name = (ast.literal_eval(arg) for arg in call.args)
    require(name == "application" and set(outputs) == {"application-results.json", "predictions.npz"},
            "Unexpected Kaggle output contract")
    locations = {k: HERE / k for k in ("experiment.py", "corpus-lock.json", "cases.npz")}
    locations["frame.py"] = HERE.parent / "lengths/frame.py"
    for task in TASKS:
        locations[task + ".py"] = HERE / (task + ".py")
        for item in ("network.json", "weights.npz", "freeze.json"):
            k = f"models/{task}/{item}"
            locations[k] = HERE / k
    require(set(payload) == set(locations), "Unexpected bundled input files")
    bundled = {k: sha(zlib.decompress(base64.b64decode(encoded, validate=True)))
               for k, encoded in payload.items()}
    local = {k: sha(path.read_bytes()) for k, path in locations.items()}
    runtime = read(output / "application-runtime.json")
    require(bundled == local == declared == receipt["input_sha256"] == runtime["input_sha256"],
            "Local files, package contents, receipt or Kaggle runtime input checksums differ")
    hashes = {k: sha((output / k).read_bytes()) for k in outputs}
    require(runtime["status"] == "complete" and runtime["output_sha256"] == hashes,
            "Incomplete execution or output checksum mismatch")
    require("T4" in runtime["device"], "Runtime did not identify a Tesla T4")
    return {"kernel": meta["id"], "version": receipt["version"], "url": receipt["url"],
            "private": meta["is_private"], "internet_enabled": meta["enable_internet"],
            "machine_shape": meta["machine_shape"], "device": runtime["device"],
            "submitted_script_sha256": sha(submitted), "downloaded_script_sha256": sha(pulled),
            "runtime_sha256": sha((output/"application-runtime.json").read_bytes()),
            "input_sha256": bundled, "output_sha256": hashes}


def verify_model(task, model):
    folder = HERE / "models" / task
    spec, freeze = read(folder/"network.json"), read(folder/"freeze.json")
    require(freeze == model, f"{task} result identifies different frozen weights")
    h = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    total = nonzero = 0
    with np.load(folder/"weights.npz", allow_pickle=False) as weights:
        width, classes = weights["readout"].shape
        require(classes == 2, f"Incorrect number of {task} output classes")
        vocab = [weights[f"e{i}"].shape[0] for i in range(len(spec["fields"]))]
        for name in sorted(weights.files):
            arr = np.ascontiguousarray(weights[name])
            require(arr.dtype == np.float64 and np.isfinite(arr).all(), f"Invalid FP64 weights {task}/{name}")
            h.update(json.dumps([name, arr.shape, arr.dtype.str]).encode())
            h.update(arr.tobytes())
            total += arr.size
            nonzero += int(np.count_nonzero(arr))
    require(h.hexdigest() == freeze["model_sha256"] and width == freeze["width"] and
            total == freeze["parameters"] and nonzero == freeze["nonzero_parameters"],
            f"Incorrect model digest or tensor count for {task}")
    return spec, vocab


def reference(task, x):
    """Grade a batch from raw fields. No benchmark oracle or transformer is imported."""
    _, size, fields = x.shape
    pos = np.arange(size)
    if task == "inventory":
        require(fields == 3 and np.all(x[:,:,0] == pos) and np.all((0 <= x[:,:,1]) & (x[:,:,1] <= 32))
                and np.all((1 <= x[:,0,2]) & (x[:,0,2] <= 49)) and np.all(x[:,1:,2] == 0),
                "Malformed inventory inputs")
        opening = x[:,0,2] - 25
        balances = opening[:,None] + np.cumsum(x[:,:,1] - 16, axis=1)
        return (balances < 0).astype(np.uint8)
    if task == "telemetry":
        readings = x[:,:,0]
        require(fields == 2 and np.all(x[:,:,1] == pos) and
                np.all((0 <= readings) & (readings <= 100)), "Malformed telemetry inputs")
        on = np.zeros(len(x), np.bool_)
        labels = np.empty(readings.shape, np.uint8)
        for t in range(size):
            on = (readings[:,t] >= 70) | (on & (readings[:,t] > 30))
            labels[:,t] = on
        return labels
    require(task == "scheduling" and fields == 3 and np.all(x[:,:,2] == pos)
            and np.all((0 <= x[:,:,0]) & (x[:,:,0] < x[:,:,1]) & (x[:,:,1] <= 1440)),
            "Malformed scheduling inputs")
    # Sorting makes the immediate successor the smallest later start. The
    # prefix maximum is the farthest end among earlier bookings. Strict
    # comparisons implement half-open intervals, including duplicate starts.
    order = np.argsort(x[:,:,0], axis=1, kind="stable")
    starts = np.take_along_axis(x[:,:,0], order, axis=1)
    ends = np.take_along_axis(x[:,:,1], order, axis=1)
    conflicts = np.zeros(starts.shape, np.bool_)
    conflicts[:,1:] |= np.maximum.accumulate(ends, axis=1)[:,:-1] > starts[:,1:]
    conflicts[:,:-1] |= ends[:,:-1] > starts[:,1:]
    labels = np.empty(starts.shape, np.uint8)
    np.put_along_axis(labels, order, conflicts, axis=1)
    return labels


def keys_for(task):
    return [f"{task}_{split}_{size}" for size in SIZES for split in
            (("dev", "test") if size in (8,16) else ("test",))]


def evaluate(run):
    identities = verify_package(run)
    output = run/"output"
    report = read(output/"application-results.json")
    locks = read(HERE/"corpus-lock.json")
    require(set(report["tasks"]) == set(locks) == set(TASKS) and
            report["gpu"] == identities["device"] and report["dtype"] == "float64" and report["batch_size"] == 16,
            "Reported classifier configuration mismatch")
    task_results = {}
    all_keys = [k for task in TASKS for k in keys_for(task)]
    with np.load(HERE/"cases.npz", allow_pickle=False) as cases, np.load(
            output/"predictions.npz", allow_pickle=False) as predictions:
        require(set(cases.files) == {k + s for k in all_keys for s in ("_x","_y")}
                and set(predictions.files) == {k for k in all_keys if "_test_" in k},
                "Incomplete or unexpected array coverage")
        for task in TASKS:
            rep = report["tasks"][task]
            spec, vocab = verify_model(task, rep["model"])
            rows = {r["size"]: r for r in rep["rows"]}
            require(len(rows) == len(rep["rows"]) == len(SIZES) and set(rows) == set(SIZES)
                    and set(locks[task]) == set(keys_for(task)), f"Missing or duplicate sizes for {task}")
            verified, dev = [], []
            seen = set()
            for key in keys_for(task):
                _, split, n = key.split("_")
                n = int(n)
                count = 12 if split == "dev" else 64
                x, y = cases[key+"_x"], cases[key+"_y"]
                lock = locks[task][key]
                require(x.shape == (count,n,len(vocab)) and x.dtype == np.int64 and
                        y.shape == (count,n) and y.dtype == np.uint8 and np.isin(y,(0,1)).all(),
                        f"Invalid corpus shape, dtype or labels in {key}")
                require(all(np.all((0 <= x[:,:,i]) & (x[:,:,i] < limit)) for i,limit in enumerate(vocab)),
                        f"Inputs outside embedding ranges in {key}")
                require(tensor_digest(x)==lock["inputs"] and tensor_digest(y)==lock["labels"] and
                        lock["count"] == count, f"Corpus lock mismatch in {key}")
                distinct = {tensor_digest(stream) for stream in x}
                require(len(distinct) == count and seen.isdisjoint(distinct), f"Repeated record within task at {key}")
                seen.update(distinct)
                target = reference(task,x)
                require(np.array_equal(y,target), f"Independent reference disagrees with stored labels at {key}")
                if split == "dev":
                    require(tensor_digest(target) == lock["literal_predictions"], f"Frozen DEV digest differs for {key}")
                    dev.append({"size": n, "records": count, "decisions": int(y.size)})
                    continue
                p = predictions[key]
                require(p.shape == target.shape and p.dtype == np.uint8 and np.isin(p,(0,1)).all(),
                        f"Invalid saved predictions in {key}")
                correct = p == target
                exact = correct.all(axis=1)
                metrics = {"records": count, "exact": int(exact.sum()), "decisions": int(target.size),
                           "correct": int(correct.sum()), "positive_labels": int(target.sum()),
                           "false_positives": int(np.count_nonzero((p==1) & (target==0))),
                           "false_negatives": int(np.count_nonzero((p==0) & (target==1)))}
                row = rows[n]
                budget = {"tokens":n,"tokens_plus_one":n+1,"log2_plus_one":(n-1).bit_length()+1}[spec["round_rule"]]
                require(all(row[k] == val for k,val in metrics.items()) and
                        tensor_digest(p) == row["predictions_sha256"] and row["rounds"] == budget,
                        f"Saved prediction scores, digest or loop budget disagree at {key}")
                require(type(row["stable_records"]) is int and 0 <= row["stable_records"] <= count
                        and all(math.isfinite(row[k]) and row[k] >= 0 for k in ("oracle_seconds","gpu_seconds")),
                        f"Invalid run-reported stability count or timing at {key}")
                errors = [{"record": int(i), "positions": np.flatnonzero(~correct[i]).tolist()}
                          for i in np.flatnonzero(~exact)]
                verified.append({"size":n, **metrics, "predictions_sha256": tensor_digest(p),
                                 "incorrect_records_zero_based": errors,
                                 "reported_stable_records": row["stable_records"]})
            totals = {k: sum(r[k] for r in verified) for k in SCORE_KEYS}
            task_results[task] = {"model":rep["model"], "dev_labels_verified":dev,
                                  "sizes":verified, "totals":totals}
    totals = {k: sum(t["totals"][k] for t in task_results.values()) for k in SCORE_KEYS}
    return {"verified":True, "all_predictions_correct":totals["correct"] == totals["decisions"],
            "run":identities, "tasks":task_results, "totals":totals,
            "stability_evidence":"Only R predictions are archived. Stability at 2R and 4R is recorded by the GPU run."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("version_folder", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    result = evaluate(args.version_folder.resolve())
    local_path = args.destination.parent / "numpy-parity.json"
    if local_path.is_file():
        cpu = read(local_path)
        for task, verified in result["tasks"].items():
            local = cpu[task]
            require(local["model_sha256"] == verified["model"]["model_sha256"], "NumPy model fingerprint differs")
            for row in verified["sizes"]:
                expected = local["sizes"][str(row["size"])]
                require(all(row[key] == expected[key] for key in
                            ("records", "exact", "decisions", "correct", "predictions_sha256")),
                        f"NumPy and Kaggle inference differ for {task} size {row['size']}")
        result["numpy_gpu_prediction_match_all_sizes"] = True
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Verified {result['totals']['records']} records and {result['totals']['decisions']} predictions.")


if __name__=="__main__":
    main()
