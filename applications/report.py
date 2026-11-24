"""Render the held-out application benchmarks from one archived Kaggle version."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("inventory", "scheduling", "telemetry")
SIZES = (8, 16, 32, 64, 128)
COUNT = 64
RULES = {
    "inventory": "A positive label flags a negative on-hand balance after a transaction. Each stream starts with a signed opening balance from −24 through 24 and contains signed quantities from −16 through 16. The generator includes shortfalls, recoveries and balances of exactly zero.",
    "scheduling": "A positive label flags a booking that overlaps another booking for the same resource. Times are whole minutes within one 1,440-minute day. Bookings use half-open intervals, so adjacent endpoints can share a time without conflict. Inputs include shuffled duplicates, nested appointments, overlaps and disjoint bookings.",
    "telemetry": "A positive label means the alarm is on after that reading. It starts off, switches on at 70 or above, resets at 30 or below and preserves its state between the thresholds. Generated values span 0 through 100 and include exact thresholds, brief pulses and long holds.",
}
COUNTERS = ("records", "exact", "decisions", "correct", "stable_records", "positive_labels", "false_positives", "false_negatives")


def read(path):
    return json.loads(path.read_bytes())


def rows_for(result):
    if set(result["tasks"]) != set(TASKS):
        raise ValueError("The archive must contain the three application tasks")
    if result["dtype"] != "float64" or not result["gpu"] or result["batch_size"] < 1:
        raise ValueError("Incomplete GPU execution metadata")
    rows = []
    for task in TASKS:
        group = result["tasks"][task]
        model = group["model"]
        existing = sorted(group["rows"], key=lambda r: r["size"])
        if [r["size"] for r in existing] != list(SIZES):
            raise ValueError(f"Missing or duplicate size for {task}")
        if not 0 < model["nonzero_parameters"] <= model["parameters"]:
            raise ValueError(f"Invalid parameter counts for {task}")
        for row in existing:
            n = row["size"]
            if any(type(row[k]) is not int or row[k] < 0 for k in COUNTERS):
                raise ValueError(f"Invalid metric for {task}/{n}")
            cases, decisions, pos = row["records"], row["decisions"], row["positive_labels"]
            fp, fn = row["false_positives"], row["false_negatives"]
            wrong = fp + fn
            failed = cases - row["exact"]
            if (cases != COUNT or decisions != cases * n or row["correct"] != decisions - wrong or
                    fn > pos or fp > decisions - pos or row["stable_records"] > cases or
                    not 0 <= failed <= wrong <= n * failed or row["rounds"] < 1):
                raise ValueError(f"Inconsistent scores for {task}/{n}")
            for metric in ("gpu_seconds", "oracle_seconds"):
                if metric == "oracle_seconds" and row.get(metric) is None:
                    continue
                if not math.isfinite(row[metric]) or row[metric] < 0:
                    raise ValueError(f"Invalid {metric} for {task}/{n}")
            rows.append({**row, "task": task, "name": group["name"], "unit": group["unit"],
                         "size_range": "development length" if n <= 16 else "longer length",
                         "gpu": result["gpu"], "dtype": result["dtype"], "batch_size": result["batch_size"],
                         "parameters": model["parameters"], "nonzero_parameters": model["nonzero_parameters"],
                         "model_sha256": model["model_sha256"], "predicted_positives": pos - fn + fp})
    return rows


def examples(version, receipt, rows):
    cases_path = ROOT / "applications/cases.npz"
    if hashlib.sha256(cases_path.read_bytes()).hexdigest() != receipt["input_sha256"]["cases.npz"]:
        raise ValueError("The local example corpus differs from the submitted archive")
    output = {}
    with np.load(cases_path, allow_pickle=False) as cases, np.load(version / "output/predictions.npz", allow_pickle=False) as pred:
        for row in (r for r in rows if r["size"] == 8):
            task = row["task"]
            key = f"{task}_test_8"
            xs, ys, predictions = cases[key + "_x"], cases[key + "_y"], pred[key]
            descriptor = json.dumps([predictions.shape, predictions.dtype.str]).encode()
            if hashlib.sha256(descriptor + np.ascontiguousarray(predictions).tobytes()).hexdigest() != row["predictions_sha256"]:
                raise ValueError(f"Example predictions differ from the Kaggle result for {task}")
            if predictions.shape != ys.shape or predictions.shape != (COUNT, 8) or xs.shape[:2] != ys.shape:
                raise ValueError(f"Malformed size-eight example arrays for {task}")
            # Choose by reference label diversity, independently of model output.
            mixed = np.flatnonzero((ys.sum(axis=1) > 0) & (ys.sum(axis=1) < ys.shape[1]))
            idx = int(mixed[0]) if len(mixed) else 0
            x = xs[idx]
            if task == "inventory":
                data = {"opening": int(x[0, 2]) - 25, "changes": (x[:, 1] - 16).tolist()}
            elif task == "scheduling":
                data = {"bookings": x[:, :2].tolist()}
            else:
                data = {"readings": x[:, 0].tolist()}
            flag = lambda values: ", ".join(str(i + 1) for i in np.flatnonzero(values)) or "none"
            output[task] = (idx + 1, json.dumps(data, separators=(",", ":")), flag(predictions[idx]), flag(ys[idx]))
    return output


def markdown(result, receipt, rows, version, dest, samples):
    def link(label, path):
        return f"[{label}](<{Path(os.path.relpath(path, dest)).as_posix()}>)"

    def total(key):
        return sum(row[key] for row in rows)

    output = version / "output"
    lines = ["# Looped transformers on three operational tasks", "",
             f"The frozen models labeled **{total('exact'):,} of {total('records'):,} complete records** correctly, "
             f"with **{total('correct'):,} of {total('decisions'):,} individual decisions** correct. "
             f"They produced {total('false_positives'):,} false positives and {total('false_negatives'):,} false negatives. "
             f"On {total('stable_records'):,} records, every prediction stayed the same at the three loop budgets. "
             f"[Private Kaggle run, version {receipt['version']}]({receipt['url']})", "",
             "## Evaluation", "",
             "These are three synthetic application benchmarks created for this reproduction. Each task has a separate transformer with analytic weights and an independent conventional scoring rule. Development used 12 records per task at each of lengths 8 and 16. The held-out run covers 64 distinct new records per task at each of lengths 8, 16, 32, 64 and 128. A deterministic generator uses different seeds for development and held-out inputs, and excludes duplicate records across splits at each size. The frozen cases, labels and model hashes identify the evaluated data and weights. The results measure these bounded synthetic distributions. "
             + link("Corpus lock", ROOT / "applications/corpus-lock.json"), "",
             f"The GPU run used {result['gpu']}, {result['dtype']} and batches of up to {result['batch_size']} records. Each held-out record was scored once, with readouts at the base loop budget R, 2R and 4R during the same evaluation. A record is exact when every output matches its reference label. Stability counts whole records with identical predictions at all three checkpoints. GPU times are the summed batch times through R. CPU oracle times measure conventional reference-label calculation on the Kaggle host and serve as timing context only. "
             + link("Evaluation code", ROOT / "applications/experiment.py"), "",
             "Each example below shows all eight inputs from one held-out record. Where available, the first record with both positive and negative reference labels is used. Flagged positions are numbered from 1. To score a raw JSON record, use `python applications/predict.py inventory record.json`, substituting `scheduling` or `telemetry` for those tasks. The command also accepts standard input and returns one boolean per input. "
             + link("Prediction CLI", ROOT / "applications/predict.py"), ""]
    for task in TASKS:
        group = [r for r in rows if r["task"] == task]
        descriptor = group[0]
        idx, raw, predicted, reference = samples[task]
        lines.extend([f"## {descriptor['name']}", "", RULES[task], "",
                      f"**Example, record {idx}.** `{raw}`", "",
                      f"The model flagged positions **{predicted}**. The reference flagged **{reference}**.", "",
                      f"The model contains **{descriptor['parameters']:,} scalar entries**, including zeros, and **{descriptor['nonzero_parameters']:,} nonzero entries**. "
                      + " · ".join([link("Model and generator", ROOT / f"applications/{task}.py"),
                                    link("Freeze record", ROOT / f"applications/models/{task}/freeze.json"),
                                    link("Weight archive", ROOT / f"applications/models/{task}/weights.npz")]), "",
                      f"| {descriptor['unit'].capitalize()} per record | R | Exact / records | Correct / decisions | Positive labels | False positives | False negatives | Stable records | GPU seconds | CPU oracle seconds |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for r in group:
            reference_time = f"{r['oracle_seconds']:.6f}" if r.get("oracle_seconds") is not None else "—"
            lines.append(f"| {r['size']} | {r['rounds']} | {r['exact']:,} / {r['records']:,} | {r['correct']:,} / {r['decisions']:,} | "
                         f"{r['positive_labels']:,} | {r['false_positives']:,} | {r['false_negatives']:,} | "
                         f"{r['stable_records']:,} | {r['gpu_seconds']:.6f} | {reference_time} |")
        lines.append("")
    lines += ["## Archive and verification", "", " · ".join([
        link("All size results (CSV)", dest / "sizes.csv"), link("Kaggle result JSON", output / "application-results.json"),
        link("Kaggle predictions", output / "predictions.npz"), link("Independent local verification", dest / "verification.json"),
        link("Submission receipt", version / "submission.json"), link("Synthetic cases", ROOT / "applications/cases.npz")]), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", type=Path, help="Archived Kaggle version folder with submission.json and output/")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    version, dest = args.version.resolve(), args.destination.resolve()
    result = read(version / "output/application-results.json")
    receipt = read(version / "submission.json")
    rows = rows_for(result)
    report = markdown(result, receipt, rows, version, dest, examples(version, receipt, rows))
    fields = ["task", "name", "size", "unit", "size_range", "parameters", "nonzero_parameters", *COUNTERS,
              "predicted_positives", "rounds", "gpu", "dtype", "batch_size", "gpu_seconds", "oracle_seconds",
              "model_sha256", "predictions_sha256"]
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "sizes.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (dest / "report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
