"""Render the loop comparison from its frozen labels, predictions and timing receipt."""

import argparse
import csv
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "comparisons/protocol.json"
PUBLISHED = ROOT / "applications/cases.npz"
CHALLENGE = ROOT / "comparisons/challenge.npz"
COHORTS = ("published", "challenge")
METRICS = ("n_records", "exact", "n_labels", "correct", "invalid", "false_positives",
           "false_negatives", "final_correct")
PAIRED = ("full_only", "one_only", "both_correct", "both_incorrect")
ORIGINAL = ("original_1", "original_4", "original_full")
TASK_NAMES = {"inventory": "Inventory", "scheduling": "Booking conflicts", "telemetry": "Alarm state"}
METHOD_NAMES = {"original_1": "1 pass", "original_4": "4 passes", "original_full": "Full depth",
                "full_feedforward": "Global feedforward", "local_w2": "Window 2", "local_w8": "Window 8",
                "local_w32": "Window 32"}


def read(path):
    return json.loads(path.read_bytes())


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def target_metrics(prediction, target):
    valid = prediction < 2
    correct = valid & (prediction == target)
    return {
        "n_records": len(target), "exact": int(correct.all(axis=1).sum()),
        "n_labels": int(target.size), "correct": int(correct.sum()), "invalid": int((~valid).sum()),
        "false_positives": int(((prediction == 1) & (target == 0)).sum()),
        "false_negatives": int(((prediction == 0) & (target == 1)).sum()),
        "final_correct": int(correct[:, -1].sum()),
    }, correct


def unpack(result, protocol, pred_path):
    expected = set(product(protocol["tasks"], protocol["sizes"], protocol["precisions"], protocol["methods"]))
    cases, scores, exact_masks = {}, {}, {}
    with np.load(PUBLISHED, allow_pickle=False) as released, np.load(CHALLENGE, allow_pickle=False) as challenge:
        for t, n in product(protocol["tasks"], protocol["sizes"]):
            cases[t, n] = (released[f"{t}_test_{n}_y"], challenge[f"{t}_{n}_y"])
    with np.load(pred_path, allow_pickle=False) as stored:
        for entry in result["rows"]:
            t, n, fp, method = (entry[k] for k in ("task", "size", "precision", "method"))
            key = (t, n, fp, method)
            if key in scores or key not in expected:
                raise ValueError(f"Unexpected or repeated measurement {key}")
            seconds = entry["seconds"]
            params, nonzeros = entry["parameters"], entry["nonzero_parameters"]
            if (len(seconds) != protocol["timed_repeats"] or not np.isfinite(seconds).all()
                    or min(seconds) <= 0 or median(seconds) != entry["median_seconds"]
                    or not 0 <= nonzeros <= params):
                raise ValueError(f"Invalid timing or parameter counts for {key}")
            row = {k: entry[k] for k in ("task", "size", "precision", "method", "steps", "parameters",
                                          "nonzero_parameters", "median_seconds", "prediction_key")}
            for i, sec in enumerate(seconds, 1):
                row[f"seconds_{i}"] = sec
            row["raw_prediction_key"] = entry.get("raw_prediction_key", entry["prediction_key"])
            for mode in ("", "raw_"):
                prediction = stored[row[mode + "prediction_key"]]
                if prediction.dtype != np.uint8 or prediction.shape != (96, n) or (prediction > 2).any():
                    raise ValueError(f"Invalid prediction array for {key}")
                for j, (cohort, pred) in enumerate(zip(COHORTS, (prediction[:64], prediction[64:]))):
                    metrics, correct = target_metrics(pred, cases[t, n][j])
                    if cohort == "challenge":
                        metrics["pairs_both_final_correct"] = int((correct[::2, -1] & correct[1::2, -1]).sum())
                    supplied = entry.get(mode + cohort, entry[cohort])
                    if any(supplied[k] != value for k, value in metrics.items()):
                        raise ValueError(f"Archived ratings differ from {mode}{cohort} predictions for {key}")
                    row.update({f"{mode}{cohort}_{k}": v for k, v in metrics.items()})
                    if not mode and method in ("original_1", "original_full"):
                        exact_masks[key, cohort] = correct.all(axis=1)
            scores[key] = row
    if set(scores) != expected:
        raise ValueError(f"Expected {len(expected)} configurations, found {len(scores)}")
    for t, n, fp in product(protocol["tasks"], protocol["sizes"], protocol["precisions"]):
        trio = [scores[t, n, fp, m] for m in ORIGINAL]
        if len({(r["parameters"], r["nonzero_parameters"]) for r in trio}) != 1:
            raise ValueError("Depth comparison changed the stored parameter counts")
        for cohort in COHORTS:
            one = exact_masks[(t, n, fp, "original_1"), cohort]
            full = exact_masks[(t, n, fp, "original_full"), cohort]
            for name, mask in zip(PAIRED, (full & ~one, one & ~full, full & one, ~full & ~one)):
                trio[-1][f"{cohort}_{name}"] = int(mask.sum())
    return scores


def table_rows(scores, protocol):
    for key in product(protocol["tasks"], protocol["sizes"], protocol["precisions"], protocol["methods"]):
        row = scores[key].copy()
        for cohort, name in product(COHORTS, PAIRED):
            row.setdefault(f"{cohort}_{name}", "")
        yield row


def budget_rows(scores, protocol):
    for t, fp in product(protocol["tasks"], protocol["precisions"]):
        for size in ["all", *protocol["sizes"]]:
            sizes = protocol["sizes"] if size == "all" else [size]
            for limit, cohort in product(protocol["parameter_budgets"], COHORTS):
                candidates = {}
                for method in protocol["methods"]:
                    rows = [scores[t, n, fp, method] for n in sizes]
                    counts = {r["parameters"] for r in rows}
                    if len(counts) != 1:
                        raise ValueError(f"Parameter count changed with length for {t}/{method}")
                    params = counts.pop()
                    if params <= limit:
                        candidates[method] = (sum(r[f"{cohort}_exact"] for r in rows), params)
                top = max((val[0] for val in candidates.values()), default=None)
                winners = [m for m, val in candidates.items() if val[0] == top]
                yield {"task": t, "size": size, "precision": fp, "cohort": cohort, "parameter_budget": limit,
                       "n_records": (64 if cohort == "published" else 32) * len(sizes),
                       "best_exact": top if top is not None else "", "eligible_count": len(candidates),
                       "eligible_methods": "|".join(candidates), "best_methods": "|".join(winners),
                       "best_methods_parameters": "|".join(str(candidates[m][1]) for m in winners)}


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown(scores, protocol, result, receipt, version, dest, pred_path, budgets):
    tasks, sizes, fps = (protocol[k] for k in ("tasks", "sizes", "precisions"))

    def total(t, precision, method, column):
        return sum(scores[task, n, precision, method][column]
                   for task in (tasks if t is None else [t]) for n in sizes)

    def link(label, path):
        return f"[{label}](<{Path(os.path.relpath(path, dest)).as_posix()}>)"

    def table(headings, rows):
        return ["| " + " | ".join(headings) + " |", "|" + "|".join("---" for _ in headings) + "|",
                *["| " + " | ".join(str(x) for x in r) + " |" for r in rows], ""]

    fp64 = "float64"
    orig, shallow, feed = "original_full", "original_1", "full_feedforward"
    wins = sum(total(None, fp64, orig, f"{c}_full_only") for c in COHORTS)
    losses = sum(total(None, fp64, orig, f"{c}_one_only") for c in COHORTS)
    heads = ["# When does looping help?", "",
             f"In FP64, changing only the depth of the frozen transformer corrected **{wins:,}** whole records "
             f"that one pass missed and lost **{losses:,}** that one pass got right. The full-depth model matched "
             f"**{total(None, fp64, orig, 'published_exact'):,}/960** released records and "
             f"**{total(None, fp64, orig, 'challenge_exact'):,}/480** challenge records. The task-specific global "
             f"feedforward models matched **{total(None, fp64, feed, 'published_exact'):,}/960** and "
             f"**{total(None, fp64, feed, 'challenge_exact'):,}/480**, respectively.", "",
             "Each task contributes 64 released records and 16 new pairs at each of five lengths. Pairs share "
             "every input after the first and have opposite final labels. Whole-record accuracy requires every "
             "event, booking, or reading to be correct. A pair passes the final-label check only when both members "
             "have the correct final decision.", ""]
    date = receipt.get("submitted_utc") or receipt.get("submitted_at_utc") or receipt.get("submitted_at")
    receipt_label = f"Private Kaggle run, version {receipt['version']}"
    if date:
        receipt_label += f", {str(date)[:10]} UTC"
    heads.extend([f"Execution used {result['gpu']}. [{receipt_label}]({receipt['url']}) · "
                  +link("Run receipt", version/"submission.json")+" · "+link("Frozen experimental protocol", PROTO),
                  "", "![Whole-record accuracy by model, precision and input length](accuracy.png)", "",
                  "## Depth and the final decision", "",
                  "Counts below pool the five lengths. In the paired depth comparison, a full-only match means "
                  "the entire record became correct at full depth. A one-only match means it became incorrect. "
                  "Both runs used exactly the same stored coefficients. The inventory block doubles its coverage "
                  "per pass. The booking and alarm blocks pass information between adjacent input positions.", ""])
    data = []
    for t, fp in product(tasks, fps):
        a = lambda method, key: total(t, fp, method, key)
        contrasts = [f"{a(orig, c+'_full_only')} / {a(orig, c+'_one_only')}" for c in COHORTS]
        pairs = " / ".join(str(a(m, "challenge_pairs_both_final_correct")) for m in (shallow, orig, feed))
        data.append([TASK_NAMES[t], fp, *contrasts, pairs])
    heads.extend(table(["Task", "Precision", "Released full-only / one-only", "Challenge full-only / one-only",
                        "Both final labels correct in 80 pairs, one / full / global"], data))
    heads.extend(["For these constructed pairs, a deterministic final decision that cannot see the first row "
                  "receives identical inputs for both members. Their opposite labels limit it to one correct "
                  "final decision per pair. Inventory and alarm windows reach the first row when the length is "
                  "at most the window width W. The symmetric booking window reaches it when the index distance "
                  "from the first to last row, length − 1, is at most W/2. A global feedforward computation also "
                  "has access to the first row.", ""])
    heads.extend(["## Results by sequence length", "",
                  "Each FP64 cell gives two whole-record counts, **released out of 64 · challenge out of 32**. "
                  "Windows use the indicated number of taps. The CSV includes all three precisions, final-label "
                  "scores, invalid decisions, and paired depth outcomes for each length. "+link("All 315 measurements", dest/"sizes.csv"), ""])
    data = []
    for t, n in product(tasks, sizes):
        row = [TASK_NAMES[t], n]
        for m in protocol["methods"]:
            r = scores[t, n, fp64, m]
            row.append(f"{r['published_exact']} · {r['challenge_exact']}")
        data.append(row)
    heads.extend(table(["Task", "Length", *(METHOD_NAMES[m] for m in protocol["methods"])], data))
    heads.extend(["## Precision and invalid outputs", "",
                  "Each cell gives released exact records out of 320 and challenge exact records out of 160. "
                  "The same 480 records per task were evaluated at every precision.", ""])
    data = []
    for t, fp in product(tasks, fps):
        columns = [" · ".join(str(total(t, fp, m, c+"_exact")) for c in COHORTS) for m in (shallow, orig, feed)]
        data.append([TASK_NAMES[t], fp, *columns])
    heads.extend(table(["Task", "Precision", "One pass", "Full depth", "Global feedforward"], data))
    heads.extend([
                  "The frozen transformer tracks nonfinite attention scores, updates, hidden computations, "
                  "and logits. A flagged token is incorrect in the strict score. The raw score uses the final "
                  "logits alone and counts nonfinite output scores as incorrect. An intermediate overflow can "
                  "disappear inside softmax or ReLU, so a finite final value can still follow a numerical failure. "
                  "The table pools 480 records and 23,808 token decisions per task at FP16.", ""])
    data = []
    for t, m in product(tasks, ORIGINAL):
        a = lambda mode, key: sum(total(t, "float16", m, mode+c+"_"+key) for c in COHORTS)
        data.append([TASK_NAMES[t], METHOD_NAMES[m], a("", "exact"), a("raw_", "exact"),
                     f"{a('', 'invalid'):,}", f"{a('raw_', 'invalid'):,}"])
    heads.extend(table(["Task", "Depth", "Strict exact / 480", "Raw exact / 480",
                        "Strict invalid decisions", "Raw invalid decisions"], data))
    strict16 = sum(total(None, "float16", orig, c+"_invalid") for c in COHORTS)
    if strict16:
        raw_passes = sum(total(None, "float16", orig, "raw_"+c+"_exact") for c in COHORTS)
        heads.extend([f"At full depth the FP16 numerical screen flagged **{strict16:,}** token decisions. "
                      f"Using the final logits alone, **{raw_passes:,}/1,440** records were exact.", ""])
    heads.extend(["## Parameter budgets", "",
                  "The three depth variants share tensors. The feedforward designs encode task knowledge in their "
                  "operators, including a causal prefix aggregation, interval-pair comparisons, and selection of the "
                  "most recent threshold event. Counts cover stored coefficients, including zeros. Masks, indices, "
                  "and fixed operations have no stored scalar cost in this accounting. The counts therefore cannot "
                  "establish a universal minimum or account for all implementation complexity.", ""])
    data = [[TASK_NAMES[t], *[scores[t, sizes[0], fp64, m]["parameters"]
                              for m in [orig, feed, "local_w2", "local_w8", "local_w32"]]] for t in tasks]
    heads.extend(table(["Task", "Original at all depths", "Global feedforward", "Window 2", "Window 8", "Window 32"], data))
    heads.extend(["Each budget cell gives the highest FP64 **released / 320 · challenge / 160** whole-record counts "
                  "among eligible methods, pooled across lengths. Each cohort selects its own best method. A dash "
                  "means no measured method fit. The CSV identifies every tied method and its actual parameter count "
                  "at each precision and length. "+link("Parameter-budget results", dest/"parameter-budgets.csv"), ""])
    frontier = {(r["task"], r["cohort"], r["parameter_budget"]): r["best_exact"] for r in budgets
                if r["size"] == "all" and r["precision"] == fp64}
    data = [[TASK_NAMES[t], *[" · ".join(str(frontier[t, c, b]) if frontier[t,c,b] != "" else "—" for c in COHORTS)
                               for b in protocol["parameter_budgets"]]] for t in tasks]
    heads.extend(table(["Task", *(f"≤ {b:,}" for b in protocol["parameter_budgets"])], data))
    heads.extend(["### Window width at 128 items", "",
                  "Each FP32 cell gives **parameter count (released exact out of 64 · challenge exact out of 32)**. "
                  "A wider local window adds tap coefficients and exposes more input rows, so both factors change "
                  "together. The paired challenge changes row 1 and checks the final decision at row 128, "
                  "127 positions away. The widest local inventory and alarm models can look back 31 positions. "
                  "The widest local booking model can inspect 16 neighbors on each side.", ""])
    methods = ("local_w2", "local_w8", "local_w32", feed, orig)
    data = []
    for t in tasks:
        row = [TASK_NAMES[t]]
        for m in methods:
            measured = scores[t, 128, "float32", m]
            row.append(f"{measured['parameters']:,} ({measured['published_exact']} · {measured['challenge_exact']})")
        data.append(row)
    heads.extend(table(["Task", *(METHOD_NAMES[m] for m in methods)], data))
    heads.extend(["## Runtime and evidence", "",
                  "Times are synchronized GPU wall times for 96 preloaded records in batches of 16. Each cell "
                  "below is the median of three runs in milliseconds at length 128 and FP32. The transformer "
                  "recomputes dense attention at every requested step, including intermediate finite-value checks. "
                  "The operators and checks differ for the task-specific baselines. Data transfer, grading, "
                  "model setup, and the Kaggle queue fall outside the timer.", ""])
    data = [[TASK_NAMES[t], *[f"{1000*scores[t, 128, 'float32', m]['median_seconds']:.3f}"
                              for m in protocol["methods"]]] for t in tasks]
    heads.extend(table(["Task", *(METHOD_NAMES[m] for m in protocol["methods"])], data))
    heads.extend(["The comparison covers the three synthetic generators and the measured lengths of 8 through "
                  "128 items. All models were written analytically. The study supplies no measurements on "
                  "customer records, learned deep models, or inputs beyond the frozen positional tables.", "",
                  " · ".join([link("Measured result JSON", version/"output/comparison-results.json"),
                              link("Archived predictions", pred_path), link("Released cases", PUBLISHED),
                              link("Paired challenges", CHALLENGE), link("Frozen transformer forward", ROOT/"comparisons/recurrent.py"),
                              link("Feedforward operators", ROOT/"comparisons/baselines.py"),
                              link("Study design and reproduction", ROOT/"comparisons/README.md"),
                              link("GPU evaluator", ROOT/"comparisons/experiment.py"),
                              link("Independent audit", dest/"audit.json"),
                              link("Challenge generator", ROOT/"comparisons/make_challenges.py")]), ""])
    return "\n".join(heads)


def generate(version, dest):
    version, dest = version.resolve(), dest.resolve()
    protocol, result = read(PROTO), read(version/"output/comparison-results.json")
    receipt = read(version/"submission.json")
    if result["protocol_sha256"] != file_hash(PROTO):
        raise ValueError("Result protocol hash differs from the report protocol")
    if file_hash(PUBLISHED) != protocol["cases_sha256"] or file_hash(CHALLENGE) != protocol["challenge_sha256"]:
        raise ValueError("A frozen corpus changed")
    pred_path = next((version/"output"/s for s in ("comparison-predictions.npz", "predictions.npz")
                      if (version/"output"/s).is_file()), None)
    if pred_path is None:
        raise FileNotFoundError("Missing comparison predictions archive")
    scores = unpack(result, protocol, pred_path)
    budgets = list(budget_rows(scores, protocol))
    text = markdown(scores, protocol, result, receipt, version, dest, pred_path, budgets)
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest/"sizes.csv", table_rows(scores, protocol))
    write_csv(dest/"parameter-budgets.csv", budgets)
    (dest/"report.md").write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", type=Path, help="Kaggle run directory containing submission.json and output/")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    generate(args.version, args.destination)
