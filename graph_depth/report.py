"""Render the graph-depth results from a collected Kaggle run and its predictions."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
TASKS = {"pointer": "Pointer chasing", "reachability": "Directed reachability"}
NAMES = {
    "loop_shared": "Tied, 6 stages", "fixed_2": "Untied, 2 stages",
    "fixed_4": "Untied, 4 stages", "fixed_6": "Untied, 6 stages",
    "untied_full": "Full-width untied, 6 stages",
    "local_tied": "Tied local, 63 stages", "local_untied_1": "Untied local, 1 stage",
    "local_untied_2": "Untied local, 2 stages", "local_untied_4": "Untied local, 4 stages",
    "global_untied_4": "Untied global, 4 stages", "global_untied_6": "Untied global, 6 stages",
    "local_unrolled": "Full-width untied local, 63 stages",
}


def read_json(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def methods(design):
    return [*design["matched"], design["unrolled"]]


def read_run(version):
    receipt = read_json(version / "submission.json")
    verified = read_json(version / "verification.json")
    if (verified["status"] != "COMPLETE" or verified["version"] != receipt["version"]
            or verified["kernel"] != receipt["kernel"] or not verified["source_match"]):
        raise ValueError("Expected a completed version with a verified source match")
    for name in ("protocol.json", "cases-manifest.json"):
        if digest(HERE / name) != receipt["input_sha256"][name]:
            raise ValueError(f"Current {name} differs from the submitted file")
    payload = version / "output"
    for name in ("graph-depth-results.json", "graph-depth-predictions.npz"):
        if digest(payload / name) != verified["output_sha256"][name]:
            raise ValueError(f"Collected output digest differs for {name}")
    results = read_json(payload / "graph-depth-results.json")
    protocol = read_json(HERE / "protocol.json")
    if (results["split"] != "test" or results["device"] != verified["device"]
            or results["precision"] != protocol["precision"]):
        raise ValueError("Result split, device or precision differs from the study protocol/receipt")
    return receipt, results, protocol


def validated_rows(version, protocol, results):
    bins, count = protocol["test"]["hops"], protocol["test"]["pairs_per_bin"]
    repeats = protocol["timing"]["full_cohort_repeats"]
    manifest = read_json(HERE / "cases-manifest.json")["arrays"]
    by_key = {(r["task"], r["budget"], r["method"], r["hops"]): r for r in results["conditions"]}
    expected = {(t, b, m, h) for t, d in protocol["tasks"].items()
                for b in d["budgets"] for m in methods(d) for h in bins}
    if set(by_key) != expected or len(results["conditions"]) != len(expected):
        raise ValueError("Missing, duplicate or unexpected result conditions")
    with (np.load(HERE / "cases.npz", allow_pickle=False) as cases,
          np.load(version / "output/graph-depth-predictions.npz", allow_pickle=False) as archive):
        names = {r["prediction_key"] for r in by_key.values()}
        if len(names) != len(expected) or names != set(archive.files):
            raise ValueError("Expected one archived prediction array per condition")
        for task, design in protocol["tasks"].items():
            for hops in bins:
                key = f"{task}_test_{hops}_label"
                gold = cases[key]
                if gold.shape != (count*2,) or sha256(gold.tobytes()).hexdigest() != manifest[key]["sha256"]:
                    raise ValueError(f"Unexpected label shape or hash for {key}")
                for idx, budget in enumerate(design["budgets"]):
                    ref_row = by_key[task, budget, design["reference"], hops]
                    ref = archive[ref_row["prediction_key"]]
                    ref_ok = ref == gold
                    for method in methods(design):
                        row = by_key[task, budget, method, hops]
                        pred = archive[row["prediction_key"]]
                        n_classes = protocol["nodes"] if task == "pointer" else 2
                        if (pred.shape != gold.shape or not np.issubdtype(pred.dtype, np.signedinteger)
                                or (pred < -1).any() or (pred >= n_classes).any()):
                            raise ValueError(f"Invalid prediction array {row['prediction_key']}")
                        ok = pred == gold
                        ledger = results["ledgers"][f"{task}/{budget}/{method}"]
                        params = budget if method in design["matched"] else design["unrolled_parameters"][idx]
                        checks = {"queries": len(gold), "correct": int(ok.sum()), "invalid": int((pred < 0).sum()),
                                  "pairs": count, "both_pair_members_correct": int(ok.reshape(-1, 2).all(1).sum()),
                                  "stored": params, "stages": design["depths"][method]}
                        if task == "reachability":
                            for name, label in [("reachable", 1), ("unreachable", 0)]:
                                checks[f"{name}_queries"] = int((gold == label).sum())
                                checks[f"{name}_correct"] = int(ok[gold == label].sum())
                        if ledger["stored"] != params or any(row[k] != v for k, v in checks.items()):
                            raise ValueError(f"Result counts disagree for {row['prediction_key']}")
                        times = row["seconds"]
                        if (len(times) != repeats or not all(isfinite(s) and s > 0 for s in times)
                                or row["median_seconds"] != median(times)):
                            raise ValueError(f"Inconsistent measurement times for {row['prediction_key']}")
                        row.update(tied_only=int((ref_ok & ~ok).sum()), comparator_only=int((ok & ~ref_ok).sum()),
                                   different=int((ref != pred).sum()))
    return by_key


def plot(rows, protocol, output):
    bins = protocol["test"]["hops"]
    queries = 2 * protocol["test"]["pairs_per_bin"]
    cmap = LinearSegmentedColormap.from_list("accuracy", ["#faefd9", "#d4dfda", "#207168", "#124a4a"])
    fig, axes = plt.subplots(2, 2, figsize=(14.1, 7.7), gridspec_kw={"height_ratios": [5, 7]})
    fig.subplots_adjust(left=.205, right=.99, top=.838, bottom=.145, hspace=.47, wspace=.09)
    fig.text(.017, .970, "Exact answers by graph path length", fontsize=22, weight="bold", va="top")
    fig.text(.017, .911, f"Each cell counts correct answers among {queries} queries. Each task uses the same queries at both budgets.", fontsize=11, va="top")
    for i, (task, design) in enumerate(protocol["tasks"].items()):
        order = methods(design)
        for j, budget in enumerate(design["budgets"]):
            ax = axes[i, j]
            values = np.array([[rows[task, budget, method, h]["correct"] for h in bins] for method in order])
            ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=queries, interpolation="nearest")
            ax.set_title(f"{TASKS[task]}  ·  {budget:,} parameters", loc="left", fontsize=12, weight="bold", pad=9)
            for (y, x), value in np.ndenumerate(values):
                ax.text(x, y, str(value), ha="center", va="center", fontsize=8.4,
                        color="white" if value >= queries*.59 else "#172721")
            ax.axhline(len(order)-1.5, color="white", linewidth=2.5)
            ax.set_yticks(range(len(order)), [NAMES[m] for m in order] if j == 0 else [""]*len(order), fontsize=9)
            ax.set_xticks(range(len(bins)), bins, fontsize=9)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.text(.205, .069, "Columns show requested hops (pointer) or shortest distance of the reachable partner (reachability).", fontsize=9.7)
    fig.text(.205, .036, "The last row in each panel is the higher-parameter control at the tied model’s width and depth.", fontsize=9.7)
    fig.savefig(output, dpi=175, facecolor="white", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def table(headers, rows):
    lines = [headers, ["---"]*len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def relation_path(path, output):
    return Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()


def write_sizes(path, rows, protocol, results):
    repeats = range(1, protocol['timing']['full_cohort_repeats'] + 1)
    cols = ['task', 'budget', 'method', 'matched_budget', 'hops', 'stages', 'stored', 'nonzero',
            'buffer_scalars', 'queries', 'correct', 'pairs', 'both_pair_members_correct', 'invalid',
            'reachable_queries', 'reachable_correct', 'unreachable_queries', 'unreachable_correct',
            'tied_only', 'comparator_only', 'different', 'median_seconds', *[f'seconds_{i}' for i in repeats]]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        writer.writeheader()
        for task, design in protocol['tasks'].items():
            for budget in design['budgets']:
                for method in methods(design):
                    ledger = results['ledgers'][f'{task}/{budget}/{method}']
                    for h in protocol['test']['hops']:
                        row = rows[task, budget, method, h]
                        data = {c: row[c] for c in cols if c in row}
                        data.update(matched_budget=int(method in design['matched']), nonzero=ledger['nonzero'],
                                    buffer_scalars=ledger['buffer_scalars'])
                        data.update(zip([f'seconds_{i}' for i in repeats], row['seconds']))
                        writer.writerow(data)


def report(rows, protocol, receipt, results, version, output):
    bins = protocol["test"]["hops"]
    pair_count = protocol["test"]["pairs_per_bin"]
    n = 2*pair_count
    totals = {}
    for task, design in protocol["tasks"].items():
        for budget in design["budgets"]:
            for method in methods(design):
                counts = Counter()
                for h in bins:
                    row = rows[task, budget, method, h]
                    keys = ["correct", "queries", "pairs", "both_pair_members_correct", "invalid",
                            "tied_only", "comparator_only", "different"]
                    if task == "reachability":
                        keys.extend(["reachable_correct", "unreachable_correct", "reachable_queries", "unreachable_queries"])
                    counts.update({k: row[k] for k in keys})
                totals[task, budget, method] = counts

    def link(label, path):
        return f"[{label}](<{relation_path(path, output)}>)"

    raw = version / "output"
    receipt_link = link("Run receipt", version / "submission.json")
    verification_link = link("Collected run verification", version / "verification.json")
    when = receipt["submitted_utc"].split("T")[0]
    ptr = protocol["tasks"]["pointer"]
    reach = protocol["tasks"]["reachability"]
    leads = []
    for task, design, m, names, unit in [
        ("pointer", ptr, "fixed_6", "The six-stage tied and untied models", "endpoints"),
        ("reachability", reach, "global_untied_6", "The 63-stage tied local and six-stage untied global models", "connectivity answers"),
    ]:
        cells, perfect = [], True
        for b in design["budgets"]:
            ref, other = totals[task, b, design["reference"]], totals[task, b, m]
            perfect = perfect and ref['correct'] == ref['queries'] and other['correct'] == other['queries']
            cells.append(f"{ref['correct']:,} and {other['correct']:,} at {b:,} parameters")
        if perfect:
            tiers = ' and '.join(f'{b:,}' for b in design['budgets'])
            leads.append(f"**{TASKS[task]}.** {names} each predicted all {n*len(bins):,} {unit} correctly at both budgets ({tiers} parameters).")
        else:
            leads.append(f"**{TASKS[task]}.** {names} predicted the following numbers of {unit} correctly out of {n*len(bins):,}, respectively. {' · '.join(cells)}.")
    sections = [
        "# Parameter sharing and graph reasoning depth",
        *leads,
        f"The held-out set contains {len(bins)*n*len(TASKS):,} queries on {protocol['nodes']}-vertex graphs across both tasks. "
        f"The same queries are reused across methods and budgets. The {results['precision'].upper()} run used a {results['device']} "
        f"with TF32 disabled. [Private Kaggle version {receipt['version']}, submitted {when} UTC]({receipt['url']}) · {receipt_link} · {verification_link}",
        "![Exact answers by graph path length, task and stored parameter budget](accuracy.png)",
        "## Exact answers by path length",
        f"Each cell is a count out of **{n} queries**. Pointer chasing asks for the endpoint after exactly *k* steps. "
        "Reachability asks whether a path exists. The column for an unreachable graph gives its reachable partner’s shortest path length. "
        "The final row in each budget group repeats the tied model’s computation at full width using independently stored blocks. Its actual parameter count appears in the table. [All measurements and three timing repeats per cell](sizes.csv)",
    ]
    for task, design in protocol["tasks"].items():
        data = []
        for budget in design["budgets"]:
            for m in methods(design):
                cells = [rows[task, budget, m, h]["correct"] for h in bins]
                data.append([f"{budget:,}", NAMES[m], f"{rows[task,budget,m,bins[0]]['stored']:,}", *cells])
        sections.extend([f"### {TASKS[task]}", table(["Budget", "Model", "Actual parameters", *bins], data)])

    sections.extend(["## Counterfactual pairs and runtime",
        "A pointer pair swaps two successors and changes the correct endpoint at the requested hop. A reachability pair swaps two edges, preserving every vertex’s in- and outdegree and flipping the connectivity answer. Both tasks keep the source fixed within a pair. Reachability also keeps the target fixed. A pair counts only when both predictions are correct.",
        f"Each model saw {pair_count*len(bins):,} pairs per task. **Tied only** counts individual queries that the tied reference answered correctly and the row model missed. **Row only** is the reverse. "
        f"Timing is the median of the {len(bins)} per-length median wall times for {n} queries. Each length used {protocol['timing']['full_cohort_repeats']} synchronized repeats in batches of {protocol['timing']['batch_size']}, after warm-up. Graph tensors and endpoint IDs were already on the GPU, and the pointer table was in vertex-ID order. The timer included model input encoding, including conversion of the pointer hop count to a device tensor, and forward execution. Grading and disk I/O ran outside the timer. [Paired comparisons at every length](comparisons.csv)"])
    for task, design in protocol["tasks"].items():
        data = []
        extra = ["Reachable correct", "Unreachable correct"] if task == "reachability" else []
        for budget in design["budgets"]:
            for m in methods(design):
                row = totals[task, budget, m]
                classes = [f"{row[c+'_correct']}/{row[c+'_queries']}" for c in ("reachable", "unreachable")] if extra else []
                ms = 1000*median(rows[task, budget, m, h]["median_seconds"] for h in bins)
                data.append([f"{budget:,}", NAMES[m], f"{row['both_pair_members_correct']}/{row['pairs']}",
                             *classes, row['tied_only'], row['comparator_only'], f"{ms:.2f}"])
        sections.extend([f"### {TASKS[task]}", table(["Budget", "Model", "Both correct", *extra, "Tied only", "Row only", "ms"], data)])

    largest = bins[-1]
    depth_rows = []
    long_path_rows = []
    for budget in reach['budgets']:
        one = totals['reachability', budget, 'local_untied_1']
        tied = totals['reachability', budget, 'local_tied']
        depth_rows.append([budget, f"{one['correct']:,}/{one['queries']:,}",
                           f"{tied['correct']:,}/{tied['queries']:,}", one['tied_only'], one['comparator_only']])
        for method in ('local_untied_4', 'local_tied', 'global_untied_6'):
            row = rows['reachability', budget, method, largest]
            long_path_rows.append([budget, NAMES[method], f"{row['reachable_correct']}/{row['reachable_queries']}",
                                   f"{row['unreachable_correct']}/{row['unreachable_queries']}",
                                   f"{1000 * row['median_seconds']:.2f}"])
    storage_rows = []
    for task, design in protocol['tasks'].items():
        for budget in design['budgets']:
            name = design['unrolled']
            total = totals[task, budget, name]
            row = rows[task, budget, name, largest]
            storage_rows.append([TASKS[task], row['stages'], f"{budget:,}", f"{row['stored']:,}",
                                 f"{total['different']:,}/{total['queries']:,}"])
    invalid = sum(r['invalid'] for r in rows.values())
    all_predictions = sum(r['queries'] for r in rows.values())
    sections.extend(["## What accounts for the differences?",
        "**Pointer chasing.** Every block composes the successor table with itself. Each stage selects whether to advance the source using one bit of the requested hop count. Two and four stages explicitly clip the request to 3 and 15 hops. Their out-of-range accuracy measures this design choice. At the same budget the six-stage untied model uses narrower address matrices than its tied counterpart. All variants keep a 64 × 64 successor table and a 64-component query state during inference.",
        "**Depth with the same local weights.** One pass carries the source flag across one edge. Reusing that block for 63 passes lets the flag reach the end of any simple directed path on 64 vertices. The table counts changes on the same queries, across all ten lengths. Width, weights and input-edge mask are identical within each row.",
        table(["Parameters", "Correct after 1 pass", "Correct after 63 passes", "Errors corrected", "New errors"], depth_rows),
        f"**Reachability at {largest} hops.** Six global stages matched the 63-stage local model on both reachable and unreachable queries. Each global stage doubles the covered path length by composing the reachability relation with itself. Four global stages cover 16 edges and six cover all simple paths on 64 vertices. The table gives median inference time for the same {n} queries.",
        table(["Parameters", "Model", "Reachable correct", "Unreachable correct", f"ms for {n} queries"], long_path_rows),
        "The global model maintains 4,096 changing relation entries per graph. The local model maintains 64 source flags. Both also read the same 64 × 64 adjacency matrix and allocate attention intermediates. Matching parameter counts therefore leaves a substantial difference in working state and in the connections available to each stage.",
        "**Storage at identical depth and width.** Giving every stage its own copy of the tied block repeats the same computation. The table counts changes in predicted labels when replacing shared tensors with independently stored copies.",
        table(["Task", "Stages", "Shared parameters", "Independently stored parameters", "Changed labels"], storage_rows),
        "These results cover analytically specified models and synthetic graphs of fixed size. The six-stage matched-budget alternatives show how width and the choice of operation can affect accuracy at a given depth. Finding the best construction at each depth and learning these algorithms from data are separate questions.",
        f"**Parameter counts and numerics.** Counts cover every stored `nn.Parameter` entry, including zeros, with reused weights counted once. Pointer projections use stored dense Fourier address matrices, with fixed one-hot IDs, identity value/output maps, hop-bit extraction and ReLU gating. Reachability heads partition vertex IDs and store scalar Q/K/V/O coefficients shared across vertices and relation coordinates. Masks, identity edges, fixed operations and working memory are excluded from the parameter budget. The scorer recorded {invalid:,} invalid outputs among {all_predictions:,} model predictions. Any nonfinite final logit invalidates that query. The raw results include every parameter and registered buffer in the tensor ledgers.",
        "## Evidence and reproduction",
        f"{link('Frozen study protocol', HERE/'protocol.json')} · {link('Generator and label checks', HERE/'corpus.py')} · {link('Exact graphs and labels', HERE/'cases.npz')} · {link('Per-array hashes', HERE/'cases-manifest.json')}",
        f"{link('Model details and run instructions', HERE/'README.md')} · {link('Pointer model', HERE/'pointer.py')} · {link('Reachability models', HERE/'reachability.py')} · {link('GPU evaluation', HERE/'experiment.py')} · {link('Report generator', HERE/'report.py')}",
        f"{link('Raw conditions, repeat timings and tensor ledgers', raw/'graph-depth-results.json')} · {link('Per-query predictions', raw/'graph-depth-predictions.npz')} · {link('GPU runtime and output hashes', raw/'graph-depth-runtime.json')}",
        f"[Independent audit](audit.json) · [Paired comparison counts](comparisons.csv) · [All accuracy and timing measurements](sizes.csv) · {link('Audit code', HERE/'audit.py')}",
        "With NumPy, PyTorch and Matplotlib installed, run from the repository root. The first command evaluates the development split on CPU. The second reruns the frozen test on CUDA. The last two independently audit the collected run and regenerate the report, CSV and figure.",
        "```sh\npython graph_depth/experiment.py dev\npython graph_depth/experiment.py gpu --output audit-output/graph-depth-gpu\n"
        f"python graph_depth/audit.py kaggle/runs/graph-depth/v{receipt['version']} results/graph-depth/audit.json results/graph-depth/comparisons.csv\n"
        f"python graph_depth/report.py kaggle/runs/graph-depth/v{receipt['version']} results/graph-depth\n```",
    ])
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", type=Path, help="Collected version directory containing verification.json")
    parser.add_argument("output", type=Path, help="Directory for report.md, sizes.csv and accuracy.png")
    args = parser.parse_args()
    receipt, results, protocol = read_run(args.version)
    rows = validated_rows(args.version, protocol, results)
    markdown = report(rows, protocol, receipt, results, args.version, args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    plot(rows, protocol, args.output / "accuracy.png")
    write_sizes(args.output / "sizes.csv", rows, protocol, results)
    (args.output / "report.md").write_text(markdown, encoding="utf-8")
    print(f"Rendered {len(rows)} verified graph-depth measurements in {args.output}")


if __name__ == "__main__":
    main()
