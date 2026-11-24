"""Summarize the three Kaggle length-*.json outputs into Markdown, CSV and PNG."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


TASKS = {"graph": ("Graph search", "Nodes"), "parity": ("Parity", "Bits"),
         "chains": ("Reasoning chains", "Operations")}
GREEN, ORANGE = "#12675f", "#b5461b"


def aggregate(rows):
    rows = list(rows)
    return (sum(row["correct"] for row in rows), sum(row["count"] for row in rows),
            math.fsum(row["inference_seconds"] for row in rows))


def accuracy(correct, count):
    return f"{100 * correct / count:.2f}%" if count else "—"


def read_results(source):
    directory = source if source.is_dir() else source.parent
    reports = {}
    for task in TASKS:
        file = directory / f"length-{task}.json"
        report = json.loads(file.read_text(encoding="utf-8"))
        if task != report["task"] or not report["rows"]:
            raise ValueError(f"Missing {task} results in {file}")
        rows = sorted(report["rows"], key=lambda row: row["size"])
        if len({row["size"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate size in {file}")
        for row in rows:
            if not (0 <= row["correct"] <= row["count"] and row["count"] > 0 and
                    math.isfinite(row["inference_seconds"]) and row["inference_seconds"] >= 0):
                raise ValueError(f"Invalid aggregate at size {row['size']} in {file}")
            if row["range"] not in ("within", "beyond"):
                raise ValueError(f"Unknown range {row['range']} in {file}")
            if task == "graph":
                parts = aggregate(row["by_distance"].values())
                if parts[:2] != (row["correct"], row["count"]) or not math.isclose(
                        parts[2], row["inference_seconds"], rel_tol=1e-8, abs_tol=1e-9):
                    raise ValueError(f"Graph distance totals disagree at {row['size']} nodes")
        reports[task] = {**report, "rows": rows}
    return reports


def csv_rows(reports):
    for task, report in reports.items():
        for row in report["rows"]:
            values = {
                "task": task, "size_unit": TASKS[task][1].lower(), "size": row["size"],
                "range": row["range"], "correct": row["correct"], "count": row["count"],
                "accuracy": row["correct"]/row["count"],
                "inference_seconds_R": row["inference_seconds"],
                "correct_2R": row["correct_at_loop_multipliers"]["2"],
                "correct_4R": row["correct_at_loop_multipliers"]["4"],
                "unchanged_R_2R_4R": row["unchanged_at_all_checkpoints"],
                "tokens_min": row["token_range"][0], "tokens_max": row["token_range"][1],
                "loops_R_min": row["base_loop_range"][0], "loops_R_max": row["base_loop_range"][1],
                "batches": row["batches"], "model_sha256": report["model_sha256"],
                "corpus_sha256": row["corpus"]["sha256"],
            }
            for group in ("reachable", "unreachable"):
                bins = [v for k, v in row.get("by_distance", {}).items()
                        if (k == "unreachable") == (group == "unreachable")]
                correct, count, seconds = aggregate(bins)
                for field, value in (("correct", correct), ("count", count), ("seconds_R", seconds)):
                    values[f"graph_{group}_{field}"] = value if task == "graph" else ""
            yield values


def plot(reports, target):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.25), layout="constrained")
    for ax, (task, report) in zip(axes, reports.items()):
        rows = report["rows"]
        x = list(range(len(rows)))
        times = [r["inference_seconds"] for r in rows]
        ax.plot(x, times, color="#abb6bd", linewidth=1.5, zorder=1)
        for label, color, marker in [("within", GREEN, "o"), ("beyond", ORANGE, "D")]:
            selected = [i for i,r in enumerate(rows) if r["range"] == label]
            ax.scatter(selected, [times[i] for i in selected], s=34, marker=marker,
                       color=color, zorder=2)
        ax.set_title(TASKS[task][0], loc="left", fontweight="bold", fontsize=12)
        ax.set_xticks(x, [str(r["size"]) for r in rows], rotation=42 if len(rows)>8 else 0)
        ax.set_xlabel(TASKS[task][1])
        ax.set_ylabel("Inference time (s)")
        ax.set_ylim(bottom=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dce1e4", linewidth=0.7)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Inference time per size at the base loop budget", fontsize=15, fontweight="bold")
    handles = [Line2D([], [], marker=m, linestyle="", color=c, label=s)
               for m,c,s in [("o",GREEN,"Within checked range"),("D",ORANGE,"Beyond checked range")]]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def result_line(label, stats):
    correct, count, seconds = stats
    return f"| {label} | {correct:,}/{count:,} | {accuracy(correct, count)} | {seconds:.4f} |"


def markdown(reports):
    all_rows = [r for report in reports.values() for r in report["rows"]]
    correct, count, total_time = aggregate(all_rows)
    device = ", ".join(dict.fromkeys(r["gpu"] for r in reports.values()))
    table = ["| Group | Correct / cases | Accuracy | R time (s) |",
             "|---|---:|---:|---:|"]
    lines = ["# Length generalization on Kaggle", "",
             f"The three frozen transformers answered **{correct:,} of {count:,} held-out cases** "
             f"correctly ({accuracy(correct, count)}). Their synchronized inference time at the base "
             f"loop budget R totaled **{total_time:.4f} seconds** on {device}.",
             "", "## Performance inside and beyond the checked range", "", *table]
    for task, report in reports.items():
        for group in ("within","beyond"):
            lines.append(result_line(f"{TASKS[task][0]}, {group}", aggregate(
                r for r in report["rows"] if r["range"] == group)))
    for group in ("within","beyond"):
        lines.append(result_line(f"All tasks, {group}", aggregate(r for r in all_rows if r["range"]==group)))
    extra_correct = [sum(r["correct_at_loop_multipliers"][str(m)] for r in all_rows) for m in (2,4)]
    unchanged = sum(r["unchanged_at_all_checkpoints"] for r in all_rows)
    size_counts = []
    for task, report in reports.items():
        counts = sorted({r["count"] for r in report["rows"]})
        label = str(counts[0]) if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
        size_counts.append(f"{label} for {TASKS[task][0].lower()}")
    lines += ["", f"At 2R and 4R the models answered {extra_correct[0]:,}/{count:,} and "
              f"{extra_correct[1]:,}/{count:,} correctly. Predictions agreed across all three "
              f"checkpoints on {unchanged:,}/{count:,} cases.", "",
              "## Runtime by size", "", "![Measured inference seconds by task and size](runtime.png)", "",
              f"Each size contains {', '.join(size_counts)}. Each dot represents one timed pass over "
              "that size’s held-out cases. Sizes are spaced "
              "evenly as categories. Each panel has its own time axis.", "",
              "| Task | Size | Range | Correct / cases | Accuracy | R time (s) |",
              "|---|---:|---|---:|---:|---:|"]
    for task, report in reports.items():
        for r in report["rows"]:
            lines.append(f"| {TASKS[task][0]} | {r['size']} {TASKS[task][1].lower()} | "
                         f"{r['range']} | {r['correct']}/{r['count']} | "
                         f"{accuracy(r['correct'],r['count'])} | {r['inference_seconds']:.4f} |")
    lines += ["", "## Graph reachability and shortest-path length", "", *table]
    graph = reports["graph"]["rows"]
    for group in ("within","beyond"):
        for unreachable in (False, True):
            parts = [r for row in graph if row["range"]==group for hop,r in row["by_distance"].items()
                     if (hop=="unreachable")==unreachable]
            lines.append(result_line(f"{group.title()}, {'unreachable' if unreachable else 'reachable'}",
                                     aggregate(parts)))
    hops = defaultdict(list)
    for row in graph:
        for hop,stats in row["by_distance"].items():
            hops[hop].append(stats)
    lines += ["", "Hop counts refer to the shortest directed path from the source to the target.", "",
              "| Shortest path | Correct / cases | Accuracy | R time (s) |", table[1]]
    for hop in sorted(hops, key=lambda h: math.inf if h=="unreachable" else int(h)):
        label = "Unreachable" if hop=="unreachable" else f"{hop} {'hop' if hop=='1' else 'hops'}"
        lines.append(result_line(label, aggregate(hops[hop])))
    lines += ["", "## Models", "",
              "| Task | Dense parameters | What the tied block computes |",
              "|---|---:|---|",
              f"| Graph search | {reports['graph']['parameters']:,} | Passes a decreasing value along each directed edge. The largest value at a node represents its shortest distance from the source. Zero represents unreachable. |",
              f"| Parity | {reports['parity']['parameters']:,} | Combines pairs of partial parities by XOR and doubles the reach of the next pass. The fixed positional table supports up to 4,095 input bits. |",
              f"| Reasoning chains | {reports['chains']['parameters']:,} | Looks up the variable on the right of each shuffled assignment and applies the signed integer update. One more dependency resolves on each pass. |",
              "", "The report counts all embedding, attention, MLP and readout entries. The pointer-doubling parity design has 68 entries in its recurrent block. These analytic builds ran in FP64.",
              "", "## Inputs and timing", "",
              "This sweep uses newly generated inputs. The checked and extended length ranges follow "
              "the authors’ graph-search, parity and reasoning-chain experiments. "
              "[LoopedWSP study](https://mbzuai-oryx.github.io/LoopedWSP/) · "
              "[Released task interfaces](https://github.com/mbzuai-oryx/LoopedWSP/tree/main/weights)",
              "",
              "The graph generator shuffles relabeled directed graphs and grades the query with "
              "breadth-first search. Parity cases balance even and odd bit counts. Reasoning chains "
              "shuffle one base assignment and signed integer updates, then query the terminal variable. "
              "The evaluator uses distinct PCG64 streams for development and evaluation, with root "
              "seeds 41 and 42, and excludes exact development duplicates. The accompanying CSV "
              "records the model and corpus hashes for each size. [Per-size measurements](results.csv)",
              "",
              "R denotes each model’s prescribed loop budget. The timer synchronizes CUDA and includes "
              "batch stacking, device transfers, attention preparation, transformer steps and readout. "
              "Each size’s time sums its batch measurements. Graph batches group by shortest-path "
              "category and token count. The evaluator skips repeated iterations after observing an "
              "exact full-state fixed point. Corpus generation, model loading and the extra 2R and 4R checks "
              "fall outside the reported R time.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Kaggle output directory or one of its length-*.json files")
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if not args.source.exists():
        parser.error(f"Missing input path {args.source}")
    reports = read_results(args.source)
    args.destination.mkdir(parents=True, exist_ok=True)
    records = list(csv_rows(reports))
    with (args.destination/"results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    plot(reports, args.destination/"runtime.png")
    (args.destination/"report.md").write_text(markdown(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
