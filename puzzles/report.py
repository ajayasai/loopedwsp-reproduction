"""Render per-size sequence verification and elapsed times from a downloaded Kaggle run."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TASKS = {"hanoi": ("Tower of Hanoi", "Disks", 20), "checkers": ("Checker Jumping", "Checkers per side", 40)}
MEASUREMENTS = ["decode_seconds", "replay_seconds", "decode_and_stability_seconds",
                "wall_seconds_including_stability_and_replay"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(source):
    result = {}
    for key, (_, _, limit) in TASKS.items():
        doc = read(source/f"puzzle-{key}.json")
        rows = sorted(doc["size_rows"], key=lambda r: r["size"])
        if doc["task"] != key or [r["size"] for r in rows] != list(range(1,limit+1)):
            raise ValueError(f"Expected exactly one {key} result for each size through {limit}")
        for r in rows:
            if not (0 <= r["legal_moves"] <= r["moves"] and
                    all(math.isfinite(r[k]) and r[k] >= 0 for k in MEASUREMENTS)):
                raise ValueError(f"Invalid move count or elapsed time for {key} size {r['size']}")
            if r["optimal"] and not (r["valid"] and r["goal"] and r["moves"] == r["optimal_moves"]):
                raise ValueError(f"Inconsistent optimality flag for {key} size {r['size']}")
        result[key] = {**doc, "size_rows": rows}
    return result


def export_csv(docs, target):
    fields = ["task", "size", "range", "device", "dtype", "parameters", "model_sha256",
              "moves", "optimal_moves", "legal_moves", "all_moves_legal", "goal", "valid",
              "optimal", "ended_with_eos", "slides", "jumps", *MEASUREMENTS, "decoding_steps",
              "positions_evaluated_in_windows", "predictions_changed_at_2R_or_4R",
              "move_ids_sha256", "first_error"]
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for task, doc in docs.items():
            for row in doc["size_rows"]:
                # Python's float string conversion round-trips the input timings.
                out = {k: row.get(k, "") for k in fields}
                out.update(task=task, device=doc["device"], dtype=doc["dtype"],
                           parameters=doc["model"]["parameters"], model_sha256=doc["model"]["model_sha256"],
                           first_error=json.dumps(row["first_error"]) if row["first_error"] else "")
                writer.writerow(out)


def figure(docs, target):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained")
    for ax, (task, doc) in zip(axes, docs.items()):
        rows = doc["size_rows"]
        xs = [r["size"] for r in rows]
        for k, label, color, style in (("decode_seconds","GPU decode","#235a8b","-"),
                                       ("replay_seconds","CPU replay","#a44121","--")):
            ax.plot(xs, [r[k] for r in rows], label=label, color=color, linestyle=style,
                    marker="o" if k=="decode_seconds" else "s", markersize=3, linewidth=1.4)
        if all(row[k] > 0 for row in rows for k in ("decode_seconds","replay_seconds")):
            ax.set_yscale("log")
        ax.set_xlabel(TASKS[task][1])
        ax.set_ylabel("Elapsed time (seconds)")
        ax.set_title(TASKS[task][0], loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="y", color="#dce0e3", linewidth=.7)
        ax.spines[["top","right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Measured time to emit and verify each trajectory", fontweight="bold", fontsize=14)
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def markdown(docs, source, dest, receipt):
    def link(name, path):
        return f"[{name}](<{Path(os.path.relpath(path, dest)).as_posix()}>)"

    rows = [r for doc in docs.values() for r in doc["size_rows"]]
    solved = sum(r["ended_with_eos"] and r["valid"] and r["optimal"] for r in rows)
    legal = sum(r["legal_moves"] for r in rows)
    moves = sum(r["moves"] for r in rows)
    changed = sum(r["predictions_changed_at_2R_or_4R"] for r in rows)
    dec = math.fsum(r["decode_seconds"] for r in rows)
    replay = math.fsum(r["replay_seconds"] for r in rows)
    elapsed = math.fsum(r["wall_seconds_including_stability_and_replay"] for r in rows)
    devices = ", ".join(dict.fromkeys(doc["device"] for doc in docs.values()))
    held = [r for r in rows if r["range"] == "held out"]
    held_opt = sum(r["ended_with_eos"] and r["optimal"] for r in held)
    lines = ["# Tower of Hanoi and Checker Jumping", "",
             f"**{solved}/{len(rows)} trajectories** ended with the end-of-sequence token, reached the goal legally "
             f"and used the minimum number of moves. The held-out sizes accounted for "
             f"{held_opt}/{len(held)} optimal trajectories. Independent simulators replayed "
             f"{legal:,} legal moves out of {moves:,} emitted on {devices}. "
             f"[Private Kaggle run, version {receipt['version']}]({receipt['url']})", "",
             "## Model and timing summary", "",
             "| Puzzle | Dense parameters | Nonzero | Base loops | GPU decode (s) | CPU replay (s) | Wall with stability and replay (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, doc in docs.items():
        m = doc["model"]
        times = [math.fsum(r[k] for r in doc["size_rows"]) for k in
                 ("decode_seconds","replay_seconds","wall_seconds_including_stability_and_replay")]
        lines.append(f"| {TASKS[key][0]} | {m['parameters']:,} | {m['nonzero_parameters']:,} | "
                     f"{m['rounds']} | {times[0]:.6f} | {times[1]:.6f} | {times[2]:.6f} |")
    lines += ["", f"Across all sizes, FP64 decoding took {dec:.6f} s and independent CPU replay took "
              f"{replay:.6f} s. The sum of measured per-size wall times was {elapsed:.6f} s, including "
              "the additional prediction checks at twice and four times the base loop count. Across "
              f"emitted positions and EOS, {changed:,} predictions changed at either checkpoint. "
              "The wall-time measurement starts with each size’s decode and ends after replay. "
              "Kaggle startup, model setup, development parity checks and writing the archives are outside it.", "",
              "![GPU decoding and CPU replay by size](runtime.png)", "",
              "Each point is a single measured trajectory. The per-size tables show seconds to six "
              "decimal places. The CSV retains the precision reported by the evaluator. "
              "[Download the size-level CSV](sizes.csv)", "", "## How moves were decoded", "",
              "Each model receives one size prompt. A move position supplies a role and numeric features "
              "of its index. A check on the frozen query and key tensors establishes that their routing "
              "coordinates remain constant through the loops. The unique prompt wins causal attention "
              "by a margin large enough that all other FP64 softmax weights round to zero. Consequently, "
              "several output positions can be evaluated together using the prompt and their individual "
              "indices. The GPU decoder evaluates windows of up to 4,096 positions and retains moves up "
              "to the first predicted EOS. It can evaluate additional positions in the final window.", "",
              "The development check decoded the full causal prefix one move at a time and compared its "
              "trajectory with the windowed decoder. The study’s transformers use move history during "
              "decoding. The timings here measure these independent position-based models. "
              "[LoopedWSP study, Appendix E](https://mbzuai-oryx.github.io/LoopedWSP/)", "",
              "For Checker Jumping, the size embedding explicitly supplies the analytic quantity "
              "N(N+2) as well as N. The network uses the supplied length to locate its forward and "
              "reflected movement phases and predict EOS. The checker simulator reconstructs the board, "
              "checks every slide and jump, and compares the final board and emitted move count with "
              "the optimum. Each red checker must cross each blue checker once, requiring N² jumps. "
              "The two groups travel a combined 2N(N+1) slots. After the jumps, 2N single-slot "
              "slides remain, giving N(N+2) total moves. The Hanoi simulator checks the disk and "
              "both pegs on each move. Moving the largest disk requires moving the smaller tower "
              "out of the way and then onto it, giving the minimum 2ᴺ−1. "
              "[Puzzle definitions and simulator rules](https://arxiv.org/html/2506.06941v1#A1.SS1)"]
    for key, doc in docs.items():
        name, unit, _ = TASKS[key]
        lines += ["", f"## {name} by size", "",
                  f"| {unit} | Range | Emitted / optimum | Legal moves | Goal | Optimal | GPU decode (s) | CPU replay (s) |",
                  "|---:|---|---:|---:|---|---|---:|---:|"]
        for r in doc["size_rows"]:
            lines.append(f"| {r['size']} | {r['range']} | {r['moves']:,} / {r['optimal_moves']:,} | "
                         f"{r['legal_moves']:,} | {'Yes' if r['goal'] else 'No'} | "
                         f"{'Yes' if r['optimal'] else 'No'} | {r['decode_seconds']:.6f} | {r['replay_seconds']:.6f} |")
    lines += ["","## Source artifacts",""]
    for key, doc in docs.items():
        lines.append(f"- **{TASKS[key][0]}** — " + " · ".join([
            link("Kaggle results", source/f"puzzle-{key}.json"),
            link("Emitted move IDs", source/f"emitted-{key}.npz"),
            link("Move alphabet", ROOT/"puzzles/models"/key/"network.json"),
            link("Frozen weights", ROOT/"puzzles/models"/key/"weights.npz")]))
    links = [link("Evaluation harness", ROOT/"puzzles/experiment.py"),
             link("Decoder", ROOT/"puzzles/frame.py"), link("Independent verifiers", ROOT/"puzzles/verify.py"),
             link("Kaggle submission receipt", source.parent/"submission.json")]
    if (source.parent/"verification.json").is_file():
        links.append(link("Remote verification record", source.parent/"verification.json"))
    if (dest/"replay-audit.json").is_file():
        links.append(link("Second replay of downloaded moves", dest/"replay-audit.json"))
    lines += ["", " · ".join(links), ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="Downloaded Kaggle output folder")
    p.add_argument("destination", nargs="?", type=Path, default=ROOT/"results/puzzles")
    args = p.parse_args()
    source = args.source.resolve()
    docs = records(source)
    receipt = read(source.parent/"submission.json")
    args.destination.mkdir(parents=True, exist_ok=True)
    export_csv(docs, args.destination/"sizes.csv")
    figure(docs, args.destination/"runtime.png")
    (args.destination/"report.md").write_text(markdown(docs,source,args.destination.resolve(),receipt),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
