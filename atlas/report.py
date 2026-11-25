"""Regrade the atlas Kaggle archive and render its measurements and worked examples."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from audit import evaluate


HERE = Path(__file__).resolve().parent
TASKS = {"rings": "Tree-ring alignment", "loom": "Loom drawdown", "braid": "Fox 3-colorings"}
RING_CHECKS = {"scaled_ring_extract": "Scaled extract", "missing_years_tie": "Tie with missing years",
               "weak_ring_winner": "Weak unique match", "unmeasured_ring_intervals": "Too few intervals"}
LOOM_CHECKS = {"loom_plain_weave": "Plain weave", "loom_two_two_twill": "Two-over, two-under twill",
               "loom_compound_and_idle": "Combined and idle treadles"}
BRAID_CHECKS = {"three_component_unlink": "Three separate circles", "closed_unknot": "Unknot",
                "stabilized_trefoil": "Trefoil", "figure_eight_knot": "Figure-eight knot",
                "grouped_crossings": "Grouped generators", "alternating_generators": "Alternating generators"}


def ratio(correct, total):
    return f"{correct:,}/{total:,}"


def table(headings, rows):
    return "\n".join("| " + " | ".join(map(str, cells)) + " |"
                     for cells in (headings, ["---"] * len(headings), *rows))


def timings_csv(path, rows, protocol):
    stats = ["parameters", "records", "exact_records", "labels", "correct_labels", "nonfinite", "median_seconds"]
    repeats = [f"seconds_{i+1}" for i in range(protocol["repeats"])]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["task", "size", "unit", *stats, *repeats])
        for row in rows:
            name = row["task"]
            writer.writerow([name, row["size"], protocol["tasks"][name]["size_unit"],
                             *[row[k] for k in stats], *row["seconds"]])


def drawdown_figure(path, checks, predictions):
    """Draw a stroke along the visible yarn in each predicted crossing."""
    warp, weft = "#274f71", "#e4ab6c"
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), gridspec_kw={"width_ratios": [4, 4, 6]})
    fig.subplots_adjust(left=.075, right=.99, top=.80, bottom=.25, wspace=.55)
    for ax, (name, title) in zip(axes, LOOM_CHECKS.items(), strict=True):
        record = checks[name]["input"]
        shafts, picks = record["threading"], record["treadling"]
        cloth = np.asarray(predictions[name]).reshape(len(picks), len(shafts))
        ax.imshow(cloth, cmap=ListedColormap([weft, warp]), vmin=0, vmax=1, interpolation="nearest")
        for (row, col), visible in np.ndenumerate(cloth):
            dx, dy = (0, .37) if visible else (.37, 0)
            ax.plot([col-dx, col+dx], [row-dy, row+dy], color="white", lw=1.8, alpha=.85)
        ax.set_xticks(range(len(shafts)), shafts)
        ax.set_yticks(range(len(picks)), ["+".join(map(str, pick)) or "idle" for pick in picks])
        ax.set_xticks(np.arange(.5, len(shafts)-.5), minor=True)
        ax.set_yticks(np.arange(.5, len(picks)-.5), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="both", length=0)
        ax.set_xlabel("Shaft by warp end", labelpad=6, fontsize=10)
        ax.set_title(title, fontsize=12, pad=8)
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("Treadles by pick", labelpad=6, fontsize=10)
    fig.suptitle("Which thread lies on top?", fontsize=16)
    fig.legend(handles=[Patch(facecolor=warp, label="Warp on top (vertical yarn)"),
                        Patch(facecolor=weft, label="Weft on top (horizontal yarn)")],
               loc="lower center", bbox_to_anchor=(.5, .03), frameon=False, ncols=2, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=.13)
    plt.close(fig)


def ring_table(checks, predictions):
    rows = []
    for name, description in RING_CHECKS.items():
        record = checks[name]["input"]
        sample, ref = record["sample"], record["reference"]
        offsets = [i for i, value in enumerate(predictions[name]) if value == 2]
        skipped = [i for i, value in enumerate(predictions[name]) if value == 0]
        scores = []
        for start in offsets:
            signs = [(a < b) == (c < d) for a,b,c,d in zip(sample[:-1], sample[1:],
                     ref[start:start+len(sample)-1], ref[start+1:start+len(sample)], strict=True)
                     if None not in (a,b,c,d) and a != b and c != d]
            scores.append(ratio(sum(signs), len(signs)))
        rows.append([description, ", ".join(map(str, offsets)) or "none",
                     ", ".join(scores) or "—", ", ".join(map(str, skipped)) or "none"])
    return table(["Check", "Best offsets (zero-based)", "Agreements / usable intervals at each best offset",
                  "Insufficient offsets"], rows)


def braid_table(checks, predictions):
    return table(["Closure", "Signed braid word", "GPU colorings"],
                 [[description, f"`{checks[name]['input']['word']}`", predictions[name][0]]
                  for name, description in BRAID_CHECKS.items()])


def render(run, dest):
    audit = evaluate(run)
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    checks = {c["name"]:c for c in json.loads((HERE / "checks.json").read_bytes())}
    with np.load(run / "output/atlas-predictions.npz", allow_pickle=False) as arrays:
        predictions = {name: arrays["check_"+name].tolist() for name in checks}
    dest.mkdir(exist_ok=True, parents=True)
    rows = audit["conditions"]
    timings_csv(dest / "sizes.csv", rows, protocol)
    drawdown_figure(dest / "loom-drawdown.png", checks, predictions)
    counts, exact = audit["records_by_category"], audit["exact_records_by_category"]
    diagnostics = audit["diagnostics"]

    def group(task, category):
        return ratio(exact[task].get(category, 0), counts[task].get(category, 0))

    def summed(rows, num, den):
        return ratio(sum(r[num] for r in rows), sum(r[den] for r in rows))

    device = audit["run"]["device"]
    base = f"../../kaggle/runs/atlas/v{audit['run']['version']}"
    large = [r for r in rows if r["size"] > max(protocol["tasks"][r["task"]]["dev"])]
    worked = audit["worked_checks"]
    summary = table(["Application", "Size", "Stored FP32 coefficients", "Exact records", "Correct labels",
                     "Median GPU time (ms)"],
                    [[TASKS[r["task"]], r["size"], r["parameters"], ratio(r["exact_records"],r["records"]),
                      ratio(r["correct_labels"],r["labels"]), f"{1000*r['median_seconds']:.2f}"] for r in rows])
    sections = [
        "# Tree-ring alignment, loom drawdown and knot coloring",
        f"On a Kaggle {device}, three hand-programmed transformers matched their independent references on "
        f"**{summed(rows, 'exact_records', 'records')} complete held-out records** and "
        f"**{summed(rows, 'correct_labels', 'labels')} individual labels**. At sizes above the development range, "
        f"{summed(large, 'exact_records', 'records')} complete records matched. The separate worked checks "
        f"passed on {ratio(sum(v['correct'] for v in worked.values()),len(worked))} inputs.",
        "## Accuracy and timing by input size", summary,
        "Size counts rings in the reference series, warp ends and picks per side of a square draft, or signed "
        "crossings in a braid word. One exact record requires every candidate start, every cloth crossing, or "
        "the single coloring count to agree. A longer reference adds possible alignments for the same eight-ring "
        "sample. Doubling a square draft’s side quadruples its output area. A longer braid requires additional "
        "sequential updates to the three-strand colors.",
        f"Each timing is the median of {protocol['repeats']} synchronized forward sweeps over "
        f"{protocol['per_size']['test']} FP32 records on the GPU, in batches of {protocol['batch']} after warm-up. "
        "Encoding, transfers, decoding and scoring occur outside the timer. Stored coefficients include zeros "
        "and count shared weights once. The fixed wiring and intermediate arrays also contribute to the "
        "computation. [All timing repeats](sizes.csv)",
        "## Dendrochronology · matching ring-width changes",
        "An eight-ring sample contributes up to seven one-year changes. A change qualifies when both series "
        "have measured endpoints and both widths changed. For each offset with at least four qualifying "
        "changes, the score is the fraction that agree in direction. The transformer returns all offsets "
        "tied for the best score. The worked cases below use relative offsets measured from the oldest "
        "reference ring.", ring_table(checks,predictions),
        f"The model matched {group('rings','unique')} records with a unique best start, "
        f"{group('rings','tied')} with tied starts, and {group('rings','insufficient')} with insufficient evidence "
        "at every start. The weak worked example has a single winner with only one agreement in seven. "
        "Real calendar dating requires independent material and corroboration.",
        f"In {diagnostics['unique_affine_windows']} held-out records, exactly one complete reference window "
        "reproduces the eight sample widths by multiplying each width by 1 or 2 and then adding a single "
        f"integer from 1 to 29. That offset was among the predicted best starts in "
        f"{ratio(diagnostics['affine_offset_among_best'],diagnostics['unique_affine_windows'])} such records. "
        f"It was the only predicted winner in "
        f"{ratio(diagnostics['affine_offset_sole_best'],diagnostics['unique_affine_windows'])}. "
        "Repeated rise/fall patterns can tie even when the complete width sequence identifies a single window. "
        "[Runnable ring sample](../../atlas/examples/rings.json)",
        "## Textile engineering · drawing the cloth from a loom draft",
        "![Three GPU-predicted loom drawdowns with visible warp and weft at each crossing](loom-drawdown.png)",
        "Across each panel the tick labels identify the shafts carrying the warp ends. Reading down the side "
        "gives the treadles pressed for each pick. In the right panel, pressing treadles 1 and 2 together "
        "raises shafts 1, 2 and 3. The following idle pick exposes the weft across the full width.",
        f"The model matched {group('loom','compound')} drafts containing a simultaneous treadle press and "
        f"{group('loom','simple')} drafts without one. Keeping only the first pressed treadle in each pick "
        f"would change the cloth in {diagnostics['first_treadle_changes_draft']} held-out drafts. The model matched "
        f"{ratio(diagnostics['changed_drafts_exact'],diagnostics['first_treadle_changes_draft'])} of those drafts exactly. "
        "These are binary interlacements on the upper face of an "
        "ideal rising-shed loom. The drawdown gives the order of the threads at a crossing. Color, thickness "
        "and tension determine the finished appearance. [Runnable twill draft](../../atlas/examples/loom.json)",
        "## Knot theory · counting Fox 3-colorings",
        "Each signed generator crosses two adjacent strands. Closing the braid joins each bottom endpoint to "
        "the top endpoint in the same position. A Fox coloring assigns one of three colors to each arc and "
        "satisfies the crossing rule that twice the overstrand color equals the sum of the two understrand "
        "colors modulo three. The model counts all assignments that close, including three constant colorings.",
        braid_table(checks,predictions),
        f"In the held-out set the model matched {group('braid','3')} closures with 3 colorings, "
        f"{group('braid','9')} with 9, and {group('braid','27')} with 27. The trefoil has six nonconstant "
        "colorings. The unknot and figure-eight knot both have just the three constant ones, so this count "
        "can be shared by distinct knots. The model enumerates 27 top-strand assignments and advances them "
        "through each crossing using the same weights. The independent audit derives one modular equation "
        "per crossing and solves the closure system by row reduction. [Runnable trefoil](../../atlas/examples/braid.json)",
        "## Run and verify",
        f"[Model guide and commands](../../atlas/README.md) · [Frozen protocol](../../atlas/protocol.json) · "
        f"[Named check inputs and labels](../../atlas/checks.json) · [Independent audit](audit.json) · "
        f"[Archived GPU predictions]({base}/output/atlas-predictions.npz) · [Submission receipt]({base}/submission.json) · "
        f"[GPU verification]({base}/verification.json) · [Private Kaggle run]({audit['run']['url']})",
        "```sh\npython atlas/experiment.py predict rings --example\npython atlas/experiment.py predict loom --example\n"
        "python atlas/experiment.py predict braid --example\n"
        f"python atlas/audit.py kaggle/runs/atlas/{run.name} audit-output/atlas-audit.json\n"
        f"python atlas/report.py kaggle/runs/atlas/{run.name} results/atlas\n```"]
    (dest / "report.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(rows)} verified GPU settings and {len(worked)} named checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.run, args.output)
