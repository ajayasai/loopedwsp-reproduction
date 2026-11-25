"""Render field-study examples and measurements from an independently regraded run."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit import evaluate

HERE = Path(__file__).resolve().parent
NAMES = {"basin": "Terrain depression filling", "transit": "Transit-template search", "bowling": "Ten-pin scorekeeping"}
PERIODS = (3, 4, 5, 7, 9)


def table(headers, rows):
    lines = [headers, ["---"] * len(headers), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in lines)


def selected_template(code):
    if code == 0:
        return None
    index = code - 1
    for period in PERIODS:
        if index < period:
            return period, index
        index -= period
    raise ValueError(f"Unknown template class {code}")


def alias_example(cases, preds):
    for i, (row, out) in enumerate(zip(cases["test/transit/24"], preds["transit_24"], strict=True)):
        trial = selected_template(int(out[0]))
        metadata = row["injected"]
        if trial is None or metadata is None:
            continue
        actual = metadata["period"], metadata["phase"]
        if actual == trial:
            continue
        rec = row["input"]
        observed = [j for j, value in enumerate(rec["flux"]) if value is not None]
        hits = [j for j in observed if j % actual[0] == actual[1]]
        if hits == [j for j in observed if j % trial[0] == trial[1]]:
            evidence = sum(rec["baseline"] - rec["flux"][j] - 4 for j in hits)
            return (f"In test record `transit_24[{i}]`, the injected period {actual[0]}, phase {actual[1]}, "
                    f"and the selected period {trial[0]}, phase {trial[1]}, both predict dips at precisely the "
                    f"observed indices {', '.join(map(str,hits))}. Both score {evidence}. The tie rule selects the shorter period.")
    return ""


def examples(cases, preds, output):
    # Deterministic selection for illustration. The score tables use every record.
    basin_rows = cases["test/basin/8"]
    basin_index = max(range(len(basin_rows)), key=lambda i: len(set(basin_rows[i]["gold"])))
    heights = np.asarray(basin_rows[basin_index]["input"]["heights"])
    depth = preds["basin_8"][basin_index].reshape(heights.shape)
    curves = cases["test/transit/48"]
    curve_index = next(i for i, row in enumerate(curves)
                       if row["injected"] and selected_template(int(preds["transit_48"][i, 0]))
                       == (row["injected"]["period"], row["injected"]["phase"]) and None in row["input"]["flux"])
    curve = curves[curve_index]["input"]
    period, phase = selected_template(int(preds["transit_48"][curve_index, 0]))
    flux = np.asarray([np.nan if v is None else v for v in curve["flux"]])
    t = np.arange(len(flux))
    hits = t % period == phase

    fig = plt.figure(figsize=(8.8, 7.2), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    panels = [(heights, "Ground elevation (input)", "cividis"), (depth, "Retained depth (prediction)", "Blues")]
    for col, (matrix, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=0, vmax=9)
        for y, x in np.ndindex(matrix.shape):
            v = int(matrix[y,x])
            rgba = im.cmap(im.norm(v))
            lum = .299*rgba[0] + .587*rgba[1] + .114*rgba[2]
            ax.text(x, y, str(v), ha="center", va="center", fontsize=8, color="black" if lum > .53 else "white")
        ax.set_title(title, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = fig.add_subplot(gs[1, :])
    ax.set_title(f"Light curve with predicted dips every {period} cadences, phase {phase}", loc="left")
    ax.axhline(curve["baseline"], color="0.65", linewidth=1, label="Known baseline")
    ax.plot(t, flux, color="#436877", lw=.9, marker=".", ms=5, label="Observed brightness")
    ax.scatter(t[hits], flux[hits], s=50, facecolors="none", edgecolors="#bd5820", lw=1.8, zorder=3,
               label="Observed at predicted phase")
    if np.isnan(flux).any():
        ax.plot(t[np.isnan(flux)], np.full(np.isnan(flux).sum(), .02), "|", transform=ax.get_xaxis_transform(),
                ms=7, color="0.45", label="Missing observation")
    ax.set_xlabel("Cadence index")
    ax.set_ylabel("Synthetic flux units")
    ax.legend(ncols=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.25), frameon=False)
    fig.savefig(output / "field-examples.png", dpi=150)
    plt.close(fig)
    bowl_rows = cases["test/bowling/5"]
    j = next(i for i, row in enumerate(bowl_rows) if 301 in row["gold"] and any(x < 301 for x in row["gold"]))
    card = table(["Frame", "Pins on each ball", "Cumulative score"], [
        [i+1, ", ".join(map(str, frame)), "Pending" if score == 301 else int(score)]
        for i, (frame,score) in enumerate(zip(bowl_rows[j]["input"]["frames"], preds["bowling_5"][j], strict=True))])
    return (f"The 8×8 raster is test record `basin_8[{basin_index}]`. Its predicted depths sum to "
            f"**{int(depth.sum())} cell-height units**. The light curve is `transit_48[{curve_index}]`. "
            f"The recovered period and phase match its injected values. The charts illustrate individual records.",
            f"The GPU returned this scorecard for `bowling_5[{j}]`. Each pending entry awaits another legal delivery.", card)


def render(run, output):
    audit = evaluate(run)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))
    rows = audit["conditions"]
    with np.load(run / "output/fieldlab-predictions.npz", allow_pickle=False) as preds:
        figure_note, score_note, card = examples(cases, preds, output)
        alias_note = alias_example(cases, preds)
    with (output / "sizes.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task","size","unit","parameters","records","exact_records","labels","correct_labels","nonfinite",
                  "median_seconds","seconds_1","seconds_2","seconds_3"]
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            record = {key: r[key] for key in fields if key in r}
            record["unit"] = protocol["tasks"][r["task"]]["size_unit"]
            record.update({f"seconds_{i}": val for i, val in enumerate(r["seconds"], 1)})
            writer.writerow(record)
    results_table = table(["Application","Size","Stored parameters","Exact records","Correct labels","Median ms"], [
        [NAMES[r["task"]], r["size"], f"{r['parameters']:,}", f"{r['exact_records']}/{r['records']}",
         f"{r['correct_labels']:,}/{r['labels']:,}", f"{1000*r['median_seconds']:.2f}"] for r in rows])
    science = audit["science"]
    fields = ["injected","recovered","missed","wrong_template","uninjected","false_alarms"]
    sums = {k: sum(r[k] for r in science) for k in fields}
    injection_table = table(["Cadences","Injected","Period + phase recovered","No candidate","Wrong template",
                             "Uninjected","False alarms"],
                            [[r["size"], *[r[k] for k in fields]] for r in science] + [["All", *[sums[k] for k in fields]]])
    totals = {k: sum(r[k] for r in rows) for k in ("records","exact_records","labels","correct_labels")}
    cats = audit["test_categories"]
    larger = [r for r in rows if r["size"] > max(protocol["tasks"][r["task"]]["dev"])]
    receipt = json.loads((run / "submission.json").read_bytes())
    base = f"../../kaggle/runs/fieldlab/v{receipt['version']}"
    text = [
        "# Hydrology, astronomy and ten-pin scoring",
        f"The frozen models matched their task references on **{totals['exact_records']}/{totals['records']} complete records** "
        f"and {totals['correct_labels']:,}/{totals['labels']:,} individual outputs on a Kaggle Tesla T4. "
        f"At sizes larger than the development range they matched {sum(r['exact_records'] for r in larger)}/{sum(r['records'] for r in larger)} records. "
        "For astronomy, the reference is an explicitly defined template search. The injected-signal recovery table below measures a separate outcome.",
        "## Accuracy and runtime by size", results_table,
        "Size counts cells per side, brightness samples or started bowling frames. One record requires every output to be correct, including pending bowling scores. "
        "Times are the medians of three synchronized forward passes over 64 GPU-resident records in batches of eight after warm-up, excluding encoding, transfers and scoring. "
        "Parameter counts include stored zeros and count shared tensors once. The counts exclude fixed features, geometry and working memory. [All timing repeats](sizes.csv)",
        "## Terrain and telescope examples",
        "![Ground elevations, predicted retained depths and a synthetic light curve](field-examples.png)", figure_note,
        f"With an open perimeter and four-neighbor flow, the basin model matched the complete depth grid on "
        f"{cats['basin']['local_heuristic_wrong_correct']}/{cats['basin']['local_heuristic_wrong']} terrains where using only immediate neighbors gives a different answer. "
        "The target is depression storage under the stated drainage assumptions. The experiment uses integer elevations and unit-area cells.",
        "## Did the search recover the injected transit?", injection_table,
        f"Recovery requires both the injected period and the zero-based phase. A wrong template is a detection at another period or phase. In "
        f"{sum(r['indistinguishable_wrong_template'] for r in science)}/{sums['wrong_template']} wrong-template cases, the selected and injected templates mark exactly the same observed cadences. A false alarm is a detection on a series with no injected periodic signal. "
        "For each series the model and reference receive only the known baseline and measured flux. The experiment uses one-cadence synthetic dips, missing measurements and a bank of five integer trial periods. Its recovery rates describe this generator.",
        alias_note, "## Live bowling scorecard", score_note, card,
        f"Across all sizes, the rule-based reference had {cats['bowling']['pending']} cards with at least one pending cumulative score. "
        f"The model matched all entries on {cats['bowling']['pending_correct']} of them. The test includes {cats['bowling']['complete_ten_frame_games']} completed ten-frame games.",
        "## Run and verify",
        "The repository contains the JSON examples, domain-specific input rules, programmed weights and an independent scorer that recomputes every target. "
        f"[Example commands and methods](../../fieldlab/README.md) · [Frozen protocol](../../fieldlab/protocol.json) · "
        f"[Raw predictions]({base}/output/fieldlab-predictions.npz) · [Independent audit](audit.json) · [Kaggle receipt]({base}/verification.json) · [Private Kaggle run]({receipt['url']})",
        "```sh\npython fieldlab/experiment.py predict bowling --input fieldlab/examples/bowling.json\n"
        f"python fieldlab/audit.py kaggle/runs/fieldlab/v{receipt['version']} audit-output/fieldlab-audit.json\n"
        f"python fieldlab/report.py kaggle/runs/fieldlab/v{receipt['version']} results/fieldlab\n```"]
    (output / "report.md").write_bytes(("\n\n".join(text) + "\n").encode())
    print(f"Rendered {len(rows)} verified GPU settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.version_folder, args.output)
