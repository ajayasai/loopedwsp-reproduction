"""Render the three application results from the independently audited GPU predictions."""

import argparse
import csv
import json
from pathlib import Path
from string import ascii_lowercase, ascii_uppercase, digits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from audit import evaluate


HERE = Path(__file__).resolve().parent
TASKS = {"quake": "Seismic source", "braille": "Braille line", "bridge": "Bridge endgame"}
QUAKE_CASES = {
    "third_station_missing": "Two recorded arrivals",
    "third_station_resolves_source": "Third arrival added",
    "adjacent_station_conflict": "Conflicting adjacent stations",
    "diagonal_first_tick": "Source at a station",
}
BRAILLE_CASES = {
    "room_label": "Room number with a letter suffix",
    "three_readings": "Dots 12 in lowercase, numeric and capital contexts",
    "letters_end_number": "A k–z cell or a capital sign ends a number",
    "long_numeric_span": "Twelve digits followed by a grade-1 letter",
}
BRIDGE_CASES = {
    "low_spade_and_heart_trump": "King of clubs or six of spades",
    "east_chooses_against_ns": "Same position rotated to East",
    "follow_suit_before_ruffing": "Every hand contains a spade",
}
QUAKE_FIGURE = "seismic-locations.png"


def table(header, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in (header, ["---"] * len(header), *rows))


def ratio(correct, count):
    return f"{correct:,}/{count:,}"


def total(rows, correct, count):
    return ratio(sum(row[correct] for row in rows), sum(row[count] for row in rows))


def write_sizes(path, rows, protocol):
    fields = ("parameters", "records", "exact_records", "labels", "correct_labels", "nonfinite", "median_seconds")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["task", "size", "size_unit", *fields,
                         *(f"seconds_{i+1}" for i in range(protocol["repeats"]))])
        for row in rows:
            writer.writerow([row["task"], row["size"], protocol["tasks"][row["task"]]["size_unit"],
                             *(row[name] for name in fields), *row["seconds"]])


def locations(labels):
    return [(i % 7, i // 7, int(label)-1) for i, label in enumerate(labels) if label > 0]


def source_figure(path, cases, worked):
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.19, wspace=.17)
    teal, ochre, gray = "#07585c", "#954524", "#69747c"
    names = ("third_station_missing", "third_station_resolves_source")
    for ax, name in zip(axes, names, strict=True):
        spots = locations(worked[name])
        readings = cases[name]["stations"]
        for x, y, tick in spots:
            ax.scatter(x, y, s=182, marker="o", color=teal, zorder=3)
            ax.text(x, y, str(tick), ha="center", va="center", color="white", fontsize=9.5, zorder=4)
        for x, y, arrival in readings:
            missing = arrival is None
            ax.scatter(x, y, s=165, marker="s", facecolors="white" if missing else ochre,
                       edgecolors=gray if missing else ochre, linewidths=1.5, zorder=3)
            ax.text(x, y, "–" if missing else str(arrival), ha="center", va="center",
                    color=gray if missing else "white", fontsize=9.5, zorder=4)
        count = sum(t is not None for _, _, t in readings)
        noun = "source" if len(spots) == 1 else "sources"
        ax.set_title(f"{count} measured arrivals · {len(spots)} compatible {noun}", fontsize=11, pad=9)
        ax.set(xticks=range(7), yticks=range(7), xlim=(-.45, 6.45), ylim=(-.45, 6.45),
               xlabel="x (grid units)", aspect="equal")
        ax.tick_params(length=0, pad=3, colors="#44545c", labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(color="#dce2e2", linewidth=.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("y (grid units)")
    fig.suptitle("Which sources fit the station arrivals?", fontsize=16, y=.95)
    legend = [
        Line2D([], [], marker="o", linestyle="none", color=teal, label="Source (origin tick)"),
        Line2D([], [], marker="s", linestyle="none", color=ochre, label="Station (arrival tick)"),
        Line2D([], [], marker="s", linestyle="none", markerfacecolor="white", markeredgecolor=gray,
               label="Station (no reading)"),
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .03), ncol=3,
               frameon=False, fontsize=10, handletextpad=.3, columnspacing=1.4)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=.17)
    plt.close(fig)


def named_examples(cases, worked):
    quake = []
    for name, explanation in QUAKE_CASES.items():
        readings = ", ".join(f"({x}, {y}) {t if t is not None else 'missing'}" for x, y, t in cases[name]["stations"])
        answers = ", ".join(f"({x}, {y}) {t}" for x, y, t in locations(worked[name]))
        quake.append([explanation, readings, answers or "No compatible source"])

    braille = []
    characters = ["", " ", *ascii_lowercase, *ascii_uppercase, *digits]
    for name, explanation in BRAILLE_CASES.items():
        dots = " ".join(c or "blank" for c in cases[name]["cells"])
        line = "".join(characters[label] if label >= 0 else "�" for label in worked[name])
        braille.append([explanation, dots, f"`{line}`"])

    bridge = []
    for name, explanation in BRIDGE_CASES.items():
        for deal, label in zip(cases[name]["deals"], worked[name], strict=True):
            ns, mask = divmod(label, 4)
            hand = deal["hands"][deal["leader"]]
            leads = [card for i, card in enumerate(hand) if mask & (1 << i)]
            bridge.append([explanation, deal["leader"], deal["trump"], " / ".join(hand),
                           ns if label >= 0 and mask else "Invalid", " / ".join(leads) if label >= 0 and mask else "Invalid"])
    return (table(["Worked event", "Stations (x, y) and arrival tick", "GPU sources (x, y) and origin tick"], quake),
            table(["Worked line", "Raised dot numbers in cell order", "GPU text"], braille),
            table(["Worked ending", "Leader", "Trump", "Opening hand", "Forced NS tricks", "Optimal leads"], bridge))


def category_table(audit, task, categories, unit):
    totals, correct = audit["labels_by_category"][task], audit["correct_labels_by_category"][task]
    return table([unit, "Correct / tested"],
                 [[text, ratio(correct.get(key, 0), totals.get(key, 0))] for key, text in categories])


def render(run, dest):
    audit = evaluate(run)
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    cases = {case["name"]: case["input"] for case in json.loads((HERE / "checks.json").read_bytes())}
    with np.load(run / "output/mosaic-predictions.npz", allow_pickle=False) as archive:
        worked = {name: archive["check_" + name].tolist() for name in cases}
    quake, braille, bridge = named_examples(cases, worked)
    dest.mkdir(parents=True, exist_ok=True)
    source_figure(dest / QUAKE_FIGURE, cases, worked)
    rows = audit["conditions"]
    write_sizes(dest / "sizes.csv", rows, protocol)

    beyond = [r for r in rows if r["size"] > max(protocol["tasks"][r["task"]]["dev"])]
    checks = audit["worked_checks"]
    sizes = table(["Application", "Size", "FP32 parameters", "Exact records", "Correct labels", "Median GPU time (ms)"],
                  [[TASKS[r["task"]], r["size"], r["parameters"], ratio(r["exact_records"], r["records"]),
                    ratio(r["correct_labels"], r["labels"]), f"{r['median_seconds']*1000:.3f}"] for r in rows])
    distribution = table(["Compatible locations in event", "Exact maps / tested events"],
                         [[name, ratio(audit["exact_quake_records"].get(key, 0), audit["quake_records"].get(key, 0))]
                          for key, name in (("none", "None"), ("unique", "Exactly one"), ("multiple", "More than one"))])
    compatible = ratio(audit["correct_labels_by_category"]["quake"].get("compatible_location", 0),
                       audit["labels_by_category"]["quake"].get("compatible_location", 0))
    braille_dist = category_table(audit, "braille", [(v, v.title()) for v in ("digit", "uppercase", "lowercase", "space", "indicator")],
                                  "Reference cell reading")
    bridge_dist = category_table(audit, "bridge", [("unique_opening", "One optimal lead"), ("tied_openings", "Both leads optimal"),
                                ("no_trump", "No-trump contract"), ("trump", "Suit contract"),
                                ("higher_opening_suboptimal", "Higher-ranked card strictly worse")], "Reference deal category")
    pairs = audit["context_pairs"]
    pair_table = table(["Cells per line", "Both final characters correct / pairs", "Both complete lines exact / pairs"],
                       [[r["size"], ratio(r["both_final_correct"], r["pairs"]), ratio(r["exact_pairs"], r["pairs"])] for r in pairs])
    src = f"../../kaggle/runs/mosaic/{run.name}"
    report = [
        "# Seismic source location, Braille reading and bridge endgames",
        f"The three hand-programmed transformers matched independent references on **{total(rows, 'exact_records', 'records')} "
        f"complete held-out records** and {total(rows, 'correct_labels', 'labels')} individual outputs on a Kaggle "
        f"{audit['run']['device']}. At sizes above the development range, {total(beyond, 'exact_records', 'records')} "
        f"records were exact. The separate worked examples matched on {ratio(sum(v['correct'] for v in checks.values()), len(checks))} inputs.",
        "## Accuracy and runtime",
        sizes,
        "Size counts station rows per event, cells per Braille line or independent bridge deals per record. A seismic "
        "record requires the entire location and origin-time map. A Braille line requires every aligned reading, "
        "including silent indicators. A bridge record requires the trick count and all optimal leads on every deal. "
        "Development used 8 records per size at 2 and 4 stations, 8 and 16 Braille cells, and 2 and 4 bridge deals.",
        f"Each time is the median of {protocol['repeats']} synchronized FP32 forward sweeps over {protocol['per_size']['test']} "
        f"GPU-resident records in batches of {protocol['batch']}, after warm-up, with TF32 disabled. Encoding, data transfers, "
        "decoding and scoring fall outside the timed interval. Parameter counts include stored FP32 coefficients, including "
        "zeros. Fixed program logic also supplies the encodings, masks, arithmetic and output interpretation. "
        "[All timing repeats in seconds](sizes.csv)",
        "## Seismology · locating a source from first arrivals",
        f"![Compatible source positions and origin ticks with two measured arrivals and then a third]({QUAKE_FIGURE})",
        "The synthetic event starts at an integer tick from 0 to 15 on a 7 × 7 horizontal grid. A circular wave travels "
        "one grid unit per tick in a homogeneous medium. Each measured station reports the first integer tick at or after "
        "arrival. For example, from (2, 3) the wave reaches (0, 0) in √13 ticks, so the recorded delay is 4. "
        "An attention lookup reads the travel delay for each station-to-source squared distance. Two more attention "
        "reads find the earliest and latest origin ticks implied at each location. A location qualifies when these agree "
        "and the tick falls in range. "
        "The figure uses the two saved GPU maps. A square with a dash marks the third station while its reading is missing.",
        quake, distribution,
        f"At individual compatible locations, {compatible} predicted origin ticks were correct. The complete-map scores "
        "above also check every location that should be excluded. The generator mixes consistent arrivals, missing "
        "measurements, layouts admitting multiple sources, one-tick perturbations and contradictory readings. "
        "More stations constrain the same 49 possible source locations. This toy inversion omits depth, variable wave "
        "speed and measurement uncertainty. [Runnable seismic example](../../mosaic/quake_notes.md)",
        "## Accessibility · reading a bounded uncontracted English Braille label",
        "The reader receives ordered six-dot cells and emits a print character or a silent indicator at each position. "
        "Dots 12 can print as b, B or 2 depending on earlier indicators. One attention head matches the raised dots "
        "to a glyph and another reads the most recent earlier cell that starts or ends numeric mode. The sampled lines exercise long integers, "
        "grade-1 and letter transitions out of number mode, capitals and spaces. The inputs use transcribed dot patterns. "
        "The supported repertoire covers the English alphabet, single capitals, integers, grade-1 symbol indicators and "
        "blanks. Contractions, punctuation and physical dot recognition fall outside it. "
        "[Runnable Braille example](../../mosaic/braille_notes.md)",
        braille, braille_dist,
        "### Can the first indicator change the final character?",
        "Within each additional pair, the first line starts with a number indicator and the second with a single-capital "
        "indicator. Every cell after that is identical in both lines and uses an a–j pattern. The last cell must read as "
        "a digit in one line and a lowercase letter in the other. A deterministic reader that sees only the shared suffix "
        "must give the same final character twice, so it can get at most one of the two right. Eight pairs end with each "
        "of the ten patterns at each size. The paired lines are distinct from the sampled benchmark and are scored separately.",
        pair_table,
        f"The GPU recovered both complete lines in {total(pairs, 'exact_pairs', 'pairs')} pairs and both final characters "
        f"in {total(pairs, 'both_final_correct', 'pairs')} pairs.",
        "## Contract bridge · choosing an opening lead with two tricks left",
        "Each player holds two cards and all four hands are visible. In this double-dummy ending North and South maximize "
        "their remaining tricks, while East and West minimize them. Each deal requires the North–South total and every "
        "optimal opening card. The model scores possible first-trick plays in parallel. Attention selects each trick winner "
        "and backs up the best legal replies for the partnership on turn. In the worked king-of-clubs hand, changing the trump suit changes the value of leading "
        "the six of spades. NT means no-trump. Card names put suit before rank. [Runnable bridge example and full hands](../../mosaic/bridge_notes.md)",
        bridge, bridge_dist,
        "The deal categories overlap across rows. The generator samples eight distinct cards from decks restricted to "
        "one, two or four suits and mixes suit and no-trump contracts. Increasing the size adds independent deals to one "
        "record. Search depth stays at two tricks. Hidden hands, bidding and full 13-trick play are outside this experiment.",
        "## Reproduce the run",
        f"[Worked inputs and labels](../../mosaic/checks.json) · [Frozen protocol](../../mosaic/protocol.json) · "
        f"[Independent audit](audit.json) · [Saved predictions]({src}/output/mosaic-predictions.npz) · "
        f"[GPU verification]({src}/verification.json) · [Submission receipt]({src}/submission.json) · "
        f"[Private Kaggle run]({audit['run']['url']})",
        "```sh\npython mosaic/experiment.py predict quake --example\npython mosaic/experiment.py predict braille --example\n"
        "python mosaic/experiment.py predict bridge --example\n"
        f"python mosaic/audit.py kaggle/runs/mosaic/{run.name} audit-output/mosaic-audit.json\n"
        f"python mosaic/report.py kaggle/runs/mosaic/{run.name} results/mosaic\n```",
    ]
    (dest / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(rows)} settings, {len(checks)} worked cases and {sum(r['pairs'] for r in pairs)} Braille pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.run, args.output)
