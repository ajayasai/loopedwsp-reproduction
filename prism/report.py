"""Render the three-field report from the verified Kaggle prediction archive."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit import evaluate
from oracles import isotopes


HERE = Path(__file__).resolve().parent
NAMES = {"isotopes": "Isotope envelopes", "calendar_round": "Maya Calendar Round", "arm": "Lattice arm"}


def ratio(a, b):
    return f"{a:,}/{b:,}"


def table(headings, rows):
    return "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in [headings, ["---"] * len(headings), *rows]
    )


def plot_spectrum(path, record, predictions):
    bins = np.arange(400, 416)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    floor = np.zeros(16)
    for ion, color in zip(record["ions"], ["#24686d", "#dbac4b", "#6387b1"], strict=True):
        values = np.asarray(isotopes({"ions": [ion]})[0])
        atoms = " + ".join(
            f"{ion[key]} {symbol}" for key, symbol in (("cl", "Cl"), ("br", "Br")) if ion[key]
        )
        description = f"Parent {ion['mz']} ({atoms or 'no halogens'})"
        ax.bar(bins, values, bottom=floor, width=0.57, color=color, label=description)
        floor += values
    ax.scatter(
        bins,
        predictions,
        s=40,
        marker="D",
        facecolor="white",
        edgecolor="#212a32",
        linewidth=1.5,
        label="GPU mixture total",
        zorder=4,
    )
    for x, y in zip(bins, predictions, strict=True):
        if y:
            ax.annotate(str(y), (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set(
        xlabel="Nominal mass-to-charge ratio (m/z)",
        ylabel="Intensity units",
        xticks=bins,
        xlim=(399.5, 415.5),
        ylim=(-4, max(max(predictions), max(floor)) * 1.17),
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.set_title("Two ion envelopes meet at m/z 402 and 404", loc="left", fontsize=16, pad=14)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.text(
        0.1,
        0.035,
        "Stacked colors show independently calculated component intensities. Diamonds are saved GPU predictions.",
        ha="left",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.19)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def worked_tables(cases, predictions):
    isotope_rows, date_rows, arm_rows = [], [], []
    for name, case in cases.items():
        record, output = case["input"], predictions[name]
        if case["task"] == "isotopes":
            ions = ", ".join(
                f"{ion['mz']} ({ion['cl']} Cl, {ion['br']} Br, amount {ion['abundance']})"
                for ion in record["ions"]
            )
            peaks = ", ".join(f"{400 + i} ({value})" for i, value in enumerate(output) if value)
            isotope_rows.append([ions, peaks or "none"])
        elif case["task"] == "calendar_round":
            for date, year, day in zip(record["dates"], output[::2], output[1::2], strict=True):
                designation = f"{date['number']} {date['name']} · {date['day']} {date['month']}"
                offset = "incompatible" if year == day == 365 else str(365 * year + day)
                date_rows.append([designation, offset])
        else:
            poses = []
            for count, code in zip(output[::2], output[1::2], strict=True):
                word = "".join("LSR"[(code // 3**n) % 3] for n in (3, 2, 1, 0)) if code < 81 else "none"
                poses.append(f"{count} ({word})")
            arm_rows.append(
                [str(record["lengths"]), str(record["obstacles"]), str(record["targets"]), ", ".join(poses)]
            )
    return (
        table(["Parent masses and component inputs", "Nonzero GPU bins (intensity)"], isotope_rows),
        table(["Paired Tzolk'in and Haab' designation", "GPU day offset or incompatibility"], date_rows),
        table(["Link lengths", "Obstacles", "Targets in input order", "GPU count (first pose)"], arm_rows),
    )


def render(run, dest):
    audit = evaluate(run)
    rows = audit["conditions"]
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    cases = {c["name"]: c for c in json.loads((HERE / "checks.json").read_bytes())}
    with np.load(run / "output/prism-predictions.npz", allow_pickle=False) as store:
        predictions = {name: store["check_" + name].tolist() for name in cases}
    dest.mkdir(parents=True, exist_ok=True)
    repeats = protocol["repeats"]
    fields = [
        "parameters",
        "records",
        "exact_records",
        "answers",
        "correct_answers",
        "labels",
        "correct_labels",
        "nonfinite",
        "median_seconds",
    ]
    with (dest / "sizes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["task", "size", "unit", *fields, *[f"seconds_{i + 1}" for i in range(repeats)]])
        for r in rows:
            writer.writerow(
                [
                    r["task"],
                    r["size"],
                    protocol["tasks"][r["task"]]["size_unit"],
                    *[r[f] for f in fields],
                    *r["seconds"],
                ]
            )
    key = "overlapping_envelopes"
    plot_spectrum(dest / "isotope-overlap.png", cases[key]["input"], predictions[key])
    example_spectra, example_dates, example_arms = worked_tables(cases, predictions)

    def result(subset, hit, all_items):
        return ratio(sum(r[hit] for r in subset), sum(r[all_items] for r in subset))

    def category(task, name):
        return ratio(
            audit["correct_answers_by_category"][task].get(name, 0),
            audit["answers_by_category"][task].get(name, 0),
        )

    extended = [r for r in rows if r["size"] > max(protocol["tasks"][r["task"]]["dev"])]
    sizes = table(
        [
            "Application",
            "Size",
            "Stored FP32 weights",
            "Exact records",
            "Complete answers",
            "Correct integer labels",
            "Median GPU ms",
        ],
        [
            [
                NAMES[r["task"]],
                r["size"],
                f"{r['parameters']:,}",
                ratio(r["exact_records"], r["records"]),
                ratio(r["correct_answers"], r["answers"]),
                ratio(r["correct_labels"], r["labels"]),
                f"{1000 * r['median_seconds']:.2f}",
            ]
            for r in rows
        ],
    )
    checks = audit["worked_checks"]
    exhaustive = audit["exhaustive_calendar"]
    src = f"../../kaggle/runs/prism/{run.name}"
    report = [
        "# Isotope envelopes, the Maya Calendar Round and tabletop robotics",
        f"On a Kaggle {audit['run']['device']}, the three hand-programmed models matched "
        f"**{result(rows, 'exact_records', 'records')} held-out records** and "
        f"**{result(rows, 'correct_answers', 'answers')} complete answers**. The two sizes beyond development "
        f"accounted for {result(extended, 'exact_records', 'records')} exact records. "
        f"The separate worked checks passed on {ratio(sum(c['correct'] for c in checks.values()), len(checks))} inputs.",
        "## Accuracy and runtime by input size",
        sizes,
        "Size counts ion components in one mixture, date pairs in one list, or target positions in one arm scene. "
        "Each intensity bin contributes one integer label. One calendar answer requires both parts of its "
        "offset or the incompatible pair. One arm answer requires the exact count and first witness, including "
        "the sentinel for an unreachable target. Exact record accuracy requires every answer in that record. "
        "Longer date lists and arm target lists increase the number of queries. The Calendar Round length and "
        "the arm’s four links stay fixed.",
        f"Each time is the median of {repeats} synchronized forward sweeps over "
        f"{protocol['per_size']['test']} GPU-resident records after warm-up, in batches of {protocol['batch']}. "
        "Timing excludes encoding, transfers, decoding and verification. Weight counts include stored FP32 "
        "coefficients, including zeros, and count shared weights once. Each model also uses fixed operations "
        "and working memory. [All timing repeats](sizes.csv)",
        "## Analytical chemistry · overlapping isotope envelopes",
        "![Calculated component intensities and Kaggle-predicted mixture totals for all 16 bins](isotope-overlap.png)",
        "Each abundance unit supplies 64 intensity units. Only chlorine and bromine contribute isotope shifts, "
        "with idealized heavy fractions of 1/4 and 1/2. The example's bin at m/z 402 receives 24 units from the "
        "two-chlorine component and 64 from the bromine component. The GPU predicted 88. The table lists "
        "only occupied peaks. All 16 nominal bins contribute to each test score.",
        example_spectra,
        f"The model matched {category('isotopes', 'occupied_bins')} occupied bins and {category('isotopes', 'empty_bins')} "
        f"empty bins. Among the occupied bins, it matched {category('isotopes', 'overlap_from_distinct_masses')} "
        "that received intensity from distinct light-isotope parent masses. The generator samples broad mass ranges, same-parity masses "
        "and narrow clusters to exercise overlaps. Real spectra can contain isotope satellites from other "
        "elements, instrumental broadening and noise. [Runnable mixture](../../prism/isotopes_notes.md)",
        "## Mesoamerican calendrics · reconciling two date cycles",
        "The 260-day Tzolk'in and 365-day Haab' repeat together after 18,980 days. A compatible pair identifies "
        "one offset in that round, counted from 4 Ajaw 8 Kumk'u. A pair incompatible with this alignment has "
        "no offset. These are relative cycle positions. The same designation recurs in subsequent rounds.",
        example_dates,
        f"The model matched {category('calendar_round', 'compatible')} compatible offsets and "
        f"{category('calendar_round', 'incompatible')} incompatible pairs. It matched "
        f"{category('calendar_round', 'short_month')} requests in the five-day final month, Wayeb'. The generator "
        "pairs valid dates with altered day names and deliberately samples Wayeb' dates. Those proportions "
        "describe the test design. [Runnable calendar example](../../prism/calendar_round_notes.md)",
        f"In a separate prespecified GPU sweep, the model answered {ratio(exhaustive['exact'], exhaustive['pairs'])} "
        f"well-formed pairs exactly. It produced the right offset for "
        f"{ratio(exhaustive['compatible_exact'], exhaustive['compatible'])} compatible pairs and recognized "
        f"{ratio(exhaustive['incompatible_exact'], exhaustive['incompatible'])} incompatible pairs. "
        "This covers the Cartesian product of the 260 Tzolk'in positions and 365 Haab' positions. "
        "The headline record counts and timing table cover the sampled benchmark separately.",
        "## Robotics · count collision-free lattice-arm poses",
        "The arm has four links and three choices per joint, yielding 81 candidate words. For each target the "
        "GPU returns the number of feasible words and the first in left, straight, right order. Each candidate "
        "must avoid the obstacle points and self-intersection, including along the interior of a link. On the "
        "unit-link board, blocking [1,0] cuts the possible poses at [2,2] from six to three.",
        example_arms,
        f"The model matched {category('arm', 'unique')} targets with one feasible pose, "
        f"{category('arm', 'multiple')} with several, and {category('arm', 'unreachable')} with none. In "
        f"{category('arm', 'all_poses_blocked')} instances, obstacles removed every pose available on an empty "
        f"board. In {category('arm', 'some_poses_blocked')}, obstacles removed some poses while leaving at least "
        "one. The generator selects from these kinds of targets when available. The model evaluates static "
        "positions on an integer lattice. Movement between two poses requires separate motion planning. [Runnable arm example](../../prism/arm_notes.md)",
        "## Run and verify",
        f"[Model guide and commands](../../prism/README.md) · [Worked inputs and labels](../../prism/checks.json) · "
        f"[Frozen protocol](../../prism/protocol.json) · [Independent regrading](audit.json) · [Raw predictions]({src}/output/prism-predictions.npz) · "
        f"[Kaggle verification]({src}/verification.json) · [Submission receipt]({src}/submission.json) · [Private Kaggle run]({audit['run']['url']})",
        "```sh\npython prism/experiment.py predict isotopes --example\npython prism/experiment.py predict calendar_round --example\n"
        "python prism/experiment.py predict arm --example\n"
        f"python prism/audit.py kaggle/runs/prism/{run.name} audit-output/prism-audit.json\n"
        f"python prism/report.py kaggle/runs/prism/{run.name} results/prism\n```",
    ]
    (dest / "report.md").write_text("\n\n".join(report) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(rows)} settings and {len(checks)} worked checks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.run, args.output)
