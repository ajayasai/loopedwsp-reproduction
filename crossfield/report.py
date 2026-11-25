"""Render the cross-field study from independently regraded Kaggle predictions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit import evaluate


HERE = Path(__file__).resolve().parent
TASKS = {"inflection": "Turkish noun inflection", "diffraction": "Crystal diffraction",
         "auction": "Combinatorial auction"}
MOTIFS = {"equal_body_centered": "Body-centered, amplitudes 1 and 1",
          "unequal_body_center": "Body-center sites, amplitudes 1 and 2",
          "equal_face_centered": "Face-centered, four equal amplitudes",
          "quarter_cell_phase": "Quarter-cell offset, amplitudes 2 and 1"}
AUCTIONS = {"specialists_beat_bundle": "Two specialists and a bundle bid",
            "disjoint_zero_price_winners": "Disjoint requests", "tied_item": "Identical single-item bids"}


def read(path):
    return json.loads(path.read_bytes())


def fraction(numerator, denominator):
    return f"{numerator:,}/{denominator:,}"


def table(columns, rows):
    all_rows = [columns, ["---"] * len(columns), *rows]
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in all_rows)


def save_timings(path, rows, protocol):
    fields = ["task", "size", "unit", "parameters", "records", "exact_records", "labels", "correct_labels",
              "nonfinite", "median_seconds", "seconds_1", "seconds_2", "seconds_3"]
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fields, lineterminator="\n")
        out.writeheader()
        for row in rows:
            record = {k: row[k] for k in fields if k in row}
            record["unit"] = protocol["tasks"][row["task"]]["size_unit"]
            record.update({f"seconds_{i}": s for i, s in enumerate(row["seconds"], start=1)})
            out.writerow(record)


def diffraction_plot(output, checks, predictions):
    names = ["equal_body_centered", "unequal_body_center", "equal_face_centered"]
    reflections = checks[names[0]]["input"]["reflections"]
    x = np.arange(len(reflections))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
    for i, (name, color) in enumerate(zip(names, ["#326488", "#ad522d", "#417c63"], strict=True)):
        if checks[name]["input"]["reflections"] != reflections:
            raise ValueError("Cannot compare these motifs at different Miller indices")
        values = predictions[name]
        bars = ax.bar(x + .25*(i-1), values, width=.23, color=color, label=MOTIFS[name])
        ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_xticks(x, ["(" + " ".join(map(str, h)) + ")" for h in reflections])
    ax.set_xlabel("Miller indices (h k l)")
    ax.set_ylabel("Predicted intensity |F|² (model units)")
    ax.set_title("Site positions and scattering contrast change the reflection intensities", loc="left", fontsize=13)
    ax.set_ylim(0, max(max(predictions[name]) for name in names) + 3)
    ax.set_yticks(range(0, 17, 4))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dfdfdf", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.21), frameon=False, fontsize=9, ncols=2)
    fig.savefig(output / "crystal-intensities.png", dpi=150)
    plt.close(fig)


def named_examples(checks, predictions, worked):
    name = "real_regular_nouns"
    requests = checks[name]["input"]["requests"]
    nouns = table(["Stem", "Requested number", "Requested case", "GPU form"],
                 [[r["stem"], r["number"], r["case"], form]
                  for r, form in zip(requests, worked[name]["forms"], strict=True)])
    crystal_rows = []
    for name, label in MOTIFS.items():
        refs = checks[name]["input"]["reflections"]
        triples = ", ".join("(" + " ".join(map(str, h)) + ")" for h in refs)
        values = ", ".join(map(str, predictions[name]))
        crystal_rows.append([label, triples, values])
    crystals = table(["Motif", "Reflections in query order", "GPU intensities"], crystal_rows)
    bids = []
    for name, label in AUCTIONS.items():
        offers = checks[name]["input"]["bids"]
        codes = predictions[name]
        requested = ", ".join(f"{b['bundle']} for {b['value']}" for b in offers)
        outcome = " · ".join("loses" if c == 0 else f"wins (pays {c-1})" for c in codes)
        welfare = sum(b["value"] for b, c in zip(offers, codes, strict=True) if c > 0)
        bids.append([label, requested, outcome, welfare])
    auctions = table(["Auction", "Bundle bids in order", "GPU outcomes in order", "Total winning value"], bids)
    return nouns, crystals, auctions


def render(run, output):
    audit = evaluate(run)
    protocol = read(HERE / "protocol.json")
    checks = {c["name"]: c for c in read(HERE / "checks.json")}
    raw = read(run / "output/crossfield-results.json")
    receipt = read(run / "submission.json")
    with np.load(run / "output/crossfield-predictions.npz", allow_pickle=False) as archive:
        predictions = {name: archive[f"check_{name}"].tolist() for name in checks}
    output.mkdir(parents=True, exist_ok=True)
    rows = audit["conditions"]
    categories = audit["test_categories"]
    bigger = [r for r in rows if r["size"] > max(protocol["tasks"][r["task"]]["dev"])]
    def counts(records, field):
        return sum(record[field] for record in records)

    exact_count = fraction(counts(rows, "exact_records"), counts(rows, "records"))
    label_count = fraction(counts(rows, "correct_labels"), counts(rows, "labels"))
    large_count = fraction(counts(bigger, "exact_records"), counts(bigger, "records"))
    numeric = [[TASKS[r["task"]], r["size"], r["parameters"], fraction(r["exact_records"],r["records"]),
                fraction(r["correct_labels"],r["labels"]), f"{1000*r['median_seconds']:.2f}"] for r in rows]
    save_timings(output / "sizes.csv", rows, protocol)
    diffraction_plot(output, checks, predictions)
    nouns, crystals, auctions = named_examples(checks, predictions, raw["checks"])
    inf, diff, auction = (categories[t] for t in ("inflection","diffraction","auction"))
    named_count = fraction(sum(c["correct"] for c in audit["checks"].values()), len(checks))
    version = receipt["version"]
    base = f"../../kaggle/runs/crossfield/v{version}"
    paragraphs = [
        "# Turkish morphology, crystal diffraction and auction pricing",
        f"On a Kaggle {raw['device']}, three hand-programmed transformers matched their independent references on "
        f"**{exact_count} complete held-out records** and {label_count} individual output labels. At input sizes above "
        f"the development range the models matched **{large_count} records**. The separate named check set matched "
        f"on {named_count} inputs. Those checks are excluded from accuracy totals and timing.",
        "## Accuracy and runtime by input size",
        table(["Application","Size","Stored FP32 coefficients","Exact records","Correct labels","Median ms"], numeric),
        "Size counts noun requests, Miller-index reflections or bidders per record. An exact record requires both "
        "suffixes for every noun, every intensity in the reflection list or every bidder’s winner and payment status. "
        "Longer noun and reflection lists contain more independent queries. More bidders increase the number of possible allocations. "
        "Each time is the median of three synchronized FP32 forward passes over 64 GPU-resident records in batches of eight "
        "after a warm-up. Encoding, transfers and scoring occur outside the timer. Stored coefficient counts include "
        "zeros and count shared weights once. Fixed input features, masks and working memory are excluded. [Individual repeats](sizes.csv)",
        "## Linguistics · inflecting regular Turkish nouns",
        f"The held-out generator combines regular nonce stems with number and case requests. The model predicted both "
        f"suffixes correctly for {fraction(inf['forms_correct'],inf['requests'])} requests covering "
        f"{inf['distinct_stems']:,} distinct stems and {inf['distinct_surface_forms']:,} distinct surface forms. The formatter copies "
        "the supplied lowercase stem and appends the predictions. In the separate real-word check below, the stated "
        "number and case select one surface form per row.",
        nouns,
        "The plural suffix can supply the vowel governing the following case suffix, as in *göl → göl-ler-i*. "
        "Consonant assimilation selects *-tan* in *kuştan*. The study covers unchanged common-noun stems with regular "
        "harmony and five cases. Stem alternations, exceptional harmony, possessives and proper names are outside its input contract. "
        "[Runnable inflection example](../../crossfield/examples/inflection.json)",
        "## Crystallography · predicting reflection intensity",
        "![GPU-predicted intensities for three point motifs at four common reflections](crystal-intensities.png)",
        crystals,
        f"The model matched {fraction(diff['zero_correct'], diff['zero'])} canceled reflections and "
        f"{fraction(diff['nonzero_correct'],diff['nonzero'])} nonzero reflections. Both members were exact in "
        f"{fraction(diff['friedel_pairs_correct'],diff['friedel_pairs'])} opposite-index pairs counted within records. "
        "In the first two-site example, the two (1 0 0) contributions have opposite phase and cancel. Changing "
        "the center amplitude from 1 to 2 leaves intensity 1 at the same reflection. The quarter-cell example also "
        "exercises imaginary amplitudes. These are ideal kinematical intensities from fixed real point amplitudes on "
        "a quarter-cell grid. Assigning a space group or predicting measured detector counts requires additional information. "
        "[Runnable diffraction example](../../crossfield/examples/diffraction.json) · [IUCr definition](https://dictionary.iucr.org/Structure_factor)",
        "## Market design · allocating bundles and computing externality payments",
        auctions,
        "The two specialist bids together yield 13 units of reported value. With the A bidder removed, the best "
        "remaining allocation is the 10-unit bundle. With A present, the other winner receives a bundle valued at 6, "
        "so the A bidder pays 4. The B bidder pays 3 by the same rule. A winner can pay zero when their participation "
        "imposes no cost on other bidders. Equal-welfare allocations use the smaller winner bit mask, where the "
        "first bidder has the least significant bit.",
        f"Across the held-out auctions, the model matched {fraction(auction['greedy_suboptimal_correct'],auction['greedy_suboptimal'])} "
        "records where taking bids greedily in descending order loses welfare and "
        f"{fraction(auction['tied_optima_correct'],auction['tied_optima'])} records with multiple welfare-maximizing allocations. "
        f"It identified {fraction(auction['zero_price_winners_correct'],auction['zero_price_winners'])} zero-price winners. "
        "Each bidder requests one bundle of the four available items. The implementation allocates one attention-memory "
        "slot per bidder subset, up to 256 slots for eight bidders. [Runnable auction example](../../crossfield/examples/auction.json)",
        "## Run and verify",
        f"[Models, input bounds and worked examples](../../crossfield/README.md) · [Frozen protocol](../../crossfield/protocol.json) · "
        f"[Named check inputs and expected answers](../../crossfield/checks.json) · [Independent regrading](audit.json) · "
        f"[Raw predictions]({base}/output/crossfield-predictions.npz) · [Submission receipt]({base}/submission.json) · "
        f"[GPU verification]({base}/verification.json) · [Private Kaggle run]({receipt['url']})",
        "The examples use NumPy and PyTorch. Regenerating the figure also requires Matplotlib.",
        "```sh\npython crossfield/experiment.py predict inflection --example\n"
        "python crossfield/experiment.py predict diffraction --example\n"
        "python crossfield/experiment.py predict auction --example\n"
        f"python crossfield/audit.py kaggle/runs/crossfield/v{version} audit-output/crossfield-audit.json\n"
        f"python crossfield/report.py kaggle/runs/crossfield/v{version} results/crossfield\n```"]
    (output / "report.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(rows)} independently audited GPU settings")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    render(args.version_folder, args.destination)
