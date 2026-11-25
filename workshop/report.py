"""Build the workshop report and media from independently audited GPU predictions."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from export import save_frame, save_roll


HERE = Path(__file__).resolve().parent
NAMES = {"mobile": "Kinetic mobile", "cels": "Animation cels", "roll": "Punched music roll"}
NOTES = ("rest", "C4", "D4", "E4", "F4", "G4", "A4", "B4",
         "C5", "D5", "E5", "F5", "G5", "A5", "B5", "C6")


def ratio(n, d):
    return f"{n:,}/{d:,}"


def counted(counts, name):
    return ratio(counts[name + "_correct"], counts[name])


def rgb_index(x, y):
    return 3 * (8*y + x)


def visible_blend(record):
    """Find a half-opacity cel visibly mixing with an earlier cel at the same pixel."""
    candidates = []
    for y in range(8):
        for x in range(8):
            before = record["background"]
            visible = []
            for index, cel in enumerate(record["cels"]):
                x0, y0, x1, y1 = cel["rect"]
                alpha, source = cel["opacity"], cel["rgb"]
                if alpha == 0 or not (x0 <= x < x1 and y0 <= y < y1):
                    continue
                visible.append((index, before, cel))
                before = [(alpha*s + (2-alpha)*b + 1)//2 for s, b in zip(source, before)]
            if len(visible) > 1:
                index, under, top = visible[-1]
                distance = sum(abs(s - b) for s, b in zip(top["rgb"], under))
                if top["opacity"] == 1 and distance > 0:
                    candidates.append((distance, x, y, index, under, top["rgb"]))
    return max(candidates, default=None)


def cel_example(cases, preds, output):
    # Pick a frame by input/oracle characteristics: visible overlap, then
    # number of distinct colors and blend contrast. Preserve the GPU pixels.
    options = []
    for i, row in enumerate(cases["test/cels/8"]):
        overlap = visible_blend(row["input"])
        if overlap is not None and np.all(preds[i] >= 0):
            colors = len(np.unique(np.asarray(row["gold"]).reshape(-1, 3), axis=0))
            options.append((colors, overlap[0], -i, overlap))
    if not options:
        raise ValueError("The eight-layer cohort has no renderable example with a visible half-opacity overlap")
    _, _, negative_index, (_, x, y, index, under, source) = max(options)
    i = -negative_index
    save_frame(output / "composite.png", preds[i], scale=32)
    j = rgb_index(x, y)
    color = preds[i, j:j+3].tolist()
    return [
        "## See the rendered animation frame",
        "![Eight-layer RGB animation frame enlarged with square pixels](composite.png)",
        f"Each square represents one of the 8 × 8 original pixels. In test record `cels_8[{i}]`, "
        f"pixel (x={x}, y={y}) was `{under}` before cel {index+1} applied a half-opacity `{source}` tint. "
        f"The GPU returned `{color}` there. The PNG scales each four-bit channel by 17 for display.",
        "Each step attends to the next cel at each pixel and applies its opacity. The canvas stays 8 × 8 at all test sizes as "
        "the stack grows from four to 32 layers. It uses an opaque background and 16 levels per RGB channel. Cels are solid "
        "axis-aligned rectangles with opacity 0, ½ or 1. Half-integer blends round upward after each layer."
    ]


def mobile_example(cases, preds):
    records = cases["test/mobile/4"]
    i = next(i for i, row in enumerate(records) if row["gold"][0] > 0)
    row, guess = records[i], int(preds[i, 0])
    record = row["input"]
    masses = record["masses"]
    limbs = record["arms"]
    display = ", ".join("?" if v is None else str(v) for v in masses)
    weights = [row["gold"][0] if v is None else v for v in masses]
    left, right = sum(weights[:2]), sum(weights[2:])
    expected = row["gold"][0]
    prediction = "no admissible mass" if guess == 0 else "a nonfinite output" if guess < 0 else f"{guess} g"
    return [
        "## Find the missing counterweight",
        f"Test record `mobile_4[{i}]` has four ornament masses `[{display}]` grams, ordered left to right. "
        f"The GPU predicted **{prediction}**. The integer oracle gives {expected} g. With that mass, all three rods balance.",
        "| Rod | Left torque | Right torque |\n| --- | ---: | ---: |\n" +
        "\n".join(f"| {name} | {a} × {l} = {a*l} | {b} × {r} = {b*r} |"
                  for name, a, b, (l,r) in [("Lower left", *weights[:2], limbs[1]),
                                             ("Lower right", *weights[2:], limbs[2]),
                                             ("Root", left, right, limbs[0])]),
        "The model tests 16 integer counterweights in parallel. A shared block propagates subtree masses and torque errors "
        "up the complete binary tree. Every rod is horizontal and massless in this idealized tabletop model. "
        "The admissible missing mass is 1–16 g."
    ]


def roll_event(cls):
    if cls == 144:
        return "—", "double error"
    if cls < 0:
        return "—", "nonfinite output"
    track, event = divmod(int(cls), 16)
    return NOTES[event], "clean" if track == 0 else f"corrected track {track}"


def roll_example(cases, preds, output):
    records = cases["test/roll/16"]
    allowed = [i for i in range(len(records)) if (preds[i] >= 0).all()]
    if not allowed:
        raise ValueError("No 16-frame GPU prediction can be rendered")
    def variety(i):
        return len({cls % 16 for cls in records[i]["gold"] if cls < 144}), -i
    i = max(allowed, key=variety)
    holes = records[i]["input"]["frames"]
    pred = preds[i]
    save_roll(output / "recovered.wav", pred)
    examples = ["| Beat | Observed holes | GPU event | GPU status |", "| ---: | --- | --- | --- |"]
    for j, cls in enumerate(pred[:8]):
        note, status = roll_event(cls)
        examples.append(f"| {j+1} | `{holes[j]}` | {note} | {status} |")
    return [
        "## Hear a recovered punched roll",
        f"The WAV synthesizes the 16 GPU-decoded beats in test record `roll_16[{i}]`. Rests and uncorrectable beats "
        "are silent. The class IDs distinguish the two. [Listen to the decoded roll](recovered.wav)",
        "These are the first eight observed track patterns and their predictions.",
        "\n".join(examples),
        "This synthetic format encodes a rest or one of 15 pitches in each eight-hole frame. The block reduces parity checks "
        "in three passes. Extended Hamming parity identifies a single changed hole and flags a two-hole change. The repair "
        "guarantee covers at most two changed holes per frame. Each frame is processed independently. Development checks "
        "covered all 256 possible observed patterns across 592 combinations of original events and zero, one or two changed holes. "
        "Longer held-out rolls test complete sequences formed from new combinations of these familiar patterns."
    ]


def render(run, output):
    from audit import evaluate

    # evaluate verifies local, embedded and remote source hashes, weight and
    # corpus digests, archive shapes, predictions, counts and the timing repeats.
    audited = evaluate(run)
    rows, cats = audited["conditions"], audited["test_categories"]
    plan = json.loads((HERE/"protocol.json").read_bytes())
    sizes = plan["splits"]["test"]["sizes"]
    keys = [(task, size) for task in NAMES for size in sizes]
    if [(r["task"],r["size"]) for r in rows] != keys:
        raise ValueError("Unexpected order or coverage in the audited condition rows")
    cases = json.loads(gzip.decompress((HERE/"cases.json.gz").read_bytes()))
    runtime = json.loads((run/"output/workshop-runtime.json").read_bytes())
    receipt = json.loads((run/"submission.json").read_bytes())
    output.mkdir(parents=True, exist_ok=True)
    repeat_names = [f"seconds_{i+1}" for i in range(plan["repeats"])]
    base_fields = ("task","size","parameters","records","exact_records","labels","correct_labels","nonfinite","median_seconds")
    with (output/"sizes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task","size","size_unit",*base_fields[2:],*repeat_names], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in base_fields}, "size_unit": plan["tasks"][row["task"]]["size_unit"],
                             **dict(zip(repeat_names,row["seconds"],strict=True))})
    table = ["| Application | Size | Stored scalars | Exact records | Correct labels | Median ms |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        task, size = row["task"], row["size"]
        table.append(f"| {NAMES[task]} | {size} {plan['tasks'][task]['size_unit']} | {row['parameters']:,} | "
                     f"{ratio(row['exact_records'],row['records'])} | {ratio(row['correct_labels'],row['labels'])} | "
                     f"{row['median_seconds']*1000:.2f} |")
    def score(group, counted_key, total_key):
        return ratio(sum(r[counted_key] for r in group),sum(r[total_key] for r in group))
    longer = [r for r in rows if r["size"] > max(plan["splits"]["dev"]["sizes"])]
    content = [
        "# Balancing mobiles, layering animation cels and repairing punched music rolls",
        f"On a Kaggle {runtime['device']}, three weight-programmed transformers matched **{score(rows,'exact_records','records')} "
        f"complete outputs** and **{score(rows,'correct_labels','labels')} individual labels**. "
        f"Sizes above the development range accounted for {score(longer,'exact_records','records')} exact records.",
        "A whole record counts when its counterweight answer, all 192 output color channels or every note and repair status agrees "
        "with the independent reference. The sizes count ornaments, animation layers or roll frames respectively.",
        "\n".join(table),
        f"Each time is the median of {plan['repeats']} synchronized forward runs over {plan['splits']['test']['per_size']} pre-encoded "
        f"GPU-resident records in batches of {plan['batch']}, after a warm-up. Each model reused its programmed FP32 tensors. "
        "Counts include stored zeros and exclude fixed geometry, operations and working memory. [All timing repeats](sizes.csv)",
        "## Cases that distinguish the tasks",
        f"- **Mobile.** The GPU matched {counted(cats['mobile'],'feasible')} feasible designs and {counted(cats['mobile'],'impossible')} "
        f"infeasible designs. A check at just the missing ornament’s parent would accept {cats['mobile']['parent_only_false_positive']} "
        f"of the infeasible designs. The GPU rejected {cats['mobile']['parent_only_false_positive_correct']} of those.\n"
        f"- **Animation.** Reversing the cel order changes the reference image in {cats['cels']['order_sensitive']} stacks. The model "
        f"matched the original frame pixel-for-pixel on {counted(cats['cels'],'order_sensitive')} of them.\n"
        f"- **Music roll.** Frame classifications matched on {counted(cats['roll'],'clean')} undamaged frames, "
        f"{counted(cats['roll'],'corrected')} single-hole repairs and {counted(cats['roll'],'double_error')} two-hole errors.",
    ]
    with np.load(run/"output/workshop-predictions.npz", allow_pickle=False) as preds:
        content += mobile_example(cases,preds["mobile_4"]) + cel_example(cases,preds["cels_8"],output)
        content += roll_example(cases,preds["roll_16"],output)
    version = receipt["version"]
    base = f"../../kaggle/runs/workshop/v{version}"
    content += ["## Evidence and reproduction",
                f"[Models and input formats](../../workshop/README.md) · [Frozen inputs and weights](../../workshop/freeze.json) · "
                f"[GPU predictions]({base}/output/workshop-predictions.npz) · [Kaggle verification]({base}/verification.json) · "
                f"[Private Kaggle version]({receipt['url']}) · [Independent audit](audit.json)",
                "From the repository root, run the audit to recompute the targets. The report command regenerates the tables, preview and audio.",
                f"```sh\npython workshop/audit.py kaggle/runs/workshop/v{version} audit-output/workshop-audit.json\n"
                f"python workshop/report.py kaggle/runs/workshop/v{version} results/workshop\n```"]
    (output/"report.md").write_text("\n\n".join(content)+"\n",encoding="utf-8",newline="\n")
    print(f"Rendered {len(rows)} independently audited GPU settings.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    render(args.version_folder.resolve(),args.output.resolve())
