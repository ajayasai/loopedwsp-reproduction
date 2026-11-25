"""Render the creative applications report from a verified Kaggle run."""
import argparse
from collections import Counter
import csv
import gzip
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median

import numpy as np


HERE = Path(__file__).resolve().parent
NAMES = {"folds": "Double-sided paper folding", "typeset": "Paragraph typesetting",
         "ballots": "Film-club ballot counting"}
UNITS = {"folds": "sheet side", "typeset": "words", "ballots": "voters"}
CLASSES = {"folds": 8, "typeset": 2, "ballots": 5}
PALETTE = ("⬛", "🟥", "🟧", "🟨", "🟩", "🟦", "🟪", "⬜")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def measurements(run):
    receipt = read(run / "submission.json")
    evidence = read(run / "verification.json")
    result_dir = run / "output"
    if (evidence["status"] != "COMPLETE" or evidence["source_match"] not in ("exact", "line_endings")
            or evidence["version"] != receipt["version"] or evidence["kernel"] != receipt["kernel"]
            or not evidence["server_private"] or evidence["server_internet_enabled"]):
        raise ValueError("Expected a collected and verified private offline Kaggle run")
    for name in ("creative-results.json", "creative-predictions.npz"):
        if digest(result_dir / name) != evidence["output_sha256"][name]:
            raise ValueError(f"Downloaded {name} differs from the collected file")
    for name in ("freeze.json", "cases.json.gz", "weights.npz"):
        if digest(HERE / name) != receipt["input_sha256"][name]:
            raise ValueError(f"Local {name} differs from the submitted file")
    results = read(result_dir / "creative-results.json")
    lock = read(HERE / "freeze.json")
    if (results["split"] != "test" or results["device"] != evidence["device"]
            or results["cuda"] != evidence["cuda"] or results["torch"] != evidence["pytorch"]
            or results["batch_size"] != lock["plan"]["batch_size"]
            or any(results[key] != lock[key] for key in ("cases_sha256", "weights_sha256"))):
        raise ValueError("Result split, GPU runtime, batch size, or input hashes differ")
    cases = json.loads(gzip.decompress((HERE / "cases.json.gz").read_bytes()))
    sizes = lock["plan"]["test"]["sizes"]
    keys = [(task, size) for task in NAMES for size in sizes]
    rows = {(r["task"], r["size"]): r for r in results["conditions"]}
    if len(results["conditions"]) != len(keys) or set(keys) != set(rows):
        raise ValueError("Unexpected measurements or missing size")
    with np.load(result_dir / "creative-predictions.npz", allow_pickle=False) as archive:
        if set(archive.files) != {f"{task}_{size}" for task, size in keys}:
            raise ValueError("Prediction arrays differ from the study plan")
        predictions = {}
        for task, size in keys:
            row = rows[task, size]
            pred = archive[f"{task}_{size}"]
            cases_key = f"test/{task}/{size}"
            gold = np.asarray([r["gold"] for r in cases[cases_key]])
            if (pred.shape != gold.shape or pred.ndim != 2 or pred.dtype.kind != "i" or gold.dtype.kind not in "iu"
                    or pred.shape[0] != lock["plan"]["test"]["records"]
                    or (pred < -1).any() or (pred >= CLASSES[task]).any()):
                raise ValueError(f"Invalid prediction array for {cases_key}")
            right = gold == pred
            checks = {"records": len(gold), "exact_records": int(right.all(1).sum()), "labels": gold.size,
                      "correct_labels": int(right.sum()), "nonfinite": int((pred == -1).sum()),
                      "parameters": lock["models"][task]["parameters"]}
            if any(row[k] != v for k, v in checks.items()):
                raise ValueError(f"Regrading failed for {cases_key}")
            timing = row["seconds"]
            if (len(timing) != lock["plan"]["repeats"] or not all(isfinite(s) and s > 0 for s in timing)
                    or median(timing) != row["median_seconds"]):
                raise ValueError(f"Inconsistent timings for {cases_key}")
            predictions[task, size] = pred
    return receipt, results, lock, cases, predictions, [rows[key] for key in keys]


def totals(rows, key):
    return sum(row[key] for row in rows)


def folds_example(cases, predictions):
    records = cases["test/folds/4"]
    def priority(i):
        row = records[i]
        folds = row["input"]["folds"]
        return (np.array_equal(predictions[i], row["gold"]),
                len({f.split("_")[1] for f in folds}), len(set(predictions[i])))
    i = max(range(len(records)), key=priority)
    record = records[i]["input"]
    height, width = len(record["front"]), len(record["front"][0])
    moves = []
    for fold in record["folds"]:
        moving, motion, stationary = fold.split("_")
        moves.append(f"fold the {moving} half {motion} the {stationary} half")
        if moving in ("left", "right"):
            width //= 2
        else:
            height //= 2
    result = predictions[i].reshape(height, width)
    def mosaic(grid):
        return "<br>".join("".join(PALETTE[c] if c >= 0 else "❓" for c in row) for row in grid)
    return [
        "### Fold a double-sided print",
        f"On this 4×4 sheet, {', then '.join(moves)}. The two input faces share coordinates as viewed from the "
        "original front of the sheet. The rightmost grid shows the GPU’s predicted top surface.",
        "| Original front | Original back | Visible after folding |\n| --- | --- | --- |\n"
        f"| {mosaic(record['front'])} | {mosaic(record['back'])} | {mosaic(result)} |",
        f"The model returned color IDs `{result.tolist()}` for test record `folds_4[{i}]`. "
        + " ".join(f"{c} = {icon}" for c, icon in enumerate(PALETTE)) + ".",
        "Each repeated block uses the crease geometry to attend to two paper stacks. "
        "It turns the moving stack, then combines the exposed faces. The reference simulator preserves every "
        "individual paper layer. The study covers centered, axis-aligned folds of rectangular, eight-color sheets.",
    ]


def layout(words, flags):
    if not flags[-1] or not np.isin(flags, (0, 1)).all():
        return None
    lines, start = [], 0
    for end in np.flatnonzero(flags) + 1:
        lines.append(" ".join(words[start:end]))
        start = end
    return lines


def greedy_layout(words, width):
    lines = []
    for word in words:
        if lines and len(lines[-1]) + 1 + len(word) <= width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def raggedness(lines, width):
    return sum((width - len(line))**2 for line in lines[:-1])


def challenge_counts(cases, predictions, sizes):
    paragraphs = improvements = exact_layouts = 0
    elections = unique_leads = upsets = exact_ballots = 0
    for size in sizes:
        for row, pred in zip(cases[f"test/typeset/{size}"], predictions["typeset", size], strict=True):
            paragraphs += 1
            record = row["input"]
            words, width = record["words"], record["width"]
            optimal = layout(words, np.asarray(row["gold"]))
            if optimal is None:
                raise ValueError("Invalid line breaks in the reference layout")
            if raggedness(optimal, width) < raggedness(greedy_layout(words, width), width):
                improvements += 1
                exact_layouts += int(np.array_equal(pred, row["gold"]))
        for row, pred in zip(cases[f"test/ballots/{size}"], predictions["ballots", size], strict=True):
            elections += 1
            record = row["input"]
            first = Counter(ballot[0] for ballot in record["ballots"])
            leaders = [name for name, count in first.items() if count == max(first.values())]
            if len(leaders) == 1:
                unique_leads += 1
                if leaders[0] != record["films"][row["gold"][-1]]:
                    upsets += 1
                    exact_ballots += int(np.array_equal(pred, row["gold"]))
    return [
        f"In **{improvements}/{paragraphs} paragraphs**, the reference layout had strictly lower squared raggedness "
        f"than greedy wrapping. The model placed every line break correctly on **{exact_layouts}/{improvements}** "
        "of those paragraphs. Equal-cost layouts are excluded from this subset.",
        f"Of {elections} film-club elections, {unique_leads} had one film with more first-choice votes than every "
        f"other film. In **{upsets}/{elections} elections**, a different film ultimately won. The model correctly "
        f"predicted all four eliminations and the winner on **{exact_ballots}/{upsets}** of those elections. "
        "These frequencies describe the generated test cases.",
    ]


def typeset_example(cases, predictions):
    examples = []
    for i, row in enumerate(cases["test/typeset/4"]):
        record = row["input"]
        words, width = record["words"], record["width"]
        selected = layout(words, predictions[i])
        if selected is None or any(len(line) > width for line in selected):
            continue
        greedy = greedy_layout(words, width)
        cost, greedy_cost = raggedness(selected, width), raggedness(greedy, width)
        examples.append(((np.array_equal(predictions[i], row["gold"]), cost < greedy_cost,
                          -sum(map(len, words)), -i), i, width, cost, selected, greedy_cost, greedy))
    if not examples:
        return ["### Break a paragraph into lines", "The model emitted no valid four-word layout."]
    _, i, width, cost, selected, greedy_cost, greedy = max(examples)
    return [
        "### Break a paragraph into lines",
        f"Test record `typeset_4[{i}]` has a {width}-column text box. A greedy wrapper fills each line "
        "with as many words as fit. The transformer considers how each break affects the remaining lines.",
        f"| Method | Lines | Squared raggedness |\n| --- | --- | ---: |\n"
        f"| GPU prediction | {'<br>'.join(f'`{line}`' for line in selected)} | {cost} |\n"
        f"| Greedy wrapping | {'<br>'.join(f'`{line}`' for line in greedy)} | {greedy_cost} |",
        "The cost is the sum of squared unused columns on all lines before the final line. "
        "Ties favor the longest possible earlier line. Each word uses one column per ASCII character, "
        "and adjacent words are separated by one space. The synthetic words exercise line lengths and break placement.",
    ]


def ballots_example(cases, predictions):
    records = cases["test/ballots/8"]
    nominees = list(range(5))
    options = []
    for i, row in enumerate(records):
        record = row["input"]
        films, pred = record["films"], predictions[i].tolist()
        if sorted(pred) != nominees:
            continue
        ballots = [[films.index(title) for title in ranking] for ranking in record["ballots"]]
        first = Counter(ballot[0] for ballot in ballots)
        leaders = [j for j in nominees if first[j] == max(first.values())]
        priority = (pred == row["gold"], len(leaders) == 1 and pred[-1] not in leaders, -i)
        options.append((priority, i, films, pred, ballots))
    if not options:
        return ["### Choose a film-club screening", "The GPU emitted no complete five-film ranking in the eight-voter cohort."]
    _, i, films, pred, ballots = max(options)
    alive = set(nominees)
    table = ["| Round | " + " | ".join(films) + " | GPU elimination |",
             "| --- | " + " | ".join(["---:"] * len(films)) + " | --- |"]
    for idx, out in enumerate(pred[:-1], 1):
        counts = Counter(next(film for film in ballot if film in alive) for ballot in ballots)
        cells = [str(counts[film]) if film in alive else "—" for film in nominees]
        table.append(f"| {idx} | " + " | ".join(cells) + f" | {films[out]} |")
        alive.remove(out)
    return [
        "### Choose a film-club screening",
        f"In test record `ballots_8[{i}]`, eight members ranked five films. The GPU predicted "
        f"**{films[pred[-1]]}** as the winner. Recounting the ballots after each predicted elimination "
        "gives these vote transfers.",
        "\n".join(table),
        "Each ballot supports its highest-ranked remaining film. The earliest nomination among the "
        "lowest-count films loses an elimination tie. The club’s rule continues through all four "
        "eliminations, including when a film already has a majority.",
    ]


def write_report(run, output):
    receipt, results, lock, cases, preds, rows = measurements(run)
    plan = lock["plan"]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "sizes.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["task", "size", "size_unit", "parameters", "records", "exact_records", "labels",
                  "correct_labels", "nonfinite", "median_seconds", *[f"seconds_{i+1}" for i in range(plan["repeats"])]]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            entry = {k: row[k] for k in fields if k in row}
            entry.update({f"seconds_{i+1}": s for i, s in enumerate(row["seconds"])})
            writer.writerow({**entry, "size_unit": UNITS[row["task"]]})
    link = lambda label, path: f"[{label}](<{Path(os.path.relpath(path, output)).as_posix()}>)"
    large = [row for row in rows if row["size"] > max(plan["dev"]["sizes"])]
    content = [
        "# Paper folding, paragraph typesetting and film-club ballots",
        f"Three hand-programmed attention models produced **{totals(rows, 'exact_records'):,}/{totals(rows, 'records'):,} "
        f"complete outputs** and **{totals(rows, 'correct_labels'):,}/{totals(rows, 'labels'):,} individual labels** on a Kaggle "
        f"{results['device']}. A complete output requires every visible cell, line-break flag or elimination and winner to match. "
        f"At sizes above the development range, {totals(large, 'exact_records'):,}/{totals(large, 'records'):,} complete outputs matched.",
        "## Held-out results",
        f"Development used sizes {' and '.join(map(str, plan['dev']['sizes']))}. The timed runs used FP32, "
        f"one warm-up batch and {plan['repeats']} synchronized passes over each cohort of {plan['test']['records']} records, "
        f"in batches of {plan['batch_size']}. The table shows the median wall time for the entire cohort. "
        "Input preparation, host-to-device transfer, decoding and scoring were outside the timer. "
        "The stored parameter count includes zero entries and counts a shared tensor once. "
        "It excludes fixed input features, masks, indexing, and workspace. [All timings in seconds](sizes.csv)",
    ]
    table = ["| Application | Input size | Stored parameters | Exact records | Correct labels | Median ms |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        task, size = row["task"], row["size"]
        unit = f"{size}×{size} sheet" if task == "folds" else f"{size} {UNITS[task]}"
        table.append(f"| {NAMES[task]} | {unit} | {row['parameters']} | {row['exact_records']}/{row['records']} | "
                     f"{row['correct_labels']:,}/{row['labels']:,} | {1000*row['median_seconds']:.2f} |")
    content += ["\n".join(table), "## When greedy wrapping or first choices give a different answer",
                *challenge_counts(cases, preds, plan["test"]["sizes"]), "## Examples from the held-out run",
                *folds_example(cases, preds["folds", 4]), *typeset_example(cases, preds["typeset", 4]),
                *ballots_example(cases, preds["ballots", 8]), "## Reproduce the results",
                "Run these commands from the repository root. The prediction commands use NumPy and PyTorch on CPU. "
                "Replace `--example` with `--input FILE.json` to run your own record.",
                "```sh\npython creative/experiment.py predict folds --example\npython creative/experiment.py predict typeset --example\n"
                "python creative/experiment.py predict ballots --example\n```",
                f"[Private Kaggle run]({receipt['url']}) · {link('Run verification', run/'verification.json')} · "
                f"{link('Frozen weights and data manifest', HERE/'freeze.json')} · {link('Frozen cases', HERE/'cases.json.gz')} · "
                f"{link('Per-record predictions', run/'output/creative-predictions.npz')} · {link('Full GPU results', run/'output/creative-results.json')}",
                "To recompute the expected answers and regrade all published predictions with NumPy, run the audit from the repository root. [Audit results](audit.json)",
                f"```sh\npython creative/audit.py kaggle/runs/creative/v{receipt['version']} audit-output/creative-audit.json\n```"]
    (output / "report.md").write_text("\n\n".join(content) + "\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(rows)} verified measurements in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version_folder", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    write_report(args.version_folder.resolve(), args.destination.resolve())
