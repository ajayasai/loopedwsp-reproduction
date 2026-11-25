"""Render the frontier application study from the independently audited GPU archive."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np

from audit import evaluate


HERE = Path(__file__).resolve().parent
TASKS = {"contours": "Contour connectivity", "plumage": "Ancestral plumage", "clifford": "Clifford measurements"}
CONTOUR = ("empty", "N–E", "E–S", "S–W", "W–N", "N–S", "E–W", "N–E and S–W", "W–N and E–S", "Four-way saddle")
PLUMAGE_CHECKS = {"plumage_outgroup_resolves": "Other branches resolve the first pair",
                  "plumage_three_way_tie": "Three different tips", "plumage_root_tie": "Two matching pairs",
                  "plumage_unchanged": "Two spotted tips"}
CIRCUIT_CHECKS = {"bell_correlations": "Bell pair", "bell_gates_reversed": "Bell gates in reverse order",
                  "phase_ghz": "GHZ state with a relative phase", "inverse_phase": "Inverse phase gate",
                  "vacuum_expectations": "Initial state, no gates"}


def ratio(a, b):
    return f"{a:,}/{b:,}"


def table(header, rows):
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |"
                     for row in (header, ["---"]*len(header), *rows))


def timing_csv(path, rows, protocol):
    fields = ["parameters","records","exact_records","labels","correct_labels","nonfinite","median_seconds"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["task","size","unit",*fields,*[f"seconds_{i+1}" for i in range(protocol["repeats"])]])
        for row in rows:
            unit = protocol["tasks"][row["task"]]["size_unit"]
            writer.writerow([row["task"],row["size"],unit,*[row[field] for field in fields],*row["seconds"]])


def contour_figure(path, record, predictions):
    """Draw the GPU edge pairings on the bilinear surface of each worked cell."""
    level, heights = record["level"], np.asarray(record["heights"])
    fig, axes = plt.subplots(1,len(predictions), figsize=(9.4,3.6))
    fig.subplots_adjust(top=.72, bottom=.13, left=.045, right=.955, wspace=.22)
    axis = np.linspace(0,1,201)
    xx,yy = np.meshgrid(axis,axis)
    for i, (ax,label) in enumerate(zip(axes,predictions,strict=True)):
        tl,tr,bl,br = heights[:,i:i+2].ravel()
        a,b,d,c = (int(v-level) for v in (tl,tr,bl,br))
        surface = (1-yy)*((1-xx)*a+xx*b) + yy*((1-xx)*d+xx*c)
        ax.imshow((surface>0).astype(float), extent=(0,1,1,0), origin="upper", interpolation="nearest",
                  cmap=ListedColormap(["#f7ead1","#c4e2de"]), vmin=0, vmax=1)
        ax.add_patch(Rectangle((0,0),1,1, fill=False, linewidth=1, edgecolor="#576366"))
        crossings = {"N":(a/(a-b),0), "S":(d/(d-c),1), "W":(0,a/(a-d)), "E":(1,b/(b-c))}
        if label in (7,8):
            pairs = ("NE","SW") if label == 7 else ("WN","ES")
            for first,second in pairs:
                x1,y1 = crossings[first]
                x2,y2 = crossings[second]
                x = np.linspace(x1,x2,240)
                top, bottom = (1-x)*a+x*b, (1-x)*d+x*c
                # Along the contour, (1-y)*top + y*bottom = 0.
                denominator = top-bottom
                if np.any(denominator == 0) or np.min(denominator)*np.max(denominator)<0:
                    ax.plot([x1,x2],[y1,y2],"--", color="#b74a2e", lw=2)
                else:
                    ax.plot(x,top/denominator, color="#243b53", lw=2.5)
        elif label == 9:
            for first,second in ("NS","WE"):
                ax.plot(*zip(crossings[first],crossings[second]), color="#243b53", lw=2.5)
            ax.scatter(*[crossings["N"][0],crossings["W"][1]], s=37, c="#243b53", zorder=5)
        for x,y in crossings.values():
            ax.plot(x,y,"o",ms=4.5, color="#243b53")
        for x,y,z in ((0,-.075,tl),(1,-.075,tr),(0,1.09,bl),(1,1.09,br)):
            ax.text(x,y,str(z),ha="center",va="center",fontsize=10,color="#26333a")
        ax.set(xlim=(-.15,1.15),ylim=(1.16,-.14),aspect="equal")
        ax.axis("off")
        ax.set_title(CONTOUR[label], fontsize=13, pad=6)
    fig.suptitle(f"Tracing the elevation-{level} contour", fontsize=17, y=.96)
    fig.text(.5,.055,"Corner numbers are sampled elevations. Teal is above the contour, cream below. North is up.",
             ha="center",fontsize=10,color="#354451")
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def states(mask):
    return " / ".join(state for i,state in enumerate(("plain","barred","spotted")) if mask & 1 << i)


def render(run, dest):
    audit = evaluate(run)
    protocol = json.loads((HERE / "protocol.json").read_bytes())
    cases = {c["name"]:c["input"] for c in json.loads((HERE / "checks.json").read_bytes())}
    with np.load(run / "output/frontier-predictions.npz", allow_pickle=False) as raw:
        worked = {name:raw["check_"+name].tolist() for name in cases}
    rows = audit["conditions"]
    dest.mkdir(parents=True, exist_ok=True)
    timing_csv(dest/"sizes.csv",rows,protocol)
    name = "three_adjacent_saddles"
    contour_figure(dest/"contour-topology.png",cases[name],worked[name])

    def score(task, category):
        return ratio(audit["correct_labels_by_category"][task].get(category,0),
                     audit["labels_by_category"][task].get(category,0))

    def total(rows, a, b):
        return ratio(sum(r[a] for r in rows),sum(r[b] for r in rows))

    disjoint = [r for r in rows if r["size"] > max(protocol["tasks"][r["task"]]["dev"])]
    checks = audit["worked_checks"]
    sizes = table(["Application","Size","Stored FP32 coefficients","Exact records","Correct labels","Median GPU time (ms)"],
                  [[TASKS[r["task"]],r["size"],r["parameters"],ratio(r["exact_records"],r["records"]),
                    ratio(r["correct_labels"],r["labels"]),f"{r['median_seconds']*1000:.2f}"] for r in rows])
    contour_checks = {"three_adjacent_saddles":"Three adjacent alternating cells", "hilltop_loop":"One high hilltop",
                      "mean_misroutes_saddle":"Center height favors the other pairing", "straight_across":"Straight contour"}
    contour_rows = table(["Worked map","Level","GPU cell connections, in row order"],
                         [[description,cases[key]["level"],", ".join(CONTOUR[label] for label in worked[key])]
                          for key,description in contour_checks.items()])
    plumage = table(["Worked tree","Observed tips in input order","Predicted optimal states per internal node, root last"],
                    [[description,", ".join(cases[key]["tips"]),
                      ", ".join(f"{node} ({states(mask)})" for node,mask in enumerate(worked[key],len(cases[key]["tips"])))]
                     for key,description in PLUMAGE_CHECKS.items()])
    quantum = table(["Worked circuit","Requested observables in qubit order 0, 1, 2","GPU expectations in query order"],
                    [[description,", ".join(cases[key]["observables"]),
                      ", ".join(["−1","0","+1"][value] for value in worked[key])] for key,description in CIRCUIT_CHECKS.items()])
    c = audit["labels_by_category"]["contours"]
    with gzip.open(HERE/"cases.json.gz","rt",encoding="utf-8") as f:
        corpus = json.load(f)
    def gate_lists(split):
        return [tuple(map(tuple, item["input"]["gates"])) for n in protocol["tasks"]["clifford"][split]
                for item in corpus[f"{split}/clifford/{n}"]]
    test_gates, dev_gates = gate_lists("test"), set(gate_lists("dev"))
    repeated = sum(word in dev_gates for word in test_gates)
    src = f"../../kaggle/runs/frontier/{run.name}"
    report = [
        "# Contour connectivity, ancestral plumage and Clifford measurement outcomes",
        f"Three hand-programmed transformers matched their independent references on "
        f"**{total(rows,'exact_records','records')} complete held-out records** on a Kaggle {audit['run']['device']}. "
        f"They matched **{total(rows,'correct_labels','labels')} individual outputs** and "
        f"{total(disjoint,'exact_records','records')} complete records at sizes above the development range. "
        f"The separate worked checks passed on {ratio(sum(v['correct'] for v in checks.values()),len(checks))} inputs.",
        "## Accuracy and runtime by size", sizes,
        "Size counts cells on each side of the square map, observed tips of a rooted tree, or gates in a "
        "three-qubit circuit. Exact scoring requires every cell’s edge connections, every internal node’s full "
        "set of optimal states, or all eight Pauli expectations in the record to match.",
        f"Each time is the median of {protocol['repeats']} synchronized FP32 forward sweeps over "
        f"{protocol['per_size']['test']} GPU-resident records in batches of {protocol['batch']}, after warm-up. "
        "Timing excludes encoding, transfers, decoding and verification. Stored coefficient counts include zeros "
        "and count shared weights once. The models also use fixed operations, addressing features and working "
        "memory. [All timing repeats](sizes.csv)",
        "## Cartography · tracing equal-elevation lines",
        "![GPU contour connections on three adjacent cells at the same elevation](contour-topology.png)",
        "All three worked cells alternate above and below elevation 100 around their corners. The GPU predicts the "
        "edge connections. The figure places the crossings and curves using bilinear interpolation. An exact "
        "saddle joins all four edges at one point. The other two cells show the two possible pairings.", contour_rows,
        f"The model matched {score('contours','four_edge_cells')} four-edge cells, including "
        f"{score('contours','exact_saddles')} exact saddle junctions. Choosing a pairing from the cell-center "
        f"elevation would disagree with the bilinear rule on {ratio(c.get('center_wrong',0),c.get('center_comparable',0))} "
        f"comparable cells. Both the center height relative to the level and the bilinear determinant are nonzero "
        f"in this comparison. The model matched {score('contours','center_wrong')} of the disagreements. "
        "The target is per-cell connectivity on a synthetic grid. [Runnable contour example](../../frontier/contours_notes.md)",
        "## Evolutionary biology · reconstructing a hypothetical feather pattern",
        "Each edge on which a pattern changes costs one. A state belongs to an ancestor’s output if some "
        "minimum-cost assignment on the entire tree gives that ancestor the state. At the first ancestor of "
        "the four-tip example, its plain and barred offspring allow either of those states if considered alone. "
        "The plain tips on the other branch resolve the global optimum to plain.", plumage,
        f"The model matched {score('plumage','ambiguous_ancestors')} ancestors with several equally optimal "
        f"patterns, including {score('plumage','all_three_states')} for which all three patterns were possible. "
        f"It matched {score('plumage','outside_changes_set')} ancestors whose optimal set changed when the rest "
        "of the tree was included. These are synthetic observations on supplied trees. States selected separately "
        "from two ancestors’ sets can belong to different optimal histories. [Runnable plumage example](../../frontier/plumage_notes.md)",
        "## Quantum information · predicting Pauli measurement outcomes",
        "Each ideal circuit starts in the three-qubit all-zero state. A Pauli expectation of +1 or −1 "
        "makes the measurement outcome certain. Zero gives equal probabilities for the two outcomes. "
        "In the Bell example the two single-qubit Z measurements are unbiased, while their product is "
        "always +1. Reversing the two gates gives zero expectation for that product.", quantum,
        f"The model matched {score('clifford','positive')} positive, {score('clifford','negative')} negative "
        f"and {score('clifford','random')} unbiased query results. In {score('clifford','definite_joint_random_marginal')} "
        "queries it identified the definite sign of a product with at least one unbiased single-qubit factor. "
        "The generator samples four deterministic and four unbiased nonidentity queries for each circuit and "
        "shuffles them, giving each category half the test queries by design. The model receives only the "
        "gates and unsigned Pauli words. A record comprises both its gate list and its queries. The "
        f"{len(test_gates)} held-out records contain {len(set(test_gates))} distinct gate lists. "
        f"A development gate list recurs with different queries in {ratio(repeated,len(test_gates))} records. These are "
        "predictions for ideal gates computed on a classical GPU. "
        "[Runnable Bell example](../../frontier/clifford_notes.md) · [Corpus provenance](../../DATA.md)",
        "## Reproduce the study",
        f"[Model guide](../../frontier/README.md) · [Worked cases](../../frontier/checks.json) · "
        f"[Frozen protocol](../../frontier/protocol.json) · [Independent audit](audit.json) · "
        f"[Saved predictions]({src}/output/frontier-predictions.npz) · [GPU verification]({src}/verification.json) · "
        f"[Submission receipt]({src}/submission.json) · [Private Kaggle run]({audit['run']['url']})",
        "```sh\npython frontier/experiment.py predict contours --example\npython frontier/experiment.py predict plumage --example\n"
        "python frontier/experiment.py predict clifford --example\n"
        f"python frontier/audit.py kaggle/runs/frontier/{run.name} audit-output/frontier-audit.json\n"
        f"python frontier/report.py kaggle/runs/frontier/{run.name} results/frontier\n```",
    ]
    (dest/"report.md").write_text("\n\n".join(report)+"\n", encoding="utf-8", newline="\n")
    print(f"Rendered {len(rows)} settings and {len(checks)} worked cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    render(args.run,args.output)
