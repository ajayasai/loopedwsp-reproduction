# Tree-ring alignment, loom drafting and knot coloring

These three applications use attention with hand-written weights. Each accepts one JSON record and runs on CPU or can be evaluated on a Kaggle GPU. The benchmark inputs, expected outputs and weights were published before GPU evaluation. [Frozen snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/2480f9e/atlas) · [GPU results and timings](../results/atlas/report.md)

## Try the three examples

With Python 3.12 or later, NumPy and PyTorch installed, run these commands from the repository root.

```sh
python atlas/experiment.py predict rings --input atlas/examples/rings.json
python atlas/experiment.py predict loom --input atlas/examples/loom.json
python atlas/experiment.py predict braid --input atlas/examples/braid.json
```

| Input | Expected result | Change the input |
| --- | --- | --- |
| An eight-ring sample whose widths are twice a reference window’s widths plus three | The unique best window starts at **offset 2**, counting from zero. | Set a width to `null` to mark an unmeasured year. The model returns every tied best offset and flags windows with too few comparable changes. [Tree-ring guide](rings_notes.md) |
| A four-shaft draft for a two-over, two-under twill | Eight rows of cloth form diagonal lines. The longest exposed warp spans **2 picks** and the longest exposed weft spans **2 ends**. | Press several treadles on a pick to combine the shafts they lift. [Loom guide and expected pattern](loom_notes.md) |
| A trefoil described by the signed three-strand braid word `[1,1,1,2]` | **9** valid colorings, of which **6** use more than one color. | Try the empty word `[]`. Closing its three strands gives three unlinked circles with **27** colorings. [Braid guide](braid_notes.md) |

Replace the path with a JSON file of your own, omit `--input` to read standard input, or use `--example` to print the built-in input and its prediction. [Example files](examples)

## What the benchmark measures

| Application | Stored FP32 coefficients | Development sizes | Held-out sizes | Unit of size |
| --- | ---: | --- | --- | --- |
| Tree rings | 11 | 12, 16 | 12, 16, 24, 40 | Rings in the reference series |
| Loom | 4 | 4, 8 | 4, 8, 16, 32 | Warp ends and weft picks in a square draft |
| Braids | 50 | 4, 8 | 4, 8, 16, 32 | Crossings in the braid word |

Each task has eight development records per development size and 64 held-out records per test size. The evaluator counts a record as exact when every requested output matches. Thirteen named checks are scored separately. We count stored coefficients, including zeros. Fixed features, operations and working memory also do task-specific work. For example, the braid model carries all 27 possible initial color triples through the crossings.

The synthetic tree-ring inputs test an alignment rule for already indexed annual positions. The loom uses an ideal rising shed. The knot model counts a link invariant that several distinct knots share. [Protocol](protocol.json) · [Inputs and targets](cases.json.gz) · [Worked checks](checks.json) · [Tensor and data hashes](freeze.json)

## Verify or rerun

The independent audit reads the public Kaggle archive, rebuilds the targets from the raw records with NumPy, and checks every saved prediction. The report renderer also requires Matplotlib.

```sh
python atlas/audit.py kaggle/runs/atlas/v1 audit-output/atlas-audit.json
python atlas/report.py kaggle/runs/atlas/v1 results/atlas
```

To evaluate locally, run `python atlas/experiment.py dev` on CPU or `python atlas/experiment.py gpu --output audit-output/atlas-gpu` on a CUDA device. For Kaggle, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your username and run `python atlas/submit_study.py --push`. Collect the completed run with `python atlas/submit_study.py --collect VERSION`. [Launcher](submit_study.py) · [Archived predictions](../kaggle/runs/atlas/v1/output/atlas-predictions.npz)
