# Contour maps, ancestral plumage and quantum circuits

These three hand-programmed transformers take ordinary JSON inputs. You can trace a contour across an elevation grid, find the feather patterns compatible with a minimum-change history of a tree, or predict the outcome statistics for a Pauli measurement on an ideal three-qubit circuit. [Measured results by size](../results/frontier/report.md)

## Try the examples

From the repository root with Python 3.12 or later, NumPy and PyTorch installed, run any of these commands on CPU.

```sh
python frontier/experiment.py predict contours --input frontier/examples/contours.json
python frontier/experiment.py predict plumage --input frontier/examples/plumage.json
python frontier/experiment.py predict clifford --input frontier/examples/clifford.json
```

| Example | What to look for |
| --- | --- |
| [Elevation map](examples/contours.json) | The level-100 contour crosses all four edges of three adjacent cells. The outer cells connect their edges in opposite pairs. In the middle cell the branches meet at a four-way saddle. [Map inputs and edge names](contours_notes.md) |
| [Bird plumage](examples/plumage.json) | The first two tips, plain and barred, leave their shared ancestor ambiguous in isolation. The other two tips are both plain. With the whole tree present, plain is the sole optimal state for all three ancestors. [Tree format and cost rule](plumage_notes.md) |
| [Bell pair](examples/clifford.json) | Either qubit gives a random Z measurement on its own. Their joint Z product is always +1 and their joint Y product is always −1. The six requested expectations are `[1,-1,1,0,0,1]`. [Gates and measurement convention](clifford_notes.md) |

Replace the path with your own JSON file, use `--example` to see a built-in input alongside its answer, or omit `--input` to read from standard input.

## Evaluation

| Task | Stored FP32 coefficients | Development sizes | Held-out sizes | Size measures |
| --- | ---: | --- | --- | --- |
| Contour topology | 54 | 3, 6 | 3, 6, 12, 24 | Map cells along each side |
| Ancestral plumage | 5 | 4, 8 | 4, 8, 12, 16 | Observed tips on a supplied binary tree |
| Clifford measurements | 1,025 | 4, 8 | 4, 8, 16, 32 | Gates in a three-qubit circuit |

The seed, weights, records, labels and scoring rules were frozen before the GPU run. Each model has 16 development records and 256 held-out records. An exact record requires a correct prediction for every cell, every internal node’s complete set of states, or all eight measurement expectations. Thirteen worked checks are scored separately. The parameter count includes zeros and counts reused coefficients once. Fixed operations and working memory also contribute to each calculation. [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/c2d04db/frontier) · [Protocol](protocol.json) · [Worked checks](checks.json) · [Model and data hashes](freeze.json)

The fields, trees and circuits are synthetic. Contour connectivity assumes bilinear interpolation between samples. Ancestral states describe equally parsimonious reconstructions on the supplied tree. Circuit predictions describe separate measurements on fresh preparations of the ideal state. Each test circuit has four deterministic and four unbiased nonidentity queries selected using the reference and shuffled.

## Verify the public run

The independent audit requires NumPy. It regrades the saved predictions by testing bilinear determinants, recalculating tree costs with each ancestor as the root, and transporting the initial stabilizer group forward through each circuit. The report renderer also requires Matplotlib.

```sh
python frontier/audit.py kaggle/runs/frontier/v1 audit-output/frontier-audit.json
python frontier/report.py kaggle/runs/frontier/v1 results/frontier
```

To run the development set on CPU, use `python frontier/experiment.py dev`. To repeat the GPU benchmark on Kaggle, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your username, and run `python frontier/submit_study.py --push`. After the run completes, collect the archive with `python frontier/submit_study.py --collect VERSION`. [GPU launcher](submit_study.py) · [Archived predictions](../kaggle/runs/frontier/v1/output/frontier-predictions.npz)
