# Looped weight-space programming on Kaggle

A hand-written transformer matched **1,000 of 1,000 official Maze-Hard test grids**, including the benchmark’s choice between equally short paths. This repository contains the analytic weights, scoring code and GPU results for a Codex-assisted reproduction of selected LoopedWSP experiments. It also includes three new synthetic applications that you can run locally. [Maze test result](results/maze/test.json) · [Application examples](results/applications/report.md)

The original LoopedWSP study asked Claude to write the weights of a transformer that solves a task by repeatedly applying the same block. For this reproduction, Codex had access to the published method and task briefs. We constructed fresh tensors from the rules and ran frozen models on Kaggle Tesla T4 hardware. These experiments measure the resulting models’ accuracy. Measuring unaided discovery, authoring cost, or performance relative to training would require separate experiments. [Original study](https://mbzuai-oryx.github.io/LoopedWSP/) · [Authors’ repository](https://github.com/mbzuai-oryx/LoopedWSP)

## Reproduction results

| Experiment | Verified outcome | Evidence |
| --- | --- | --- |
| Maze-Hard, 30×30 | 1,000/1,000 development grids and 1,000/1,000 official test grids matched at all three loop budgets. The network has 7,003 scalar entries, including 235 nonzero. | [Architecture](src/maze/EXPLAIN.md) · [DEV](results/maze/dev.json) · [TEST](results/maze/test.json) |
| Larger mazes | 50/50 exact grids at 60×60 and 40/40 at 100×100. The recurrent weights stayed fixed. Two analytic position tables were extended for each size. | [60×60](results/maze/scale-60.json) · [100×100](results/maze/scale-100.json) |
| Five parameter limits | Each model matched all 1,000 official test grids. Under caps of 1k, 5k, 10k, 20k and 40k, the builds used 965, 2,801, 4,459, 5,770 and 30,139 scalar entries respectively, counting zeros. | [Build details](caps/README.md) · [Scores and tensor hashes](results/maze/caps/summary.json) |
| Length generalization | Graph search 576/576 through 48 nodes, parity 704/704 through 4,095 bits, and shuffled reasoning chains 512/512 through 63 steps. All 1,184 examples beyond the development ranges were correct. | [Per-size results and timings](results/lengths/report.md) |
| Sequential puzzles | Tower of Hanoi passed 20/20 sizes, including all 1,048,575 moves for 20 disks. Checker Jumping passed 40/40 sizes, including 1,680 moves at 40 checkers per side. Independent replay confirmed legality, the final states and optimal move counts. | [Puzzle results](results/puzzles/report.md) · [Replay audit](results/puzzles/replay-audit.json) |

The larger-maze and length-generalization evaluations use seeded replacement corpora. The puzzle models use the move index, which permits several positions to be decoded together. The checker size embedding includes the known optimal length. Those choices matter when comparing results or timings with the original study. The historical plan identifies broader work such as Sudoku, GraphWalks, training baselines and blind interpretability experiments. [Data provenance](DATA.md) · [Historical study plan](STUDY-PLAN.md)

## Three new applications

Each application has its own frozen transformer. Development used records of length 8 and 16. A single held-out GPU evaluation covered 64 records per task at each of five lengths, from 8 through 128. All **960/960 complete records** and **47,616/47,616 individual labels** matched independent scoring rules. Every prediction remained stable at the three loop budgets. These tests cover the specified synthetic rules and input ranges. [Examples and scores by size](results/applications/report.md) · [Independent verification](results/applications/verification.json)

| Model | One output for each input | Held-out records |
| --- | --- | ---: |
| [Inventory](applications/inventory.py) | Flag a negative stock balance after applying each signed transaction to the opening balance. | 320/320 |
| [Booking conflicts](applications/scheduling.py) | Flag every booking that overlaps another for the same resource. Touching endpoints are compatible. | 320/320 |
| [Sensor alarms](applications/telemetry.py) | Switch on at 70 or above, switch off at 30 or below, and keep the previous alarm state in between. | 320/320 |

Across the 15 task-size settings, GPU model inference took 0.710 seconds in total. The conventional CPU scoring rules produced the same labels in 0.043 seconds on the Kaggle host. [Timing breakdown](results/applications/sizes.csv)

## When does looping help?

On a Kaggle Tesla T4, the same frozen weights went from **155/960** exact published records after one pass to **381/960** after four and **960/960** at full depth in both FP32 and FP64. For 240 new pairs in which changing the first item flips the final answer, one pass got both answers correct on **0/240** pairs. Full depth and task-specific global feedforward models each got **240/240**. The global baselines use a causal prefix mean for inventory, pairwise overlap checks for bookings, and attention to the latest decisive sensor reading. They also passed all 1,440 records in FP16. All three original models encountered nonfinite intermediate values in FP16 at full depth.

Looping helped the original blocks carry information beyond the positions they could reach in one pass. A feedforward computation with direct access to the relevant inputs also solved these tasks, using 3, 16 and 4 stored scalar coefficients respectively. Those counts exclude the fixed operations and masks. The study covers 315 settings at lengths 8–128 with equal-parameter depth comparisons, local-window controls, and three timed repeats per setting. [Full comparison and accuracy plot](results/loop-comparison/report.md) · [All measurements](results/loop-comparison/sizes.csv) · [Parameter budgets](results/loop-comparison/parameter-budgets.csv) · [Protocol frozen before the run](https://github.com/ajayasai/loopedwsp-reproduction/tree/04111d2a806743ae4011bfb4a4310f28591cc867/comparisons)

## Run a frozen model

Use Python 3.12 or later from the repository root. The local predictor uses NumPy on CPU. Install the pinned CPU and report dependencies with `python -m pip install -r requirements.txt`, then run this example in Python.

```python
from applications.predict import predict

result = predict("inventory", {"opening": 3, "changes": [-3, -1, 4]})
print(result["negative_balance"])
# [False, True, False]
```

The three successive balances are 0, −1 and 3. The same predictor accepts a raw JSON file or standard input. For example, save `{"readings":[50,70,52,30]}` as `readings.json` and run `python applications/predict.py telemetry readings.json` to get `{"alarm_on": [false, true, true, false]}`. GPU inference additionally requires a CUDA-enabled PyTorch installation and a supported GPU. [Predictor and input bounds](applications/predict.py)

## Verify the archived runs

From the repository root, the following commands run the maze unit tests, replay the downloaded puzzle moves, and independently check the application labels against the archived Kaggle predictions. The audit commands write fresh JSON files under `audit-output/`.

```sh
python -m unittest discover -s src/maze -p test_weights.py -v
python puzzles/audit_outputs.py kaggle/runs/sequential-puzzles/v1/output audit-output/puzzles.json
python applications/audit.py kaggle/runs/practical-applications/v1 audit-output/applications.json
```

To repeat the Maze-Hard training-label audit, first run `python scripts/fetch_maze.py` to download the pinned dataset and then `python src/maze/audit.py`. The repository includes scoring artifacts from the private Kaggle runs so readers can inspect results locally. [Dataset provenance and acquisition](DATA.md) · [Maze-label audit](results/maze/uniqueness.json) · [Run log](logs/progress.md)

## Repeat a GPU experiment

To submit a new private Kaggle run, use a fresh checkout, authenticate the Kaggle CLI and set the environment variable `LOOPEDWSP_KAGGLE_OWNER` to your Kaggle username. The numbered run folders in the checkout contain the original receipts. In a shell with that variable set, `python submit.py dev --push` runs the headline development split. For the three applications, use `python applications/submit_applications.py --push`. Omitting `--push` builds the submission locally. [Maze launcher](submit.py) · [Application launcher](applications/submit_applications.py)

## License and attribution

The reproduction code, new model weights, and synthetic artifacts are released under MIT. The original LoopedWSP authors retain their copyright in the adapted material. The official Maze-Hard data is obtained separately from its publisher. [Project license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Data downloads](DATA.md)
