# Reproducing Looped Weight-Space Programming on Kaggle

The first experiment asks whether freshly written transformer weights can reproduce the official path on 1,000 held-out 30×30 Maze-Hard grids. Each grid has 900 output cells. One incorrect cell makes the entire grid incorrect. This repository contains that experiment and the plan for the rest of the study.

The published study provided the task, architecture constraints, and broad algorithm. We used the authors' description and first prompt, audited the official training labels, and constructed new tensors. Our builder had access to the public method description. The authors’ original session and our run have different information, hardware, and accounting. A controlled experiment on what a model discovers unaided would use fresh isolated sessions starting from the brief alone. [Study](https://mbzuai-oryx.github.io/LoopedWSP/) · [First prompts at the source revision](https://github.com/mbzuai-oryx/LoopedWSP/tree/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/prompts)

## Completed Kaggle runs

| Official split | Full grids matched | FP64 inference on Tesla T4 | Private run |
| --- | ---: | ---: | --- |
| DEV (train.csv) | 1,000 / 1,000 | 28.43 s | [DEV v1](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-independent-maze-dev/versions/1) |
| Held-out TEST (test.csv) | 1,000 / 1,000 | 27.45 s | [TEST v1](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-independent-maze-test/versions/1) |

Both runs used the same 7,003 scalar parameters, 235 of them nonzero. The model matched the reference grid on every example at each of the three loop budgets. All 16 TEST batches reached an exact fixed point within 320 executed blocks. The times above cover inference, including attention preparation and GPU synchronization. Kaggle scheduling, setup, downloads and the backend comparison are outside that timing.

The run used one full development evaluation and one held-out evaluation. Both private Kaggle receipts carry the same source bundle hash, and the result files carry the frozen tensor hash. [Aggregate DEV results](results/maze/dev.json) · [Aggregate TEST results](results/maze/test.json) · [30×30 weight archive](results/maze/weights.npz) · [Pre-TEST freeze](results/maze/freeze.json) · [Run log](logs/progress.md)

## Larger-maze results

The frozen transformer solved every reference grid in a private FP64 Tesla T4 run. The study's released maze generator supplied 50 fresh 60×60 grids and 40 fresh 100×100 grids. We fixed the seeds and verified the reference paths before submission. These samples reproduce the study's sizes and counts; the original 90 inputs and seeds were unavailable. [Generator](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/weights/run.py) · [Locked corpus hashes](scale/corpus-lock.json) · [Private Kaggle job](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-independent-maze-scale/versions/1)

| Size | Exact full-grid matches | Total inference | Time per maze, amortized | Shortest-path length, min–max |
| --- | ---: | ---: | ---: | ---: |
| 60×60 | 50 / 50 (100%) | 13.72 s | 0.274 s | 1–400 moves |
| 100×100 | 40 / 40 (100%) | 73.98 s | 1.849 s | 3–1,090 moves |

The benchmark loaded the original weight archive and extended its two analytic position tables. Times cover synchronized inference in batches of up to eight, including attention setup and prediction transfer. Corpus generation, reference scoring and Kaggle startup are outside the timed region. All batches reached exact full-state equality before their loop budgets of 7,200 and 20,000. [60×60 report](results/maze/scale-60.json) · [100×100 report](results/maze/scale-100.json)

## Headline experiment

1. Pin the HRM dataset to revision `549754de3d67eebe57b6b22c7d226b9412d55b07`. Audit only `train.csv`. For every grid, verify that the labeled cells form a shortest path, count shortest paths, and check all 24 neighbor orders in both tracing directions. Use these results only to establish the labeling convention. [Official dataset](https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k)
2. Construct the embeddings, four neighbor-reading attention heads, ReLU layers, and linear output by formula. A wave spreads from the goal. At first arrival a cell remembers the highest-priority neighbor a step closer to the goal. A path bit travels from the start along those pointers. Python performs character encoding, a generic tied block, and a per-cell readout. The independent BFS checker lives in the audit script. [Architecture and tensor layout](src/maze/EXPLAIN.md) · [Weight builder](src/maze/weights.py)
3. Check dense NumPy inference on hand-built tie, edge and disconnected examples. Compare FP64 CUDA logits to the literal NumPy logits on synthetic inputs before accessing the evaluation split in each Kaggle job. A cached head uses indexed reads only when the computed softmax contains exactly one unit entry per row and zero elsewhere. [Runtime](src/maze/looped.py) · [GPU equivalence check](src/maze/check_backend.py)
4. Run DEV in a private Kaggle GPU session. Record exact-grid scores at 1,800, 3,600, and 7,200 tied blocks. The first budget covers a wave and a path across the entire board. If applying the complete block leaves the entire batch's hidden state exactly equal to its previous value, all subsequent applications produce the same state. Record the actual number of evaluated blocks. Cap full DEV attempts at 20 and log each attempt.
5. Review the analytic weights, data separation, reference equivalence and DEV results. Freeze the source bundle and tensor hash. Run the same model once on all 1,000 official TEST examples in a separate private Kaggle job, returning aggregate scores. Compare DEV and TEST hashes and preserve both versioned outputs. The reproduction target is 1,000/1,000 exact grids with all 1,000 readouts stable across the three budgets. [Original brief](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/prompts/01_maze_hard_network_brief.txt)

## Parameter-cap results

Five freshly constructed networks scored 1,000/1,000 exact official test grids apiece. For limits of 1,000, 5,000, 10,000, 20,000 and 40,000 parameters, the actual dense counts were 965, 2,801, 4,459, 5,770 and 30,139. [Builds, evaluation details and Kaggle runs](caps/README.md) · [Combined results](results/maze/caps/summary.json)

## Length-generalization results

Three freshly written looped transformers answered all 1,792 seeded held-out examples correctly on a private Kaggle Tesla T4. The 1,184 examples beyond the development ranges included 24–48-node directed graphs with unreachable targets and paths of 8–16 hops, parity through 4,095 bits, and shuffled reasoning chains through 63 operations. The report gives accuracy and synchronized inference time for each of the 25 sizes and a separate breakdown by graph distance. [Results by size](results/lengths/report.md) · [CSV](results/lengths/results.csv) · [Private Kaggle run](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-length-generalization/versions/1)

## Sequential puzzle results

Two fresh looped transformers solved all 20 Tower of Hanoi sizes and all 40 Checker Jumping sizes on a private Kaggle Tesla T4. Two independent simulators replayed all 2,120,910 emitted moves and confirmed every final state and optimal count. For 20 disks the model emitted 1,048,575 moves in 0.723382 seconds. For 40 checkers per side it emitted 1,680 moves in 0.013087 seconds. These are batched decoding times. The models use the move index, and the checker size embedding includes the known optimal length. [All 60 sizes and timing method](results/puzzles/report.md) · [Full-precision CSV](results/puzzles/sizes.csv) · [Downloaded move replay](results/puzzles/replay-audit.json) · [Private Kaggle run](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-sequential-puzzles/versions/1)

## Inventory, booking conflicts and sensor alarms

Three fresh looped transformers labeled every transaction, booking and sensor reading correctly in 960 held-out synthetic records on a private Kaggle Tesla T4. Each task contributed 320 records at lengths 8, 16, 32, 64 and 128. The 576 records longer than the development range were all exact. Inference across the five sizes took 0.091832 seconds for inventory, 0.434181 seconds for booking conflicts and 0.183564 seconds for alarm states. An independent replay checked all 47,616 decisions and the NumPy model predictions matched the GPU at every size. [Examples and results by size](results/applications/report.md) · [JSON predictor](applications/predict.py) · [Independent verification](results/applications/verification.json) · [Private Kaggle run](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-practical-applications/versions/1)

## Full-study execution plan

Each additional build gets a fresh authoring session, its original brief where available, a pinned dataset or generator, development-only examples, and an independent grader. Freeze the model, tokenizer, loop budget and evaluator before scoring a holdout. For a benchmark with multiple valid answers, state exactly which equivalence the grader accepts. Log failed builds, development calls, dense and nonzero parameter counts, wall time, peak memory, accelerator and token usage when available. Release one results table with source, split, metric, numerator, denominator and artifact hash for each claim.

| Stage | Build or comparison | Evaluation and deciding evidence |
| --- | --- | --- |
| 1. Maze scaling and parameter limits | Keep the recurrent weights fixed, extend analytic position features, then evaluate 50 generated 60×60 and 40 generated 100×100 mazes. Run five fresh builds under caps of 1k, 5k, 10k, 20k and 40k scalar entries, including all zeros. | Fix generator, tie order and seeds before scoring. Report exact grids, valid shortest paths, steps, runtime and memory at each size. Check attention in chunks at 100×100. The authors report a 928-weight model in the 1k-cap condition. |
| 2. Length generalization | Implement graph shortest-distance search, bit-string parity, and shuffled arithmetic reasoning chains. Match the published checking ranges (graph routes up to 4, parity up to 20 bits, chains up to 8 steps). | Hold out longer routes and unreachable nodes, parity through 4,095 bits, and chains through the paper's longer ranges. Report accuracy by size and after extra loops, alongside the trained controls. |
| 3. Sequential puzzles | Author Tower of Hanoi and Checker Jumping models from their puzzle rules. Use a separate simulator to replay outputs. | Hanoi must finish all sizes 1–20 legally in the optimum number of moves, including 1,048,575 at size 20. Check optimal solutions through 40 checkers per side. Save first failure and count legal moves in aggregate. |
| 4. Sudoku-Extreme | Write a model with candidate elimination, decisions, and backtracking. Run a separate build capped at 1,000 loops. | Report exact completed boards, solve depth and loop use under the published budgets. Score the first 10,000 for comparison and the published 141,040-puzzle workload separately. Profile a development batch before allocating the full sweep. |
| 5. GraphWalks | Write the graph-propagation model, preserve the task's input and output encoding, and keep DEV and TEST graphs disjoint. | Score the 200 long-context BFS problems using the official answer extraction and F1. Profile the largest case for memory and verify any sparse/chunked attention implementation against the dense operator on small inputs. |
| 6. BIG-Bench reasoning | Write one model each for BBH Boolean Expressions and Dyck Languages, plus BBEH Web of Lies and Shuffled Objects. | Use each original answer normalization. Repeat the BBEH tests on freshly generated questions and on the published wording/layout perturbations. Report exact counts for every condition. |
| 7. Graph algorithms, SAT and planning | Build CLRS-Text strongly connected components, Prim’s minimum spanning tree and longest common subsequence; SATLIB random 3-SAT; and PlanBench Blocksworld, Mystery Blocksworld and Logistics. | Use all five CLRS test splits and exact reference outputs, additionally checking minimum cost on Prim ties. Validate SAT assignments clause by clause and score satisfiable and unsatisfiable sets separately. Replay plans under the domain rules and check their final goals. |
| 8. Trained comparisons | Reset the written maze architecture to random initialization. Reproduce the normalization, learning-rate, width/depth, seed and curriculum conditions. Also compare feedforward and published recurrent baselines (including HRM, TRM and EBR) and the model families used in the three generalization studies. | Keep selection on DEV, log every seed/checkpoint, record exact-path and any-shortest-path scores as separate columns. Match examples, optimizer steps and precision first. Measure Kaggle GPU time directly; the paper's H200 wall times and dollar estimates are hardware-specific. |
| 9. Direct answers and interpretability | Run the original direct-answer maze prompts (ordinary grid, search instructions, one cell per token, path only and cell-list inputs), harness comparison, and direct-answer controls for the other benchmarks. Give three isolated readers anonymized tensors and bare architecture and use a separate scorer. | Record model/version, reasoning settings, seeds, token usage and grader per condition. Use the published model only when available and explicitly budgeted. Score blind reports against a rubric fixed before readers see the weights. Use a neutral tensor naming scheme to avoid giving away the task. |

The repository publishes first prompts and reference weights. Some prompts describe files in the authors’ original environment and omit later messages. Inventory the available splits, generator versions, private harnesses and exact item IDs before each phase. When an original subset or generator cannot be recovered, publish the replacement, its seed and its hash, and label that comparison as a reconstruction. [Prompt inventory](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/prompts/README.md) · [Evaluation tables and appendices](https://mbzuai-oryx.github.io/LoopedWSP/)

## Kaggle scheduling

The maze DEV and TEST jobs each request one private GPU session on the user's Kaggle account. For subsequent stages, time one development batch, measure peak memory, and extrapolate GPU-hours as measured seconds per item times the target item count divided by 3,600. Split long runs into disjoint, predeclared item ranges; save outputs and manifests per range, check coverage and combine counts across ranges. The large Sudoku sweep, million-move Hanoi run, long-context GraphWalks and training baselines each need their own measurements. Paid model-API comparisons are a separate resource from Kaggle GPU execution.

## Local entry points

From this project directory:

```powershell
python -m unittest discover -s src/maze -p test_weights.py -v
python src/maze/audit.py
python submit.py dev
```

The audit uses the pinned training CSV in `data/maze/`. The final command materializes the private Kaggle script and metadata locally. Adding `--push` submits the named split. Existing versioned runs can be inspected from their receipts in `kaggle/runs/` without spending additional quota. The original held-out evaluation is a single frozen run.
