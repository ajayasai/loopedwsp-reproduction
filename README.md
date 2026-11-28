# Looped weight-space programming on Kaggle

A hand-written transformer matched **1,000 of 1,000 official Maze-Hard test grids**, including the benchmark’s choice between equally short paths. This repository contains the analytic weights, scoring code and GPU results for a Codex-assisted reproduction of selected LoopedWSP experiments. We also built 27 application models that run locally, from detecting booking conflicts to choosing the best opening lead in a bridge endgame. [Maze test result](results/maze/test.json) · [Operational applications](results/applications/report.md) · [Creative applications](creative/README.md) · [Workshop applications](workshop/README.md) · [Field studies](fieldlab/README.md) · [Linguistics, crystallography and auctions](crossfield/README.md) · [Tree rings, weaving and knots](atlas/README.md) · [Maps, plumage and quantum circuits](frontier/README.md) · [Isotopes, calendrics and robotics](prism/README.md) · [Earthquakes, Braille and bridge](mosaic/README.md)

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

## Inventory, scheduling and sensor alarms

Each application has its own frozen transformer. Development used records of length 8 and 16. A single held-out GPU evaluation covered 64 records per task at each of five lengths, from 8 through 128. All **960/960 complete records** and **47,616/47,616 individual labels** matched independent scoring rules. Every prediction remained stable at the three loop budgets. These tests cover the specified synthetic rules and input ranges. [Examples and scores by size](results/applications/report.md) · [Independent verification](results/applications/verification.json)

| Model | One output for each input | Held-out records |
| --- | --- | ---: |
| [Inventory](applications/inventory.py) | Flag a negative stock balance after applying each signed transaction to the opening balance. | 320/320 |
| [Booking conflicts](applications/scheduling.py) | Flag every booking that overlaps another for the same resource. Touching endpoints are compatible. | 320/320 |
| [Sensor alarms](applications/telemetry.py) | Switch on at 70 or above, switch off at 30 or below, and keep the previous alarm state in between. | 320/320 |

Across the 15 task-size settings, GPU model inference took 0.710 seconds in total. The conventional CPU scoring rules produced the same labels in 0.043 seconds on the Kaggle host. [Timing breakdown](results/applications/sizes.csv)

## Paper folding, typesetting and film-club ballots

| Application | What you give it | What it produces |
| --- | --- | --- |
| Paper-fold preview | Both colored faces of a sheet and a sequence of center folds, including which flap goes over or under | The visible color of every remaining cell |
| Paragraph typesetter | Words and a line width in monospaced columns | Line breaks that minimize squared unused space, with a free final line |
| Film-club ballot counter | Five titles and each voter's complete ranking | All four eliminations and the winning film, including vote transfers and deterministic tie-breaking |

Each model scored **256/256 complete outputs** on a Kaggle Tesla T4, including all 384 records across the three tasks at input sizes larger than development. The typesetter found the optimal layout in all 78 cases where greedy wrapping cost more. The ballot counter reproduced all rounds in the 85 elections where the unique first-choice leader ultimately lost. [Measured results and illustrated examples](results/creative/report.md) · [Per-size timings](results/creative/sizes.csv) · [Independent audit](results/creative/audit.json)

For a runnable example, use `python creative/experiment.py predict folds --example` from the repository root with NumPy and PyTorch installed. Replace `folds` with `typeset` or `ballots` to try the other models. Use `--input FILE` to supply a JSON record of your own. [Input formats and examples](creative/README.md) · [Study frozen before the GPU run](https://github.com/ajayasai/loopedwsp-reproduction/tree/09024da/creative)

## Kinetic sculpture, animation and music recovery

Three more programmed transformers matched **768/768 complete records** on a Kaggle Tesla T4. Development used sizes 4 and 8. The same weights also solved all 384 held-out records of sizes 16 and 32. [Frozen study](https://github.com/ajayasai/loopedwsp-reproduction/tree/acad7ea/workshop) · [Full results and timings](results/workshop/report.md)

| Application | Stored parameters | What the held-out run verified |
| --- | ---: | --- |
| Kinetic mobile | 7 | Found the missing 1–16 g counterweight on all 128 solvable designs and rejected all 128 impossible ones. Checking only the top rod would have accepted 71 impossible designs. |
| Animation cel compositor | 64 | Rendered every RGB channel in 256 layered 8×8 scenes, including 240 whose image changes when the layer order reverses. |
| Punched music-roll recovery | 1,342 | On 256 synthetic rolls, recovered the note and track for all 1,466 single-hole errors and marked all 870 double-hole errors as unreadable. |

Run `python workshop/experiment.py predict mobile --example` to try a mobile. The compositor exports a PNG and the music-roll decoder exports a playable WAV with unreadable beats preserved as silence. [Run your own input](workshop/README.md) · [Example frame](results/workshop/composite.png) · [Example melody](results/workshop/recovered.wav) · [Independent audit](results/workshop/audit.json)

## Hydrology, astronomy and sports

| Model | Stored parameters | Held-out result on a Kaggle Tesla T4 |
| --- | ---: | --- |
| Terrain depression filling | 6 | Reconstructed the full retained-water depth map on 256/256 terrains, including 223 where inspecting immediate neighbors alone gives the wrong map. |
| Transit search | 12 | Matched an independent template search on 256/256 synthetic light curves. It recovered the injected period and phase for 83/128 signals and produced 0 false alarms on 128 controls. Observation gaps made a shorter period indistinguishable in 27 cases, and 18 injections fell below the detection rule. |
| Live ten-pin scorekeeping | 7 | Matched every entry on 256/256 scorecards, including 147 with unresolved frame scores and 32 finished ten-frame games. |

Run `python fieldlab/experiment.py predict bowling --example` to see a scorecard with an unresolved strike bonus. Substitute `basin` to get a water-depth map or `transit` for a period and phase. [Runnable JSON examples](fieldlab/examples) · [Input formats and commands](fieldlab/README.md) · [GPU results and figure](results/fieldlab/report.md) · [Independent audit](results/fieldlab/audit.json) · [Protocol frozen before submission](https://github.com/ajayasai/loopedwsp-reproduction/tree/ffb4367/fieldlab)

## Turkish nouns, crystal diffraction and bundle auctions

| Application | Stored FP32 coefficients | Exact held-out records | What the model did |
| --- | ---: | ---: | --- |
| Regular Turkish noun inflection | 449 | 256/256 | Chose both endings in all 1,920 generated noun requests. A separate check matched 12 real-word forms, including *göl* → *gölleri* (plural accusative). |
| Crystal diffraction | 43 | 256/256 | Predicted all 3,840 reflection intensities for idealized quarter-cell motifs, including the 1,102 reflections where the scattering contributions canceled. |
| Combinatorial auction pricing | 6 | 256/256 | Found the winning bundles and each bidder’s payment. Taking the highest bids first would have lost value in 52 of these auctions. |

We froze the weights and cases before the Kaggle Tesla T4 run. All 384 records at input sizes above the development range matched the independent scoring rules. The auction model enumerates up to 256 bidder subsets. The parameter counts exclude this memory and each model’s fixed operations. [Inputs and runnable commands](crossfield/README.md) · [Per-size GPU timings and examples](results/crossfield/report.md) · [Independent audit](results/crossfield/audit.json) · [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/1d1419b/crossfield)

## Tree-ring alignment, loom drafting and knot theory

| Application | Stored FP32 coefficients | Exact held-out records | What the experiment showed |
| --- | ---: | ---: | --- |
| Match an eight-ring sample to a reference series | 11 | 256/256 | Of 56 samples with a single window that reproduced every width after rescaling and shifting, all 56 had that offset among the best direction matches. Only 45 gave it the sole best score. |
| Draw cloth from a rising-shed loom draft | 4 | 256/256 | Predicted which yarn was on top at all 87,040 crossings. In 114 drafts, keeping only the first treadle pressed on each pick would have changed the cloth. |
| Count Fox 3-colorings of a closed three-strand braid | 50 | 256/256 | Counted all valid colorings through 32 crossings. The worked examples give 9 for the trefoil, 3 for the figure-eight knot and 27 for three unlinked circles. |

The three frozen models matched all 384 records at sizes above their development ranges on a Kaggle Tesla T4. The tree-ring scores rank synthetic alignments, the loom predicts ideal interlacements, and the braid count can be shared by different knots. [Runnable examples and guides](atlas/README.md) · [GPU measurements and loom drawdowns](results/atlas/report.md) · [Independent audit](results/atlas/audit.json) · [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/2480f9e/atlas)

## Contour maps, ancestral plumage and quantum circuits

| Application | Stored FP32 coefficients | Exact held-out records | What the model recovered |
| --- | ---: | ---: | --- |
| Cartography | 54 | 256/256 | Traced the contour through all 48,960 cells. It identified 648 four-way saddles and chose the correct pairing in all 374 cells where a center-height shortcut picked the wrong one. |
| Evolutionary biology | 5 | 256/256 | Found every plumage state compatible with a minimum-change history at 2,304 ancestors. For 959 of them, including the rest of the tree changed the optimal set. |
| Quantum information | 1,025 | 256/256 | Predicted all 2,048 measurement expectations for ideal three-qubit circuits, including 654 joint outcomes with a definite sign and at least one individually random factor. |

All 384 records at sizes above development matched on a Kaggle Tesla T4. The examples let you change a map elevation, a bird’s observed plumage, or the order of two quantum gates and see the resulting predictions. [Run the three examples](frontier/README.md) · [Per-size GPU results and contour figure](results/frontier/report.md) · [Independent audit](results/frontier/audit.json) · [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/c2d04db/frontier)

## Isotope spectra, Maya calendrics and tabletop robotics

| Application | Stored FP32 coefficients | Exact held-out records | What the model recovered |
| --- | ---: | ---: | --- |
| Analytical chemistry | 3 | 256/256 | Predicted all 4,096 mass-bin intensities for idealized chlorine and bromine mixtures, including 771 peaks with contributions from different parent masses. |
| Maya Calendar Round | 1,292 | 256/256 | Returned the full offset or identified an impossible pairing for 1,920 date requests. In a separate exhaustive sweep it resolved all 18,980 compatible pairs and rejected all 75,920 incompatible pairs. |
| Discrete robot-arm poses | 9 | 256/256 | Gave both the number of collision-free poses and the first valid joint configuration for 1,920 targets. For 339 targets, placing obstacles removed every pose that worked on an empty board. |

The frozen models ran on a Kaggle Tesla T4. The worked examples let you separate the contributions to an overlapping mass peak, try a designation at the end of the calendar cycle, or see how one obstacle cuts the number of arm poses at a target from six to three. [Examples and input formats](prism/README.md) · [Per-size timings and isotope spectrum](results/prism/report.md) · [Independent audit](results/prism/audit.json) · [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/a4ff7d6/prism)

## Earthquake location, Braille reading and bridge endgames

| Application | Stored FP32 coefficients | Exact held-out records | What the GPU test showed |
| --- | ---: | ---: | --- |
| Locate an earthquake on a synthetic 7×7 grid | 74 | 256/256 | Recovered every compatible location and origin tick in 107/107 ambiguous events, and identified 84/84 events for which no source fit the readings. |
| Read an uncontracted English Braille label | 127 | 256/256 | Decoded lines up to 64 cells long. In a separate test, changing only the first indicator changed how the final dots should read. Both lines were exact in 320/320 pairs. |
| Choose a lead in a two-trick contract bridge ending | 3 | 256/256 | Recovered the North–South trick count and all optimal opening cards in 1,920/1,920 deals, including 95/95 where leading the higher-ranked card would cost a trick. |

The same weights matched 384/384 records at sizes beyond development on a Kaggle Tesla T4. The seismic model uses constant-speed travel times, the Braille model reads supplied dot patterns, and the bridge model sees all four hands. Larger bridge inputs add deals while keeping each ending at two tricks. [Try the three examples](mosaic/README.md) · [Measurements and seismic maps](results/mosaic/report.md) · [Independent audit](results/mosaic/audit.json) · [CPU/GPU comparison](results/mosaic/cpu-parity.json) · [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/920767e/mosaic)

## When does looping help?

On a Kaggle Tesla T4, the same frozen weights went from **155/960** exact published records after one pass to **381/960** after four and **960/960** at full depth in both FP32 and FP64. For 240 new pairs in which changing the first item flips the final answer, one pass got both answers correct on **0/240** pairs. Full depth and task-specific global feedforward models each got **240/240**. The global baselines use a causal prefix mean for inventory, pairwise overlap checks for bookings, and attention to the latest decisive sensor reading. They also passed all 1,440 records in FP16. All three original models encountered nonfinite intermediate values in FP16 at full depth.

Looping helped the original blocks carry information beyond the positions they could reach in one pass. A feedforward computation with direct access to the relevant inputs also solved these tasks, using 3, 16 and 4 stored scalar coefficients respectively. Those counts exclude the fixed operations and masks. The study covers 315 settings at lengths 8–128 with equal-parameter depth comparisons, local-window controls, and three timed repeats per setting. [Full comparison and accuracy plot](results/loop-comparison/report.md) · [All measurements](results/loop-comparison/sizes.csv) · [Parameter budgets](results/loop-comparison/parameter-budgets.csv) · [Protocol frozen before the run](https://github.com/ajayasai/loopedwsp-reproduction/tree/04111d2a806743ae4011bfb4a4310f28591cc867/comparisons)

### Pointer chasing and directed reachability

On 64-vertex graphs, a six-stage non-looped transformer matched the looped model on **1,280/1,280 queries per task** at both parameter budgets. We varied path length from 1 to 32 hops and held graph size fixed. Each pair of test cases swaps two edges, preserves all vertex degrees and changes the answer. We published the models, cases and scoring protocol before submitting the Tesla T4 run. [Frozen study](https://github.com/ajayasai/loopedwsp-reproduction/tree/4983e21fcac0b524c15eea7a5f9bb0df8d162750/graph_depth)

| Task | Matched parameter budgets | What changed with depth |
| --- | ---: | --- |
| Pointer chasing | 3,072 and 6,144 | Six applications of one block and six independently stored narrower blocks each found every endpoint. The two- and four-stage controls explicitly cap the requested distance at 3 and 15 hops. Both missed all 128 endpoints at 32 hops. |
| Directed reachability | 52 and 100 | One local pass solved 704/1,280. Reusing that block for 63 passes solved 1,280/1,280. A six-stage non-looped model that doubles its path coverage at each stage also solved 1,280/1,280. |

At 32 hops and 52 parameters, the global non-looped model took **30.40 ms** for 128 reachability queries, compared with **197.47 ms** for the looped local model. The global model maintains 4,096 evolving relation entries per graph, while the local model maintains 64 vertex flags. The measured advantage of looping was access to more rounds of local propagation at fixed parameter storage. These are hand-programmed models evaluated on synthetic graphs. [Exact scores by length and timing protocol](results/graph-depth/report.md) · [All 240 measurements](results/graph-depth/sizes.csv) · [Independent audit](results/graph-depth/audit.json)

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
