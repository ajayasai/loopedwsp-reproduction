# Three field studies with programmed transformer weights

The models take JSON inputs and return a map of retained water, a candidate transit period and phase, or a live bowling scorecard. They use task-specific attention mechanisms with 6, 12 and 7 stored FP32 coefficients respectively. On a Tesla T4 they matched their references on 768/768 held-out records. The transit search recovered the exact period and phase on 83/128 injected signals, with zero detections on 128 controls. [Results, figure and timing breakdown](../results/fieldlab/report.md)

## Run an example

From the repository root, use Python 3.12 or later with NumPy and PyTorch installed. These predictions run on CPU. Replace the example path with your own file, or omit `--input` to read JSON from standard input.

```sh
python fieldlab/experiment.py predict basin --input fieldlab/examples/basin.json
python fieldlab/experiment.py predict transit --input fieldlab/examples/transit.json
python fieldlab/experiment.py predict bowling --input fieldlab/examples/bowling.json
```

| Example | Expected model output |
| --- | --- |
| [Basin draining across a saddle](examples/basin.json) | A 5×5 water-depth map totaling **22 cell-height units**. The saddle at elevation 3 controls the interior spill level. |
| [Brightness curve with a missing sample](examples/transit.json) | `{"period":5,"phase":2}`. The missing observation falls at an expected dip. |
| [Five started bowling frames](examples/bowling.json) | `{"cumulative":[27,47,66,75,null]}`. The fifth frame is a strike whose bonus balls have yet to be delivered. |

## Hydrology · depression storage

Input `heights` is a square integer raster, 3–12 cells per side, with elevations from 0 to 9. Water can travel north, south, east or west, and all perimeter cells drain freely. Each cell's spill elevation is the lowest elevation at which a path to the perimeter opens. The output gives water depths in elevation units and their sum for unit-area cells. It describes static capacity under these boundary conditions.

The transformer starts interior cells at the highest elevation. A tied attention block reads the lowest neighboring spill level and constrains the new level to be at least the local ground height. The frozen target comes from a heap-based priority flood. The independent audit instead checks which cells connect to an edge at each integer water level. [Model](basin.py) · [Priority-Flood paper](https://arxiv.org/abs/1511.04463)

## Astronomy · detecting periodic brightness dips

Input `baseline` is a known integer brightness from 16 to 4,095. The `flux` array contains 1–192 regularly spaced measurements, with `null` for a gap. Each observation must lie within 16 units of the baseline. The model tests a one-cadence-wide, eight-unit dip at each phase of periods 3, 4, 5, 7 and 9. Cross-attention pools the evidence and counts supporting observations for all 28 templates in one pass.

For a candidate, sum `baseline − flux − 4` across its observed dip positions. This is the improvement in squared error over a flat baseline, divided by 16. A candidate needs a score of at least 12 and at least three observations five or more units below baseline. The greatest score wins. Ties favor the shorter period and then the earlier phase. If none qualifies, both output fields are `null`. Phase is the zero-based cadence index modulo the period.

The test series include gaps, noise, transients and deliberately ambiguous periods. The model receives only the baseline and measured flux. Injection metadata is stored separately so the audit can report what the model actually recovered. A candidate in this synthetic benchmark is an output of the specified search. [Model](transit.py) · [NASA's introduction to transit photometry](https://science.nasa.gov/exoplanets/whats-a-transit/)

## Sports · live ten-pin bowling scores

Input `frames` groups credited pin counts by started frame. Earlier frames must be complete. The latest frame may end after any legal delivery. Frame boundaries are supplied in the input, and the scoring model derives strike and spare bonuses from the following deliveries. It outputs one cumulative score per started frame, using `null` while any required delivery is missing. A zero-pin delivery is a recorded score; it can settle a pending entry. Inputs use scratch scoring and the standard tenth-frame fill rules.

Attention locates the following deliveries needed for bonuses. A shared attention block then accumulates points and pending status across frames. Development checks included perfect games, all-spare games, gutters and partial tenth frames. [Model](bowling.py) · [USBC scoring examples](https://bowl.com/keeping-score)

## Study design and verification

| Field | Development sizes | Test sizes | Exact record |
| --- | --- | --- | --- |
| Hydrology | 4×4, 6×6 | 4×4, 6×6, 8×8, 12×12 | Every cell's water depth |
| Astronomy | 24, 48 cadences | 24, 48, 96, 192 cadences | The search rule's period and phase, or no candidate. Injected-signal recovery is reported independently. |
| Sports | 3, 5 started frames | 3, 5, 8, 10 started frames | Every cumulative score, including all pending markers |

Development used eight unique records per size, and held-out testing used 64. The three tasks cover 768 held-out records in total. Short bowling scorecards are prefixes of regulation ten-frame games. The model ledger counts stored coefficient entries including zeros. Fixed coordinate features, attention masks, candidate definitions and working memory are outside the parameter counts. The astronomy model has a fixed 28-pair period/phase index. [Protocol](protocol.json) · [Stored tensors and fingerprints](freeze.json) · [Cases and scoring targets](cases.json.gz)

```sh
python fieldlab/experiment.py dev
python fieldlab/experiment.py gpu --output audit-output/fieldlab-gpu
python fieldlab/audit.py kaggle/runs/fieldlab/v1 audit-output/fieldlab-audit.json
```

To submit a new private run, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your username and run `python fieldlab/submit_study.py --push`. Collect its completed version with `python fieldlab/submit_study.py --collect VERSION`. The launcher requests an offline Tesla T4 and verifies the downloaded source and output hashes. [Launcher](submit_study.py)
