# Isotope spectra, the Maya Calendar Round and robot-arm poses

These three hand-programmed transformers predict a spectrum from an ion mixture, resolve paired Maya calendar designations within one cycle, and find the possible joint positions of a simulated arm. Each accepts JSON and runs locally on CPU. [Kaggle results and per-size timings](../results/prism/report.md)

## Try the examples

Use Python 3.12 or later with NumPy and PyTorch installed. From the repository root, run any of the following.

```sh
python prism/experiment.py predict isotopes --input prism/examples/isotopes.json
python prism/experiment.py predict calendar_round --input prism/examples/calendar_round.json
python prism/experiment.py predict arm --input prism/examples/arm.json
```

| Application | What to look for |
| --- | --- |
| [Isotope spectrum](examples/isotopes.json) | A chlorine-containing ion contributes 24 intensity units at nominal m/z 402. A bromine-containing ion contributes another 64 there. The mixture has a single peak of **88 units** at that mass. [Mass bins and isotope assumptions](isotopes_notes.md) |
| [Maya Calendar Round](examples/calendar_round.json) | The three answers are **0**, **null**, and **18,979**. The calendar origin is 4 Ajaw 8 Kumk’u. Changing only the Haab’ day to 9 makes that pairing impossible. The third pairing falls on the last day before the cycle returns to the origin. [Calendar names and conventions](calendar_round_notes.md) |
| [Robot arm](examples/arm.json) | With an obstacle at (1, 0), three poses reach (2, 2). The first in the model’s declared order is **LSRS**, four successive choices of left, straight, right, straight. The same obstacle blocks the straight reach to (4, 0). [Arm geometry and collision rules](arm_notes.md) |

Replace the file with your own JSON, use `--example` to print the built-in input and its prediction together, or omit `--input` to read standard input.

## What the test measures

Each task uses eight development records at each of sizes 2 and 4, and 64 held-out records at each of sizes 2, 4, 8 and 16. The size counts ion components, calendar requests or robot targets. More ion components can contribute to the same peak. More dates or targets ask the model to answer additional queries under the same calendar or arm rules.

A spectrum must match exactly in all 16 bins from nominal m/z 400 through 415. A calendar answer must match the entire offset or identify an impossible pairing. A robot target requires both its exact number of poses and the first feasible pose in the declared order, including an empty answer when unreachable. The strict record score requires every answer in the record. A separate sweep tests all 94,900 possible date pairings under the declared calendar convention. Its scores and the eleven worked checks are reported separately from the sampled test. [Pre-run snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/a4ff7d6/prism) · [Protocol](protocol.json) · [Worked cases](checks.json) · [Weights and data hashes](freeze.json)

## Verify or repeat the experiment

The audit recomputes the targets from raw inputs with independent solvers. It enumerates heavy and light isotope assignments, advances calendar counters for a full 18,980-day cycle, and checks robot link intersections from absolute headings. Running the audit requires NumPy. Rendering the report also requires Matplotlib.

```sh
python prism/audit.py kaggle/runs/prism/v1 audit-output/prism-audit.json
python prism/report.py kaggle/runs/prism/v1 results/prism
```

For local CPU development checks use `python prism/experiment.py dev`. To repeat the GPU test on Kaggle, authenticate the Kaggle CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your username and run `python prism/submit_study.py --push`. Collect the completed run with `python prism/submit_study.py --collect VERSION`. [Kaggle launcher](submit_study.py) · [Archived predictions](../kaggle/runs/prism/v1/output/prism-predictions.npz)
