# Locate an earthquake, read a Braille label, choose a bridge lead

These three hand-programmed attention models run on CPU from JSON inputs. Each uses fixed FP32 weights. The worked examples let you see what happens when a seismic station goes silent, a Braille indicator changes, or a different suit becomes trump.

## Run the examples

From the repository root with Python 3.12 or later, NumPy and PyTorch installed, run any of these commands.

```sh
python mosaic/experiment.py predict quake --input mosaic/examples/quake.json
python mosaic/experiment.py predict braille --input mosaic/examples/braille.json
python mosaic/experiment.py predict bridge --input mosaic/examples/bridge.json
```

| Example | Answer | Try changing the input |
| --- | --- | --- |
| [Earthquake source](examples/quake.json) | Three stations locate the source at **(2, 3), origin tick 4**. | Change the last arrival from 8 to `null`. The two remaining readings admit five locations. [Station and time format](quake_notes.md) |
| [Braille room label](examples/braille.json) | The line reads **Room 204b**. Dots 1 and 2 print as the digit 2 inside the number and as the letter b after the grade-1 indicator. | Remove the grade-1 indicator `"56"`. The same final dots now give **Room 2042**. [Dot patterns and supported rules](braille_notes.md) |
| [Contract bridge endgame](examples/bridge.json) | In no-trump, North must lead the **six of spades** to secure one trick. With hearts as trumps, North–South gets zero tricks after either lead. | The example contains the same hands twice. Change the second deal’s trump from `"H"` to `"NT"` and compare the optimal leads. [Card format and trick rules](bridge_notes.md) |

Use `--example` to display the built-in input and output together. Omit both options to read a JSON record from standard input.

## What the models solve

| Model | Programmed coefficients | Development sizes | Held-out sizes | Exact scoring rule |
| --- | ---: | --- | --- | --- |
| Seismology | 74 | 2, 4 stations | 2, 4, 8, 16 | Match the full 7×7 map of compatible sources and origin ticks, including empty and ambiguous solutions. |
| Braille accessibility | 127 | 8, 16 cells | 8, 16, 32, 64 | Match the print symbol, blank or nonprinting indicator at every position in a transcribed six-dot line. |
| Contract bridge | 3 | 2, 4 deals | 2, 4, 8, 16 | Match the remaining tricks North–South can force and every equally optimal opening card on every deal. |

The seismic simulator uses rounded travel times at a constant wave speed on a flat grid. The Braille task starts from known dots and covers uncontracted letters, integer digits and single-symbol indicators. Each bridge position has two cards per player, with all hands visible. Increasing the bridge input size gives the model more independent deals at the same search depth. The coefficient counts include the stored parameter tensors. Each implementation also uses fixed operations such as distance arithmetic, glyph masks or the enumeration of legal card continuations.

Each held-out size has 64 seeded records per task. A separate test changes only the first cell in 320 pairs of Braille lines. Each pair ends with identical dots that should print as a digit in one line and a lowercase letter in the other. A reader with access only to the shared suffix receives identical evidence for both answers. [Pre-run public snapshot](https://github.com/ajayasai/loopedwsp-reproduction/tree/920767e/mosaic) · [Protocol and seeds](protocol.json) · [Model weights and corpus hashes](freeze.json)

The local CPU run matched the Kaggle Tesla T4 on all 27 prediction arrays. Together these cover 22,144 held-out labels, 251 labels across 11 worked examples and 19,200 labels in the 320 matched Braille pairs. [Per-size CPU results](../results/mosaic/cpu-results.json) · [CPU/GPU comparison and reproducible commands](../results/mosaic/cpu-parity.json)

## Inspect or reproduce the measurements

The measured results include exact record and label counts, all 12 per-size GPU timings, the matched-pair scores and a plot showing which locations fit two and three station readings. The audit reconstructs target labels from the raw observations, dot strings and hands. It also checks the frozen files against the script Kaggle ran. [Results and figure](../results/mosaic/report.md) · [GPU predictions](../kaggle/runs/mosaic/v1/output/mosaic-predictions.npz) · [Independent audit](../results/mosaic/audit.json)

```sh
python mosaic/experiment.py dev
python mosaic/audit.py kaggle/runs/mosaic/v1 audit-output/mosaic-audit.json
python mosaic/report.py kaggle/runs/mosaic/v1 results/mosaic
```

The audit needs NumPy. The figure also needs Matplotlib. You can rerun the frozen test locally on a CUDA GPU with `python mosaic/experiment.py gpu --output audit-output/mosaic-gpu`. To repeat it on your own Kaggle account, use a fresh checkout, authenticate the Kaggle CLI and set `LOOPEDWSP_KAGGLE_OWNER` to your username. Run `python mosaic/submit_study.py --push`, then `python mosaic/submit_study.py --collect VERSION` after completion. The launcher writes into the numbered run directory in that checkout.
