# When does looping help?

This experiment asks two questions about inventory shortfalls, booking conflicts and sensor alarms. First, how many records can the **same weights** label correctly with one, four or the full number of passes? Second, how well can a **separately designed feedforward model** solve the same task with global or limited context? The numerical comparison uses the same records, output positions and precision for every candidate. [Results](../results/loop-comparison/report.md) · [Protocol frozen before GPU submission](https://github.com/ajayasai/loopedwsp-reproduction/tree/04111d2a806743ae4011bfb4a4310f28591cc867/comparisons)

## Models

| Task | What another pass does in the original | What the global feedforward baseline does |
|---|---|---|
| Inventory | Doubles the length of the transaction prefix already accumulated. The full budget is ⌈log₂ n⌉ + 1. | A single causal attention head takes the uniform mean of all signed quantities through the current event, including the opening balance. For integer balance S and prefix length k, the score −(S + ½)/k is positive exactly when S < 0. |
| Scheduling | Propagates interval endpoints one row left and right in the shuffled list. The full budget is n. | A shared ReLU network compares each pair of half-open intervals. It excludes the booking itself and takes a maximum across the other bookings. This materializes a quadratic collection of pairs. |
| Telemetry | Moves the last alarm state forward by one row through a run of readings inside the hold band. The full budget is n. | One causal attention head assigns higher scores to trip/reset events and to more recent positions. Every position reads the newest decisive event, with the initial state off. |

The fixed-window feedforward controls have 2, 8 or 32 active taps. For inventory and telemetry the window ends at the current event. Scheduling distributes its taps equally across earlier and later list positions. Every reported scalar count covers stored coefficients, including zeros. The masks, extraction of pairs, comparisons, pooling and input encodings are part of the implementation; counting parameters alone does not measure their cost. [Feedforward models and coefficient metadata](baselines.py) · [Original tensors and instrumented forward](recurrent.py)

Running balance and one-bit state tracking have established precedents. Tracr implements a running signed balance for paired tokens, and flip-flop language modeling studies the effect of changing or retaining a bit. [Tracr balance implementation](https://github.com/google-deepmind/tracr/blob/main/tracr/compiler/lib.py#L59-L121) · [Liu et al., NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/510ad3018bbdc5b6e3b10646e2e35771-Abstract-Conference.html)

## Inputs and measurements

The original 960 public test records are reused. For each of the three tasks and five lengths, 16 additional pairs have identical inputs from row 2 onward and opposite correct final flags. A deterministic model that cannot see the first row at the last position can get at most one final flag per pair correct there. The two cohorts are scored separately. These constructed cases exercise information flow across distance, and their scores depend on that design. [Challenge generator](make_challenges.py) · [Fixed inputs](challenge.npz)

Each GPU timing covers 96 records at one task and length in six batches of 16. The measurements use one warm-up batch and three synchronized repeats. Original timing includes dense attention and intermediate numerical checks at each requested pass. The earlier application benchmark used routing caches and fixed-point stopping. The strict comparison counts a nonfinite checked numerical value as an invalid flag. The results also score the original models' final logits independently of that screen. [Evaluation code](experiment.py)

## Reproduce and verify

From the repository root, the commands below verify the released predictions and regenerate the report using the CPU dependencies in the root requirements file.

```sh
python comparisons/audit.py kaggle/runs/loop-comparisons/v1 audit-output/loop-comparison.json
python comparisons/report.py kaggle/runs/loop-comparisons/v1 results/loop-comparison
python comparisons/plot.py kaggle/runs/loop-comparisons/v1 results/loop-comparison/accuracy.png
```

To run the evaluator on your own Kaggle GPU, authenticate the Kaggle CLI and set `LOOPEDWSP_KAGGLE_OWNER` to your username in a disposable checkout. Run `python comparisons/submit_study.py --push`, then `python comparisons/submit_study.py --collect VERSION` once Kaggle reports completion. A local CPU smoke test with PyTorch installed is available through `python comparisons/experiment.py dev`. It evaluates only the original development records at lengths 8 and 16.
