# Looped transformers on three operational tasks

The frozen models labeled **960 of 960 complete records** correctly, with **47,616 of 47,616 individual decisions** correct. They produced 0 false positives and 0 false negatives. On 960 records, every prediction stayed the same at the three loop budgets. [Private Kaggle run, version 1](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-practical-applications/versions/1)

## Evaluation

These are three synthetic application benchmarks created for this reproduction. Each task has a separate transformer with analytic weights and an independent conventional scoring rule. Development used 12 records per task at each of lengths 8 and 16. The held-out run covers 64 distinct new records per task at each of lengths 8, 16, 32, 64 and 128. A deterministic generator uses different seeds for development and held-out inputs, and excludes duplicate records across splits at each size. The frozen cases, labels and model hashes identify the evaluated data and weights. The results measure these bounded synthetic distributions. [Corpus lock](<../../applications/corpus-lock.json>)

The GPU run used Tesla T4, float64 and batches of up to 16 records. Each held-out record was scored once, with readouts at the base loop budget R, 2R and 4R during the same evaluation. A record is exact when every output matches its reference label. Stability counts whole records with identical predictions at all three checkpoints. GPU times are the summed batch times through R. CPU oracle times measure conventional reference-label calculation on the Kaggle host and serve as timing context only. [Evaluation code](<../../applications/experiment.py>)

Each example below shows all eight inputs from one held-out record. Where available, the first record with both positive and negative reference labels is used. Flagged positions are numbered from 1. To score a raw JSON record, use `python applications/predict.py inventory record.json`, substituting `scheduling` or `telemetry` for those tasks. The command also accepts standard input and returns one boolean per input. [Prediction CLI](<../../applications/predict.py>)

## Inventory shortfall detection

A positive label flags a negative on-hand balance after a transaction. Each stream starts with a signed opening balance from −24 through 24 and contains signed quantities from −16 through 16. The generator includes shortfalls, recoveries and balances of exactly zero.

**Example, record 1.** `{"opening":12,"changes":[-12,-3,3,8,-14,5,-2,-6]}`

The model flagged positions **2, 5, 6, 7, 8**. The reference flagged **2, 5, 6, 7, 8**.

The model contains **1,400 scalar entries**, including zeros, and **617 nonzero entries**. [Model and generator](<../../applications/inventory.py>) · [Freeze record](<../../applications/models/inventory/freeze.json>) · [Weight archive](<../../applications/models/inventory/weights.npz>)

| Events per record | R | Exact / records | Correct / decisions | Positive labels | False positives | False negatives | Stable records | GPU seconds | CPU oracle seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4 | 64 / 64 | 512 / 512 | 204 | 0 | 0 | 64 | 0.012470 | 0.001030 |
| 16 | 5 | 64 / 64 | 1,024 / 1,024 | 431 | 0 | 0 | 64 | 0.015213 | 0.001329 |
| 32 | 6 | 64 / 64 | 2,048 / 2,048 | 834 | 0 | 0 | 64 | 0.018179 | 0.001705 |
| 64 | 7 | 64 / 64 | 4,096 / 4,096 | 2,146 | 0 | 0 | 64 | 0.021048 | 0.002681 |
| 128 | 8 | 64 / 64 | 8,192 / 8,192 | 3,961 | 0 | 0 | 64 | 0.024922 | 0.004120 |

## Booking conflicts

A positive label flags a booking that overlaps another booking for the same resource. Times are whole minutes within one 1,440-minute day. Bookings use half-open intervals, so adjacent endpoints can share a time without conflict. Inputs include shuffled duplicates, nested appointments, overlaps and disjoint bookings.

**Example, record 2.** `{"bookings":[[847,851],[867,871],[842,847],[851,860],[867,871],[854,856],[862,866],[862,866]]}`

The model flagged positions **2, 4, 5, 6, 7, 8**. The reference flagged **2, 4, 5, 6, 7, 8**.

The model contains **61,830 scalar entries**, including zeros, and **9,384 nonzero entries**. [Model and generator](<../../applications/scheduling.py>) · [Freeze record](<../../applications/models/scheduling/freeze.json>) · [Weight archive](<../../applications/models/scheduling/weights.npz>)

| Bookings per record | R | Exact / records | Correct / decisions | Positive labels | False positives | False negatives | Stable records | GPU seconds | CPU oracle seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 64 / 64 | 512 / 512 | 314 | 0 | 0 | 64 | 0.029807 | 0.001040 |
| 16 | 16 | 64 / 64 | 1,024 / 1,024 | 619 | 0 | 0 | 64 | 0.042991 | 0.001140 |
| 32 | 32 | 64 / 64 | 2,048 / 2,048 | 1,309 | 0 | 0 | 64 | 0.077359 | 0.001414 |
| 64 | 64 | 64 / 64 | 4,096 / 4,096 | 2,736 | 0 | 0 | 64 | 0.114338 | 0.002301 |
| 128 | 128 | 64 / 64 | 8,192 / 8,192 | 5,788 | 0 | 0 | 64 | 0.169685 | 0.005535 |

## Hysteresis alarm

A positive label means the alarm is on after that reading. It starts off, switches on at 70 or above, resets at 30 or below and preserves its state between the thresholds. Generated values span 0 through 100 and include exact thresholds, brief pulses and long holds.

**Example, record 1.** `{"readings":[70,31,60,38,30,47,33,54]}`

The model flagged positions **1, 2, 3, 4**. The reference flagged **1, 2, 3, 4**.

The model contains **1,508 scalar entries**, including zeros, and **509 nonzero entries**. [Model and generator](<../../applications/telemetry.py>) · [Freeze record](<../../applications/models/telemetry/freeze.json>) · [Weight archive](<../../applications/models/telemetry/weights.npz>)

| Readings per record | R | Exact / records | Correct / decisions | Positive labels | False positives | False negatives | Stable records | GPU seconds | CPU oracle seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 64 / 64 | 512 / 512 | 238 | 0 | 0 | 64 | 0.008741 | 0.000875 |
| 16 | 16 | 64 / 64 | 1,024 / 1,024 | 465 | 0 | 0 | 64 | 0.013784 | 0.001495 |
| 32 | 32 | 64 / 64 | 2,048 / 2,048 | 976 | 0 | 0 | 64 | 0.025073 | 0.002808 |
| 64 | 64 | 64 / 64 | 4,096 / 4,096 | 1,913 | 0 | 0 | 64 | 0.045702 | 0.005516 |
| 128 | 128 | 64 / 64 | 8,192 / 8,192 | 3,816 | 0 | 0 | 64 | 0.090263 | 0.010293 |

## Archive and verification

[All size results (CSV)](<sizes.csv>) · [Kaggle result JSON](<../../kaggle/runs/practical-applications/v1/output/application-results.json>) · [Kaggle predictions](<../../kaggle/runs/practical-applications/v1/output/predictions.npz>) · [Independent local verification](<verification.json>) · [Submission receipt](<../../kaggle/runs/practical-applications/v1/submission.json>) · [Synthetic cases](<../../applications/cases.npz>)
