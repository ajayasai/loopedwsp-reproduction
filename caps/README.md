# Parameter-capped Maze-Hard experiment

Each builder writes a distinct transformer from analytic formulas. The five builds share the evaluation code and the down, left, right, up tie convention established on the original development split. The builders had prior context from the uncapped reproduction. This experiment measures the performance of freshly constructed networks under the five limits; the original study also tested discovery in five isolated authoring sessions. [Original cap brief](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/prompts/13_maze_size_limit_network_brief.txt)

| Limit | Build | How the model finds the labeled path |
| ---: | --- | --- |
| 1,000 | [Two dynamic heads](c1000/write_weights.py) | One head selects a visited neighbor, and the second finds path cells that point to the current cell. Both operate on positions encoded from text line and column. |
| 5,000 | [Four-bit neighbor mask](c5000/write_weights.py) | Four positional heads encode neighbor arrivals as a bit mask. The first arrival latches one scalar direction code. |
| 10,000 | [Minimum distance](c10000/write_weights.py) | Computes the shortest distance to the goal by repeated minimum operations, then passes a path signal to the first downhill neighbor. |
| 20,000 | [First-arrival direction](c20000/write_weights.py) | Four predecessor bits record the wave’s first arrival. A single attention/MLP layer propagates a route from the start. |
| 40,000 | [Maximum potential](c40000/write_weights.py) | An explicit position table addresses adjacent characters. A decreasing potential spreads from the goal, and a route follows the first neighbor with the highest potential. |

## Counting and scoring

All five models matched all 1,000 official test grids in their individual frozen evaluations. Each ran in FP64 on a private Kaggle Tesla T4. Timings below cover the batched evaluation loop after GPU model preparation and CPU/GPU DEV verification.

| Parameter cap | Actual dense parameters | Nonzero | Exact TEST matches | Evaluation seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 965 | 112 | 1,000/1,000 | 352.20 |
| 5,000 | 2,801 | 279 | 1,000/1,000 | 34.15 |
| 10,000 | 4,459 | 314 | 1,000/1,000 | 53.02 |
| 20,000 | 5,770 | 261 | 1,000/1,000 | 34.20 |
| 40,000 | 30,139 | 6,658 | 1,000/1,000 | 56.36 |

The 965-parameter model forms input-dependent attention scores during every block. Its two highest scores were separated enough for FP64 softmax to give exactly zero to every losing position on 11,104 batched head calls during TEST. The four other models use position-dependent attention routes that the frame can compute once. [Combined results](../results/maze/caps/summary.json) · [Private 1k Kaggle run](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-maze-caps-1/versions/1) · [Private 5k–40k Kaggle run](https://www.kaggle.com/code/ajayasaithambi/loopedwsp-maze-caps-5-10-20-40/versions/1)

The inspector reloads each archived NPZ and counts every scalar in every embedding, Q/K/V and output projection, MLP, readout and bias, including zeros. It verifies each array's expected dense shape and checks a conservative FLOP upper bound. The model hash covers the tokenizer and network configuration as well as all tensor names, shapes, dtypes and contents. [Inspector and transformer](frame.py)

One character corresponds to one input token. Each 30×30 puzzle is presented as 929 characters, including its 29 newlines. The network writes the output characters in order. A puzzle scores one only when the output string matches the full reference after the grader's allowed trimming of outer newlines. All models use a declared budget of two blocks per token. Inference can stop once one additional block leaves the entire batch's hidden state exactly unchanged.

The first eight official training puzzles are the fixed CPU validation subset. The GPU job reproduces the CPU predictions in FP64 and verifies model hashes before opening the official test split. Every frozen model receives one evaluation on all 1,000 test puzzles. Both splits are pinned to Hugging Face revision `549754de3d67eebe57b6b22c7d226b9412d55b07`. The Kaggle jobs return aggregate scores. The local `c*/dev-ledger.json` files record development attempts, including the superseded first 1k attempt. [Evaluator](experiment.py) · [Official dataset](https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k)
