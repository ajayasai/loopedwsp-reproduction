# Pointer chasing and directed reachability

This experiment compares analytically specified transformers at equal stored parameter counts. Each task has two budgets, ten path lengths and an independent development split. The test set contains 64 counterfactual pairs per task at each length, or 2,560 queries in all. Every model within a task sees the same examples. The experiment holds the graph at 64 vertices while varying the number of links needed to answer. [Frozen design](protocol.json) · [Corpus generator](corpus.py) · [Exact inputs](cases.npz)

## Pointer chasing

Given a successor for each vertex, source ID and hop count k, return the ID reached after exactly k successor lookups. Each pair changes the last edge of the common path and swaps a second edge to preserve every vertex's indegree. Its two answers differ. Half the pairs use permutations, the others use many-to-one functions. Graph labels and input table row order are randomized. The evaluator indexes rows by their supplied IDs once before timing.

A stage uses softmax attention to compose the current successor table with itself. The same attention evaluates the next query position, and a ReLU gate selects it when the corresponding bit of k is set. A D-stage model can represent hop counts through 2ᴰ − 1. The two- and four-stage controls explicitly clip larger requests to 3 and 15 respectively. Their out-of-range scores measure that design choice. They do not measure the best possible accuracy of every shallow transformer. No stage uses the correct endpoint to stop.

| Budget | Tied, 6 passes | Untied, 2 stages | Untied, 4 stages | Untied, 6 stages | Full-width, untied control |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,072 | 24 | 12 | 6 | 4 | 24 (18,432 parameters) |
| 6,144 | 48 | 24 | 12 | 8 | 48 (36,864 parameters) |

Cells show address width per stage. Each independently stored stage contains two 64 × width Fourier address projections. Width changes the number of Fourier components in the address scores. All models use fixed one-hot IDs and identity value and output maps. The control at the right copies the tied model's weights into six independently stored stages and performs the same arithmetic. [Pointer model](pointer.py)

## Directed reachability

Given a directed adjacency matrix, source ID and target ID, decide if **any** directed path connects them. Each positive has shortest distance k. Its partner swaps two edges, preserving the degree of every vertex and making the target unreachable. Graphs have branches and directed cycles. The source and target stay the same within a pair. The reported distance for an unreachable query refers to its positive partner. The model sees only the adjacency and the two IDs.

The local model carries one reached bit per vertex. Each attention stage advances the reached set by at most one edge. Its tied version runs 63 passes for every query. The global model stores a row of the reachability relation at every vertex, starting with adjacency plus the identity. Each stage composes the relation with itself using dot-product softmax attention and a residual ReLU clamp. Four stages cover 16 edges and six cover all simple paths on 64 vertices. Global attention uses 4,096 changing relation coordinates per graph compared with 64 changing vertex coordinates for local propagation. Both models also use the supplied 64 × 64 adjacency.

| Budget | Tied local, 63 passes | Untied local, 1 / 2 / 4 stages | Untied global, 4 / 6 stages | Full-width, 63-stage untied control |
| ---: | ---: | ---: | ---: | ---: |
| 52 | 11 heads | 11 / 5 / 2 | 2 / 1 | 11 heads (3,028 parameters) |
| 100 | 23 heads | 23 / 11 / 5 | 5 / 3 | 23 heads (6,052 parameters) |

Each head handles a disjoint subset of vertex IDs and stores four scalar Q/K/V/O gains, shared across coordinates. Each stored stage also has four scalar residual-MLP entries, and the final classifier stores four more. The local model uses input edges to mask attention. The global model uses fixed ID coordinates and can attend across the graph. The budgets exclude fixed wiring, masks and working memory. They measure storage under these particular parameterizations. [Reachability model](reachability.py)

## Running the study

With NumPy and PyTorch installed, run the following from the repository root. The development command evaluates short paths on CPU. The GPU command requires CUDA and writes a results JSON and a prediction archive. Both commands regenerate the cases and verify the hash of every input array. Use an empty output directory to keep runs separate.

```sh
python graph_depth/experiment.py dev
python graph_depth/experiment.py gpu --output audit-output/graph-depth-gpu
```

To submit on Kaggle, authenticate its CLI, set `LOOPEDWSP_KAGGLE_OWNER` to your account and use `python graph_depth/submit_study.py --push`. The launcher requests a private Tesla T4 run with internet off. Collect a completed version with `python graph_depth/submit_study.py --collect VERSION`. It checks the returned source, GPU settings and output hashes against the submission. [Kaggle packager](submit_study.py)
