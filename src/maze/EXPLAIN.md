# A hand-written Maze-Hard transformer

The model uses one token per cell and repeats two residual attention/MLP layers. All projections, biases and embeddings come from `weights.py`. Each parameter follows from a formula. On the official development split, every maze had multiple shortest paths. Only down, left, right, up consistently matched the labeled choice when moving from the start toward the goal.

## Residual stream

| Index | Field | Meaning |
| --- | --- | --- |
| 0 | Constant | One at every cell |
| 1–3 | Geometry | Row, column and their squared norm |
| 4–6 | Tile | Wall, start and goal indicators |
| 7 | Reached | The wave from the goal has visited this cell |
| 8 | Path | The route from the start has visited this cell |
| 9–12 | Parent | Direction of the chosen neighbor one step closer to the goal |
| 13–16 | Send | This path cell sends toward its chosen parent |
| 17–20 | Nearby reach | Four neighboring reach bits, cleared by the first MLP |
| 21 | Incoming | Sum of the four neighboring sends pointing here, cleared by the first MLP |
| 22–23 | Unused | Zero |

Directions always appear in down, left, right, up order. Character embeddings set the constant and tile fields, seed `Reached` at the goal and seed `Path` at the start. The sum of two positional embeddings adds row, column and squared norm to each token.

## First layer

For a token at position `p`, the head for direction `δ` points at `p+δ`. The scaled query/key dot product for a key at `j` equals `α(2(p+δ)·j − ‖j‖²)` with `α = 1024`. Within a query, this differs from `−α‖j−(p+δ)‖²` by a constant. The nearest key is unique. At a boundary, an outward query selects its own cell. On the 30×30 grid the score margin makes all other exponentials underflow to zero in float32 and float64, giving one-hot softmax probabilities. The handmade selector test checks every query in both dtypes. Each head retrieves the neighbor's reach flag and its send flag for the reverse direction.

Name the four neighboring reach bits `b_j`. The MLP computes `a_j = ReLU(b_j − sum(b_i for i<j) − Wall − Reached)`. On the first visit, exactly the first eligible neighbor sets its `a_j`. The residual update adds this bit to the corresponding parent field and to `Reached`. The goal starts reached and never selects a parent. A wall cannot acquire one. A boundary self-read cannot cause a first visit.

The same MLP updates `Path` to `min(1, Path + Incoming)` using the incoming sum and a single ReLU at one. It also removes the five temporary attention results. A boundary self-read can only return an active send when the cell is already on the path, so the saturation absorbs it.

## Second layer and readout

The second attention output is identically zero. Its MLP adds `ReLU(Path + Parent_j − Send_j − 1)` to each `Send_j`. Path and parent bits stay set once acquired. The resulting `Send_j` is therefore the conjunction of path membership and that parent direction. On the next round, the selected neighbor sees the incoming send.

The per-cell linear readout gives wall, start and goal logits of two when the tile indicator is set. It gives the path marker `o` the path bit as its logit. The blank logit is always one half. This preserves both endpoints and marks only the interior cells of the selected route.

## Loop bound

After `t` rounds, exactly the traversable cells at graph distance at most `t` from the goal have `Reached=1`. A cell reached at round `t` chooses the first neighbor at distance `t−1` under the verified move order. When the start is reached at round `L`, where `L` is its shortest-path distance, it sends its first route signal. Every further round advances the route one cell down the stored pointers, reaching the goal by round `2L`.

There are `n = side²` cells and every shortest path has at most `n−1` edges. All reachable cells have selected their final parent by round `n−1`, and the entire state is stationary by `2(n−1)`. The benchmark uses fixed `R_min = 1800` at 30×30. The requested snapshots fall at 1800, 3600 and 7200. An evaluator can certify later snapshots by detecting exact equality of **all** residual entries after one complete shared block.

## Parameter count at 30×30

The residual width is 24. Each layer has four heads of width six. The first and second MLP widths are ten and four. Counts include the zero attention tensors of the second layer and the two formula-generated positional tables.

| Component | Dense scalars | Nonzero scalars |
| --- | ---: | ---: |
| Token, row and column tables | 1,536 | 125 |
| First layer | 2,818 | 85 |
| Second layer | 2,524 | 20 |
| Readout and bias | 125 | 5 |
| **Total** | **7,003** | **235** |
