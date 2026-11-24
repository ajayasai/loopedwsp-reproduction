# Licenses and sources

The reproduction contributors license their original code, documentation, analytic weight tensors, and generated synthetic inputs, labels, predictions, plots, and reports under the [MIT License](LICENSE). The following notices identify material with separate authorship or terms.

## Looped Weight-Space Programming

The experiments build on *Looped Weight-Space Programming* by Abdelrahman Elsayed, Ahmed Heakl, Abdelrahman Shaker, Fahad Khan, Salman Khan, and Claude Opus 5.5, as credited by the authors. Their publication supplies the method, task briefs, and comparison results. The reproduction contributors wrote this release's model builders and ran the evaluations reported here. The upstream MIT notice names Abdelrahman Elsayed and Ahmed Heakl as copyright holders. A verbatim copy accompanies this release and applies to the adapted material below. [Study](https://mbzuai-oryx.github.io/LoopedWSP/) · [Pinned upstream release](https://github.com/mbzuai-oryx/LoopedWSP/tree/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c) · [Upstream license](third_party/LoopedWSP-LICENSE.txt)

| Local material | Origin and adaptation |
| --- | --- |
| `scale/corpus.py` | The large-maze generator follows the sampling order of the upstream `random_input` maze branch. This release fixes fresh seeds, flattens the grids, and independently computes the shortest-path labels. [Upstream generator](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/weights/run.py) |
| `lengths/graph.py` | The directed-graph topology generator adapts the interactive demo's planted path, side branches, extra edges, and edge reversal. The reproduction stratifies distances, randomly renames nodes, shuffles tokens, and grades with breadth-first search. [Upstream demo implementation](https://github.com/mbzuai-oryx/LoopedWSP/blob/b64314d6c0b5d73bed223a8fe94cece034de4a5a/demo/graph_task.js) · [Graph task brief](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/prompts/07_graph_maps_network_brief.txt) |
| `lengths/chains.py` and `lengths/parity.py` | The chain distribution follows the upstream runner's shuffled variable assignments, bounded integer values, and signed increments. The parity task follows its binary-string interface. The reproduction uses its own seeded generators, structured token representation, and graders. [Upstream task runner](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/weights/run.py) |

The upstream license copy is 1,092 bytes. Its SHA-256 is `06b0af64ac2a3f15aeb794cc17fcb0a4184d53a48fd47f797b6be4a3047e6cbc`. [Pinned license file](https://github.com/mbzuai-oryx/LoopedWSP/blob/2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/LICENSE)

## Official Maze-Hard data

Sapient Intelligence released the official Maze-Hard training and test grids for *Hierarchical Reasoning Model*. The reproduction pins dataset revision `549754de3d67eebe57b6b22c7d226b9412d55b07`. These external grids and their supplied labels are outside the reproduction contributors' license grant. Fetch the CSVs from the original host and consult the dataset publisher for reuse terms. The public results summarize exact-grid matches and identify the input revision and hashes. [Dataset files](https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k/tree/549754de3d67eebe57b6b22c7d226b9412d55b07) · [HRM paper](https://arxiv.org/abs/2506.21734)

## Puzzles and applications

The Tower of Hanoi and Checker Jumping evaluations follow the rules described by Shojaee and colleagues in *The Illusion of Thinking*, Appendix A.1. The reproduction supplies its own model formulas, move streams, and replay implementations. The inventory, scheduling, and telemetry cases are synthetic examples created for this reproduction. [Puzzle specifications](https://arxiv.org/html/2506.06941v1#A1)

## External runtime dependencies

Python, NumPy, PyTorch, NVIDIA CUDA, Kaggle tooling, and other separately installed packages and services retain their respective licenses and terms. This release distributes the reproduction's code and experiment artifacts. Users obtain the runtime packages separately.
