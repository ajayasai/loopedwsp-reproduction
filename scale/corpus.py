"""Generate fixed larger mazes and score-side labels, independently of the model."""

from collections import deque
from hashlib import sha256
import random
from statistics import median


# Fresh seeds declared before looking at any generated mazes or model outputs.
SPLITS = {60: (50, 2026112460), 100: (40, 2026112500)}
GENERATOR = (
    "https://github.com/mbzuai-oryx/LoopedWSP/blob/"
    "2752cd9f2d2e1e3e30e6686b3dbc23d8bcab875c/weights/run.py"
)
STUDY = "https://mbzuai-oryx.github.io/LoopedWSP/"
CARVE_STEPS = ((2, 0), (-2, 0), (0, 2), (0, -2))


def _draw(side, rng):
    """Follow the published random_input('maze', side) sampling order."""
    rows = [["#"] * side for _ in range(side)]
    rows[1][1] = " "
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        choices = []
        for dr, dc in CARVE_STEPS:
            y, x = r + dr, c + dc
            if 0 < y < side - 1 and 0 < x < side - 1 and rows[y][x] == "#":
                choices.append((y, x))
        if choices:
            y, x = rng.choice(choices)
            rows[(r + y) // 2][(c + x) // 2] = rows[y][x] = " "
            stack.append((y, x))
        else:
            stack.pop()

    for _ in range(side):
        rows[rng.randrange(1, side - 1)][rng.randrange(1, side - 1)] = " "
    open_cells = [(y, x) for y, row in enumerate(rows) for x, char in enumerate(row) if char == " "]
    (sr, sc), (gr, gc) = rng.sample(open_cells, 2)
    rows[sr][sc], rows[gr][gc] = "S", "G"
    return "".join("".join(row) for row in rows)


def _reference(grid, side):
    """Forward BFS first-discovery parents yield the DLRU-first shortest route."""
    start, goal = grid.index("S"), grid.index("G")
    distance, ways, parent = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        if p == goal:
            break
        r, c = divmod(p, side)
        d = distance[p] + 1
        for y, x in ((r + 1, c), (r, c - 1), (r, c + 1), (r - 1, c)):
            if not (0 <= y < side and 0 <= x < side):
                continue
            n = y * side + x
            if grid[n] == "#":
                continue
            if n not in distance:
                parent[n], distance[n], ways[n] = p, d, ways[p]
                queue.append(n)
            elif distance[n] == d:
                ways[n] += ways[p]
    if goal not in distance:
        return None
    solved = list(grid)
    p = parent[goal]
    while p != start:
        solved[p] = "o"
        p = parent[p]
    return "".join(solved), distance[goal], ways[goal]


def _corpus_digest(rows):
    """SHA-256 of ordered, flattened ASCII grids, each followed by LF."""
    return sha256("".join(row + "\n" for row in rows).encode("ascii")).hexdigest()


def get_corpus(side):
    """Return flattened ASCII inputs, exact targets, and a reproducibility record."""
    count, seed = SPLITS[side]
    rng = random.Random(seed)
    questions, targets, lengths, route_counts = [], [], [], []
    discarded = 0
    while len(questions) < count:
        grid = _draw(side, rng)
        result = _reference(grid, side)
        if result is None:
            discarded += 1
            continue
        target, length, paths = result
        questions.append(grid)
        targets.append(target)
        lengths.append(length)
        route_counts.append(paths)

    info = {
        "side": side,
        "count": count,
        "seed": seed,
        "study_url": STUDY,
        "generator_url": GENERATOR,
        "provenance": "Fresh seeded instances from the authors' public demo generator. "
                      "The study reports these sizes and counts; its individual inputs and seeds were unavailable.",
        "sampling": "One CPython random.Random stream per size. DFS carve with extra openings, uniformly sampled "
                    "distinct open endpoints. Keep the first reachable pairs; no difficulty or tie filter.",
        "discarded_unreachable": discarded,
        "label_oracle": "Forward BFS from S with first-discovery parents in down, left, right, up order. "
                        "Interior path cells are marked o. Endpoints and all other cells are preserved.",
        "hash_format": "SHA-256 of ordered, row-major ASCII grids with one trailing LF per grid.",
        "questions_sha256": _corpus_digest(questions),
        "targets_sha256": _corpus_digest(targets),
        "path_steps": {"min": min(lengths), "median": median(lengths), "max": max(lengths)},
        "path_steps_per_maze": lengths,
        "grids_with_multiple_shortest_paths": sum(n > 1 for n in route_counts),
    }
    return questions, targets, info
