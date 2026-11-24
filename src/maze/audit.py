"""Verify the official development split using a scorer-side BFS oracle."""

import csv
import hashlib
import json
from collections import Counter, deque
from itertools import permutations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REVISION = "549754de3d67eebe57b6b22c7d226b9412d55b07"
SIDE = 30
DIRECTIONS = {-SIDE: "U", SIDE: "D", -1: "L", 1: "R"}
ORDERS = ["".join(order) for order in permutations("UDLR")]


def neighbors(index):
    row, col = divmod(index, SIDE)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        y, x = row + dy, col + dx
        if 0 <= y < SIDE and 0 <= x < SIDE:
            yield y * SIDE + x


def shortest_paths(start, allowed):
    distances = {start: 0}
    counts = {start: 1}
    pending = deque([start])
    while pending:
        cell = pending.popleft()
        step = distances[cell] + 1
        for nxt in neighbors(cell):
            if nxt not in allowed:
                continue
            if nxt not in distances:
                distances[nxt] = step
                counts[nxt] = counts[cell]
                pending.append(nxt)
            elif distances[nxt] == step:
                counts[nxt] += counts[cell]
    return distances, counts


def matching_orders(start, goal, marked, target_distances):
    """Derive the priority constraints imposed by each choice on this path."""
    precedence = set()
    cell = start
    while cell != goal:
        remaining = target_distances[cell] - 1
        choices = [n for n in neighbors(cell) if target_distances.get(n) == remaining]
        marked_choices = [n for n in choices if n in marked]
        if len(marked_choices) != 1:
            raise ValueError("Ambiguous successor inside the labeled path")
        chosen = marked_choices[0]
        winner = DIRECTIONS[chosen - cell]
        precedence.update((winner, DIRECTIONS[n - cell]) for n in choices if n != chosen)
        cell = chosen
    return [order for order in ORDERS if all(order.index(a) < order.index(b) for a, b in precedence)]


def examine(question, answer):
    if len(question) != SIDE**2 or len(answer) != SIDE**2:
        raise ValueError("Grid has an unexpected length")
    if question.count("S") != 1 or question.count("G") != 1:
        raise ValueError("Expected one start and one goal")
    if set(question) - set("# SG") or set(answer) - set("# SGo"):
        raise ValueError("Grid has an unexpected symbol")
    if answer.replace("o", " ") != question:
        raise ValueError("Target changed a wall, endpoint, or existing symbol")

    start, goal = question.index("S"), question.index("G")
    walkable = {i for i, char in enumerate(question) if char != "#"}
    marked = {i for i, char in enumerate(answer) if char in "SGo"}
    distances, counts = shortest_paths(start, walkable)
    marked_distances, _ = shortest_paths(start, marked)
    if goal not in distances:
        raise ValueError("Goal is unreachable")
    length = distances[goal]
    # A shortest route of length d has d+1 distinct cells. Requiring a route
    # of that length inside exactly d+1 marked cells excludes stray marks.
    if marked_distances.get(goal) != length or len(marked) != length + 1:
        raise ValueError("Marked cells do not form a shortest start-to-goal path")
    to_goal, _ = shortest_paths(goal, walkable)
    return (
        length, counts[goal],
        matching_orders(start, goal, marked, to_goal),
        matching_orders(goal, start, marked, distances),
    )


def main():
    source = ROOT / "data/maze/train.csv"
    lengths, counts = [], []
    forward_orders, reverse_orders = Counter(), Counter()
    input_symbols, output_symbols = set(), set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line, row in enumerate(reader, 2):
            question, answer = row["question"], row["answer"]
            try:
                length, count, forward, reverse = examine(question, answer)
            except ValueError as exc:
                raise ValueError(f"train.csv line {line}: {exc}") from exc
            input_symbols.update(question)
            output_symbols.update(answer)
            lengths.append(length)
            counts.append(count)
            forward_orders.update(forward)
            reverse_orders.update(reverse)
    if len(counts) != 1000:
        raise ValueError(f"Expected 1000 development grids, found {len(counts)}")
    unique = counts.count(1)
    report = {
        "dataset": "sapientinc/maze-30x30-hard-1k",
        "revision": REVISION,
        "split": "train",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "grids": len(counts),
        "shape": [SIDE, SIDE],
        "input_symbols": sorted(input_symbols),
        "output_symbols": sorted(output_symbols),
        "target_convention": "Preserve walls and endpoints; mark interior path cells with o.",
        "valid_shortest_path_labels": len(counts),
        "unique_shortest_paths": unique,
        "multiple_shortest_paths": len(counts) - unique,
        "shortest_path_steps": {"min": min(lengths), "max": max(lengths)},
        "max_shortest_path_count": max(counts),
        "tie_break": {
            "method": "At each cell choose the first neighbor in the listed order that is one step closer to the target. U,D,L,R mean up, down, left, right.",
            "tested_orders_per_direction": len(ORDERS),
            "start_to_goal_perfect_orders": [order for order in ORDERS if forward_orders[order] == len(counts)],
            "goal_to_start_perfect_orders": [order for order in ORDERS if reverse_orders[order] == len(counts)],
            "start_to_goal_best_matches": max(forward_orders.values(), default=0),
            "goal_to_start_best_matches": max(reverse_orders.values(), default=0),
        },
    }
    destination = ROOT / "results/maze/uniqueness.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
