"""Dense float64 check on independently generated mazes, including text boundaries."""

from collections import deque
import random

import numpy as np

import write_weights as m


def predict(text, spec, arrays):
    row = col = 0
    vectors = []
    for char in text:
        h = arrays["e0"][m.VOCAB.index(char)].copy()
        for i, field in enumerate(spec["fields"][1:], 1):
            value = {"line": row, "column": col,
                     "squared_norm": (row - 15)**2 + (col - 15)**2}[field["coordinate"]]
            h += arrays[f"e{i}"][(value >> field["place"]) % 2]
        vectors.append(h)
        if char == "\n":
            row, col = row + 1, 0
        else:
            col += 1
    h = np.array(vectors)
    soft_rows = 0
    for _ in range(2 * len(text)):
        old = h.copy()
        for layer in range(2):
            p = f"l{layer}.h0."
            q, k, v, o = [arrays[p + key] for key in ("q", "k", "v", "o")]
            scores = (h @ q) @ (h @ k).T / 2
            probs = np.exp(scores - scores.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            soft_rows += np.count_nonzero((probs != 0).sum(axis=1) != 1)
            h += (probs @ (h @ v)) @ o
            p = f"l{layer}."
            h += np.maximum(0, h @ arrays[p + "w1"] + arrays[p + "b1"]) @ arrays[p + "w2"] + arrays[p + "b2"]
        if np.array_equal(h, old):
            break
    indices = (h @ arrays["readout"] + arrays["bias"]).argmax(axis=1)
    return "".join(spec["alphabet"][i] for i in indices), soft_rows


def target(text):
    grid = text.split("\n")
    height, width = len(grid), len(grid[0])
    cells = "".join(grid)
    goal, start = cells.index("G"), cells.index("S")

    def neighbors(pos):
        row, col = divmod(pos, width)
        return [(row + dy) * width + col + dx for dy, dx in ((1, 0), (0, -1), (0, 1), (-1, 0))
                if 0 <= row + dy < height and 0 <= col + dx < width]

    dist, work = {goal: 0}, deque([goal])
    while work:
        pos = work.popleft()
        for new in neighbors(pos):
            if new not in dist and cells[new] != "#":
                dist[new] = dist[pos] + 1
                work.append(new)
    if start not in dist:
        return text
    result, pos = list(cells), start
    while pos != goal:
        pos = next(new for new in neighbors(pos) if dist.get(new) == dist[pos] - 1)
        if pos != goal:
            result[pos] = "o"
    return "\n".join("".join(result[i:i + width]) for i in range(0, len(result), width))


def main():
    spec, arrays = m.build()
    rng = random.Random(4768)
    soft_rows = 0
    for _ in range(300):
        size = rng.randint(2, 8)
        start, goal = rng.sample(range(size * size), 2)
        cells = ["#" if rng.random() < 0.26 else " " for _ in range(size * size)]
        cells[start], cells[goal] = "S", "G"
        text = "\n".join("".join(cells[i:i + size]) for i in range(0, len(cells), size))
        found, soft = predict(text, spec, arrays)
        soft_rows += soft
        if found != target(text):
            raise AssertionError((text, found, target(text)))
    print(f"300/300 exact raw strings, {soft_rows} softmax rows with more than one nonzero")


if __name__ == "__main__":
    main()
