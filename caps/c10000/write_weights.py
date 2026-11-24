"""Bellman distance relaxation followed by directed path transport."""

import json
from pathlib import Path

import numpy as np


def build():
    vocab = " #SG\n"
    alphabet = ["", " ", "#", "S", "G", "\n", "o"]
    spec = {
        "fields": [{"kind": "char", "vocab": vocab},
                   {"kind": "pos_digit", "power": 1, "base": 31, "place": 1},
                   {"kind": "pos_digit", "power": 1, "base": 31, "place": 0}],
        "alphabet": alphabet, "layers": [4, 0, 0, 0, 0], "rounds_per_token": 2,
    }
    width = 23
    infinity = 2048  # Exceeds every possible finite distance in the 900-cell grid.
    one, y, x, norm, blocked, start, goal, newline, distance, path = range(10)
    outgoing = (10, 11, 12, 13)
    neighbor = (14, 15, 16, 17)
    pair_a, pair_b, nearest, incoming, rank = (18, 19, 20, 21, 22)
    a = {}

    def matrix(key, *shape):
        a[key] = np.zeros(shape, dtype=np.float64)
        return a[key]

    def mlp(layer, width_ff):
        prefix = f"l{layer}."
        w1 = matrix(prefix + "w1", width, width_ff)
        b1 = matrix(prefix + "b1", width_ff)
        w2 = matrix(prefix + "w2", width_ff, width)
        matrix(prefix + "b2", width)
        return w1, b1, w2

    chars = matrix("e0", len(vocab) + 1, width)
    chars[:, one], chars[:, distance] = 1, infinity
    chars[vocab.index("S"), [start, path]] = 1
    chars[vocab.index("G"), [goal, distance]] = (1, 0)
    chars[[vocab.index("#"), vocab.index("\n")], blocked] = 1
    chars[vocab.index("\n"), newline] = 1
    for key, channel in (("e1", y), ("e2", x)):
        coordinates = np.arange(31, dtype=np.float64)
        if channel == x:
            coordinates[-1] = 64
        table = matrix(key, 31, width)
        table[:, channel] = coordinates
        table[:, norm] = coordinates**2

    for i, (dy, dx) in enumerate(((1, 0), (0, -1), (0, 1), (-1, 0))):
        prefix = f"l0.h{i}."
        q, k = matrix(prefix + "q", width, 3), matrix(prefix + "k", width, 3)
        scale = 1024 * np.sqrt(3)
        q[one] = [2*dy*scale, 2*dx*scale, -scale]
        q[y, 0], q[x, 1] = 2*scale, 2*scale
        k[y, 0], k[x, 1], k[norm, 2] = 1, 1, 1
        v, o = matrix(prefix + "v", width, 2), matrix(prefix + "o", 2, width)
        v[distance, 0], o[0, neighbor[i]] = 1, 1
        v[outgoing[3-i], 1], o[1, incoming] = 1, 1

    # Compute min(d_down,d_left) and min(d_right,d_up) in parallel.
    # Each distance is nonnegative, so min(a,b) = ReLU(a) - ReLU(a-b).
    w1, b1, w2 = mlp(0, 6)
    for offset, result in ((0, pair_a), (2, pair_b)):
        first, second = neighbor[offset:offset+2]
        w1[first, offset:offset+2] = 1
        w1[second, offset+1] = -1
        w2[offset:offset+2, result] = (1, -1)
    w1[incoming, 4], w2[4, [incoming, path]] = 1, (-1, 1)
    w1[[incoming, path], 5], b1[5], w2[5, path] = 1, -1, -1

    w1, b1, w2 = mlp(1, 3)
    w1[pair_a, 0:2], w1[pair_b, 1], w1[pair_b, 2] = 1, -1, 1
    w2[0:2, nearest], w2[0, pair_a], w2[2, pair_b] = (1, -1), -1, -1

    # d' = min(d, 1+minimum neighbor distance) on passable tokens.
    w1, b1, w2 = mlp(2, 2)
    w1[distance, 0], w1[nearest, 0], w1[blocked, 0] = 1, -1, -infinity
    b1[0], w2[0, distance] = -1, -1
    w1[nearest, 1], w2[1, nearest] = 1, -1

    # Mark exactly the lower-distance neighbors of an active token. The
    # adjacent grid distances differ by at most one once both are finite.
    w1, b1, w2 = mlp(3, 12)
    priorities = (8, 4, 2, 1)
    for i, field in enumerate(neighbor):
        for k, sign in ((0, 1), (1, -1)):
            unit = 2*i + k
            w1[distance, unit], w1[field, unit] = 1, -1
            w1[path, unit], w1[blocked, unit] = infinity, -infinity
            b1[unit] = -infinity - k
            w2[unit, rank] = priorities[i] * sign
        w1[field, 8+i], w2[8+i, field] = 1, -1

    # Integer intervals [8,15], [4,7], [2,3], [1,1] uniquely choose
    # the DLRU-first descent direction; rank zero sends no outgoing edge.
    w1, b1, w2 = mlp(4, 13)
    for i, cut in enumerate(priorities):
        for k, sign in ((0, 1), (1, -1)):
            unit = 2*i + k
            w1[rank, unit] = 1
            b1[unit] = 1 - cut - k
            w2[unit, outgoing[i]] = sign
            if i < 3:
                w2[unit, outgoing[i+1]] = -sign
        w1[outgoing[i], 8+i], w2[8+i, outgoing[i]] = 1, -1
    w1[rank, 12], w2[12, rank] = 1, -1

    out = matrix("readout", width, len(alphabet))
    bias = matrix("bias", len(alphabet))
    bias[alphabet.index(" ")] = 0.5
    for channel, symbol, value in ((blocked, "#", 2), (start, "S", 2), (goal, "G", 2),
                                   (newline, "\n", 3), (path, "o", 1)):
        out[channel, alphabet.index(symbol)] = value
    return spec, a


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    specification, arrays = build()
    (directory / "network.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    np.savez(directory / "weights.npz", **arrays)
