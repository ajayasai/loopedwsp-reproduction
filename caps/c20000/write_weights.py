"""Latch the first wavefront predecessor, then relay one path signal from S."""

import numpy as np


def build():
    # The input serializes a 30-by-30 grid with 29 intervening newlines.
    vocab = " #SG\n"
    alphabet = ["", " ", "#", "S", "G", "\n", "o"]
    pos, square, one, newline_key = range(4)
    invalid = [4, 5, 6, 7]
    source, goal, barrier, newline = range(8, 12)
    predecessor = [12, 13, 14, 15]
    signal = [16, 17, 18, 19]
    adjacent = [20, 21, 22, 23]
    incoming, width = 24, 25
    offsets = [31, -1, 1, -31]  # D, L, R, U
    opposite = [3, 2, 1, 0]
    fields, arrays = [], {}

    def field(description, rows):
        arrays[f"e{len(fields)}"] = np.zeros((rows, width), dtype=np.float64)
        fields.append(description)
        return arrays[f"e{len(fields) - 1}"]

    chars = field({"kind": "char", "vocab": vocab}, len(vocab) + 1)
    chars[vocab.index("S"), source] = chars[vocab.index("G"), goal] = 1
    chars[[vocab.index("#"), vocab.index("\n"), len(vocab)], barrier] = 1
    chars[vocab.index("\n"), newline] = 1
    # Add base-eight digits to assemble i and i^2 exactly.
    for power, places, axis in [(1, 4, pos), (2, 7, square)]:
        for place in range(places):
            e = field({"kind": "pos_digit", "power": power, "base": 8, "place": place}, 8)
            e[:, axis] = np.arange(8) * 8**place
    col = field({"kind": "pos_mod", "modulus": 31}, 31)
    col[:, one] = 1
    col[30, newline_key] = col[0, invalid[1]] = col[29, invalid[2]] = 1
    row = field({"kind": "pos_digit", "power": 1, "base": 31, "place": 1}, 31)
    row[0, invalid[3]] = row[29, invalid[0]] = 1

    # A head maximizes 2048*(2*(i+offset)*j-j^2). FP64 softmax is exactly
    # one-hot on this integer grid. Off-edge queries look up newline barriers.
    for h, delta in enumerate(offsets):
        head = f"l0.h{h}."
        q = np.zeros((width, 3))
        k = np.zeros_like(q)
        q[pos, 0], q[one, 0] = 4096, 4096 * delta
        q[one, 1], q[invalid[h], 2] = -2048, 2**40
        k[pos, 0] = k[square, 1] = k[newline_key, 2] = 1
        v = np.zeros((width, 2))
        v[[goal] + predecessor, 0] = 1
        v[signal[opposite[h]], 1] = 1
        o = np.zeros((2, width))
        o[0, adjacent[h]] = o[1, incoming] = 1
        arrays.update({head + key: mat for key, mat in zip("qkvo", (q, k, v, o))})

    units = []
    def unit(inputs, bias, dest, sign=1):
        units.append((inputs, bias, dest, sign))

    for h in range(4):
        # First contact opens exactly one predecessor. Earlier DLRU neighbors
        # inhibit later ones, and a latched predecessor inhibits all four.
        rule = {idx: -1 for idx in [goal, barrier] + predecessor + adjacent[:h]}
        rule[adjacent[h]] = 1
        unit(rule, 0, predecessor[h])
        # Only the start or a vertex receiving the route may signal its parent.
        unit({predecessor[h]: 1, source: 1, incoming: 1, signal[h]: -1}, -1, signal[h])
    for axis in adjacent + [incoming]:
        unit({axis: 1}, 0, axis, -1)
    w1, w2 = np.zeros((width, len(units))), np.zeros((len(units), width))
    b1 = np.zeros(len(units))
    for i, (rule, bias, target, sign) in enumerate(units):
        for axis, value in rule.items():
            w1[axis, i] = value
        b1[i], w2[i, target] = bias, sign
    arrays.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": np.zeros(width)})

    readout = np.zeros((width, len(alphabet)))
    readout[barrier, 2], readout[newline, 2] = 4, -4
    readout[source, 3] = readout[goal, 4] = readout[newline, 5] = 4
    readout[signal, 6] = 2
    bias = np.zeros(len(alphabet))
    bias[1] = 1
    arrays.update(readout=readout, bias=bias)
    return {"fields": fields, "alphabet": alphabet, "layers": [4], "rounds_per_token": 2}, arrays
