"""A 4-bit arrival mask selects and latches a scalar predecessor."""

import json
from pathlib import Path

import numpy as np


def build():
    vocab = " #SG\n"
    alphabet = ["", " ", "#", "S", "G", "\n", "o"]
    fields = [{"kind": "char", "vocab": vocab}]
    fields.extend({"kind": "pos_digit", "power": 1, "base": 31, "place": i}
                  for i in (1, 0))
    spec = {"fields": fields, "alphabet": alphabet, "layers": [4, 0], "rounds_per_token": 2}
    width = 17
    one, y, x, norm, blocked, start, goal, newline, seen, path, parent = range(11)
    send = (11, 12, 13, 14)
    mask, incoming = 15, 16
    a = {}

    def matrix(key, *shape):
        a[key] = np.zeros(shape, dtype=np.float64)
        return a[key]

    chars = matrix("e0", len(vocab) + 1, width)
    chars[:, one] = 1
    chars[[vocab.index("#"), vocab.index("\n")], blocked] = 1
    chars[vocab.index("S"), [start, path]] = 1
    chars[vocab.index("G"), [goal, seen]] = 1
    chars[vocab.index("\n"), newline] = 1
    for key, channel in (("e1", y), ("e2", x)):
        coordinates = np.arange(31, dtype=np.float64)
        if channel == x:
            coordinates[30] = 64  # Absolute text positions of newline tokens.
        table = matrix(key, 31, width)
        table[:, channel], table[:, norm] = coordinates, coordinates**2

    moves = ((1, 0), (0, -1), (0, 1), (-1, 0))
    bits = (8, 4, 2, 1)
    for i, (dy, dx) in enumerate(moves):
        prefix = f"l0.h{i}."
        q, k = matrix(prefix + "q", width, 3), matrix(prefix + "k", width, 3)
        gain = 1024 * np.sqrt(3)
        q[y, 0], q[x, 1], q[one] = 2*gain, 2*gain, (2*dy*gain, 2*dx*gain, -gain)
        k[y, 0], k[x, 1], k[norm, 2] = 1, 1, 1
        v, o = matrix(prefix + "v", width, 2), matrix(prefix + "o", 2, width)
        v[seen, 0], v[send[3-i], 1] = 1, 1
        o[0, mask], o[1, incoming] = bits[i], 1

    # On first arrival, select a scalar code (D,L,R,U = 1,2,3,4)
    # by testing the most significant nonzero bit of the 4-bit mask.
    w1, b1 = matrix("l0.w1", width, 11), matrix("l0.b1", 11)
    w2 = matrix("l0.w2", 11, width)
    matrix("l0.b2", width)
    for j, (cut, increment) in enumerate(((1, 4), (2, -1), (4, -1), (8, -1))):
        for k, sign in ((0, 1), (1, -1)):
            unit = 2*j + k
            w1[mask, unit] = 1
            w1[[blocked, seen], unit] = -16
            b1[unit] = 1 - cut - k
            w2[unit, parent] = sign * increment
            if cut == 1:
                w2[unit, seen] = sign
    w1[mask, 8], w2[8, mask] = 1, -1
    w1[incoming, 9], w2[9, [incoming, path]] = 1, (-1, 1)
    w1[[incoming, path], 10], b1[10], w2[10, path] = 1, -1, -1

    # Linear combinations of integer step functions decode that scalar to
    # four directed output bits, gated by membership on the selected path.
    w1, b1 = matrix("l1.w1", width, 12), matrix("l1.b1", 12)
    w2 = matrix("l1.w2", 12, width)
    matrix("l1.b2", width)
    for i in range(4):
        for k, sign in ((0, 1), (1, -1)):
            unit = 2*i + k
            w1[parent, unit], w1[path, unit] = 1, 8
            b1[unit] = -8 - i - k
            w2[unit, send[i]] = sign
            if i > 0:
                w2[unit, send[i-1]] = -sign
        w1[send[i], 8+i], w2[8+i, send[i]] = 1, -1

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
