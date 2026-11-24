"""A max-plus distance potential followed by a deterministic ascent from S."""

import numpy as np


def build():
    width, side, stride = 29, 30, 31
    n = side * stride - 1
    height_at_goal = 1024
    alphabet = ["", " ", "#", "S", "G", "\n", "o"]
    vocab = " #SG\n"
    position, squared_position, constant = range(3)
    address = [3, 4, 5, 6]
    source, goal, blocked, newline, potential = range(7, 12)
    emit = [12, 13, 14, 15]
    neighbors = [16, 17, 18, 19]
    incoming, first_pair, second_pair, highest = range(20, 24)
    winners = [24, 25, 26, 27]
    ready = 28

    char_table = np.zeros((len(vocab) + 1, width), dtype=np.float64)
    char_table[vocab.index("S"), source] = char_table[vocab.index("G"), goal] = 1
    char_table[vocab.index("G"), potential] = height_at_goal
    char_table[[vocab.index("#"), vocab.index("\n"), len(vocab)], blocked] = 1
    char_table[vocab.index("\n"), newline] = 1
    # A directly indexed position table gives each of the four heads a target
    # in the serialized grid. The first newline is the sentinel for every edge.
    pos_table = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        row, col = divmod(i, stride)
        pos_table[i, [position, squared_position, constant]] = [i, i * i, 1]
        for h, (dr, dc) in enumerate(((1, 0), (0, -1), (0, 1), (-1, 0))):
            r, c = row + dr, col + dc
            valid = col < side and 0 <= r < side and 0 <= c < side
            pos_table[i, address[h]] = r * stride + c if valid else side

    arrays = {"e0": char_table, "e1": pos_table}
    for h, back in enumerate((3, 2, 1, 0)):
        q = np.zeros((width, 2))
        k = np.zeros((width, 2))
        q[address[h], 0], q[constant, 1] = 4096, -2048
        k[position, 0] = k[squared_position, 1] = 1
        v = np.zeros((width, 2))
        v[potential, 0] = v[emit[back], 1] = 1
        o = np.zeros((2, width))
        o[0, neighbors[h]] = o[1, incoming] = 1
        for key, mat in zip("qkvo", (q, k, v, o)):
            arrays[f"l0.h{h}.{key}"] = mat

    # Each unit is (input coefficients, bias, output coordinate, coefficient).
    # max(a,b) = ReLU(a-b) + b for the nonnegative integer potentials.
    pairwise = []
    for a, b, target in [(neighbors[0], neighbors[1], first_pair),
                         (neighbors[2], neighbors[3], second_pair)]:
        pairwise += [({a: 1, b: -1}, 0, target, 1), ({b: 1}, 0, target, 1)]
    reduce_max = [({first_pair: 1, second_pair: -1}, 0, highest, 1),
                  ({second_pair: 1}, 0, highest, 1)]

    # Bellman relaxation with a fixed height at G. The exact positive solution
    # is height_at_goal minus shortest distance. First positive arrivals have
    # their final height under synchronous unit-cost propagation.
    relax = [
        ({highest: 1, potential: -1, blocked: -height_at_goal}, -1, potential, 1),
        ({highest: 1}, -1, ready, 1), ({highest: 1}, -2, ready, -1),
    ]
    for h in range(4):
        # Integral neighbor heights never exceed their shared maximum.
        relax.append(({neighbors[h]: 1, highest: -1}, 1, winners[h], 1))

    trace = []
    for h in range(4):
        # Emit towards the DLRU-first maximal neighbor once the routed signal
        # is present and a positive height can reach this cell.
        condition = {source: 1, incoming: 1, ready: 1, winners[h]: 1,
                     emit[h]: -1, goal: -1, blocked: -1}
        condition.update({axis: -1 for axis in winners[:h]})
        trace.append((condition, -2, emit[h], 1))
    temporary = neighbors + [incoming, first_pair, second_pair, highest, ready] + winners
    trace.extend(({axis: 1}, 0, axis, -1) for axis in temporary)

    for layer, units in enumerate((pairwise, reduce_max, relax, trace)):
        w1, w2 = np.zeros((width, len(units))), np.zeros((len(units), width))
        b1 = np.zeros(len(units))
        for j, (inputs, bias, output, gain) in enumerate(units):
            for axis, coefficient in inputs.items():
                w1[axis, j] = coefficient
            b1[j], w2[j, output] = bias, gain
        for key, mat in (("w1", w1), ("b1", b1), ("w2", w2), ("b2", np.zeros(width))):
            arrays[f"l{layer}.{key}"] = mat
    out = np.zeros((width, len(alphabet)))
    out[blocked, 2], out[newline, 2] = 4, -4
    out[source, 3] = out[goal, 4] = out[newline, 5] = 4
    out[emit, 6] = 2
    bias = np.zeros(len(alphabet))
    bias[1] = 1
    arrays.update(readout=out, bias=bias)
    fields = [{"kind": "char", "vocab": vocab}, {"kind": "pos_mod", "modulus": n}]
    return {"fields": fields, "alphabet": alphabet, "layers": [4, 0, 0, 0],
            "rounds_per_token": 2}, arrays
