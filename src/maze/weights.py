"""Analytic weights for a recurrent softmax-attention/ReLU transformer."""

import numpy as np


CHARS = " #SG"
OUTPUTS = " #SGo"
D_MODEL = 24
HEADS = 4
D_HEAD = D_MODEL // HEADS
ALPHA = 1024

# Indices in the shared residual stream. Two trailing features are unused.
ONE, ROW, COL, NORM2, WALL, START, GOAL, REACHED, PATH = range(9)
PARENT = (9, 10, 11, 12)
SEND = (13, 14, 15, 16)
NEAR = (17, 18, 19, 20)
INCOMING = 21

# The direction order is fixed by the development-split audit. An entry in
# OPPOSITE indexes the direction in which that neighbor sends back to us.
DIRECTIONS = ((1, 0), (0, -1), (0, 1), (-1, 0))
OPPOSITE = (3, 2, 1, 0)


def _layer(width):
    return {
        "q": np.zeros((HEADS, D_MODEL, D_HEAD)),
        "k": np.zeros((HEADS, D_MODEL, D_HEAD)),
        "v": np.zeros((HEADS, D_MODEL, D_HEAD)),
        "o": np.zeros((HEADS, D_HEAD, D_MODEL)),
        "w1": np.zeros((D_MODEL, width)),
        "b1": np.zeros(width),
        "w2": np.zeros((width, D_MODEL)),
        "b2": np.zeros(D_MODEL),
    }


def build_weights(side):
    """Return a single shared block, its embeddings and its linear readout."""
    if not isinstance(side, (int, np.integer)) or side < 1:
        raise ValueError("side must be a positive integer")

    token = np.zeros((len(CHARS), D_MODEL))
    token[:, ONE] = 1
    token[CHARS.index("#"), WALL] = 1
    token[CHARS.index("S"), [START, PATH]] = 1
    token[CHARS.index("G"), [GOAL, REACHED]] = 1

    position = np.arange(side, dtype=np.float64)
    row = np.zeros((side, D_MODEL))
    col = np.zeros((side, D_MODEL))
    row[:, ROW] = col[:, COL] = position
    row[:, NORM2] = col[:, NORM2] = position**2

    flood = _layer(10)
    # Q.K/sqrt(d_head) is -alpha * squared distance to the requested
    # neighbor, up to an additive constant shared by all keys in this row.
    score_scale = ALPHA * np.sqrt(D_HEAD)
    for direction, (dr, dc) in enumerate(DIRECTIONS):
        q, k = flood["q"][direction], flood["k"][direction]
        q[ROW, 0] = q[COL, 1] = 2 * score_scale
        q[ONE, :3] = (2 * dr * score_scale, 2 * dc * score_scale, -score_scale)
        k[ROW, 0] = k[COL, 1] = k[NORM2, 2] = 1

        v, o = flood["v"][direction], flood["o"][direction]
        v[REACHED, 0] = v[SEND[OPPOSITE[direction]], 1] = 1
        o[0, NEAR[direction]] = o[1, INCOMING] = 1

        # Unit direction turns on only for the first eligible neighbor, and
        # only at this cell's first arrival. The residual latches its result.
        w1, w2 = flood["w1"], flood["w2"]
        w1[NEAR[direction], direction] = 1
        w1[list(NEAR[:direction]), direction] = -1
        w1[[WALL, REACHED], direction] = -1
        w2[direction, [PARENT[direction], REACHED]] = 1

        # Clear the nonnegative attention scratch fields for the next round.
        clear_near = 4 + direction
        w1[NEAR[direction], clear_near] = 1
        w2[clear_near, NEAR[direction]] = -1

    # path' = min(1, path + incoming). Reuse the incoming scratch-clear unit.
    flood["w1"][INCOMING, 8] = 1
    flood["w2"][8, [INCOMING, PATH]] = (-1, 1)
    flood["w1"][[INCOMING, PATH], 9] = 1
    flood["b1"][9] = -1
    flood["w2"][9, PATH] = -1

    route = _layer(4)
    for direction in range(HEADS):
        # send' = send OR (path AND parent), with irreversible parent/path.
        route["w1"][[PATH, PARENT[direction]], direction] = 1
        route["w1"][SEND[direction], direction] = -1
        route["b1"][direction] = -1
        route["w2"][direction, SEND[direction]] = 1

    readout = np.zeros((D_MODEL, len(OUTPUTS)))
    bias = np.zeros(len(OUTPUTS))
    bias[OUTPUTS.index(" ")] = 0.5
    readout[PATH, OUTPUTS.index("o")] = 1
    for feature, char in ((WALL, "#"), (START, "S"), (GOAL, "G")):
        readout[feature, OUTPUTS.index(char)] = 2

    return {"chars": CHARS, "outputs": OUTPUTS, "token": token, "row": row,
            "col": col, "layers": [flood, route], "readout": readout, "bias": bias}
