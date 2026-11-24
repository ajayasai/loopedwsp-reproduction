"""Write an analytic raw-text maze network with at most 1,000 scalar weights."""

import json
from pathlib import Path

import numpy as np


WIDTH = 10
ONE_REACHED, ROW, COL, NORM, TILE, MARK, DR, DC, DN, TEMP = range(WIDTH)
EPS = 2.0**-28
CENTER = 15
SCALE = 2.0**24
ALPHABET = ["", " ", "#", "S", "G", "\n", "o"]
VOCAB = " #SG\n"
TILE_VALUES = {" ": 0.0, "S": 0.125, "G": 0.25, "#": 1.0, "\n": 1.125, "": 1.25}


def build():
    fields = [{"kind": "char", "vocab": VOCAB}]
    a = {"e0": np.zeros((len(VOCAB) + 1, WIDTH))}
    for i, ch in enumerate([*VOCAB, ""]):
        a["e0"][i, ONE_REACHED] = 1 + EPS * (ch == "G")
        a["e0"][i, TILE] = TILE_VALUES[ch]
        a["e0"][i, MARK] = int(ch == "S")

    # A text token's line/column is its position before consuming that
    # character. Thus a newline occupies its own position at the line end.
    for name, feature, bits in (("line", ROW, 5), ("column", COL, 5), ("squared_norm", NORM, 9)):
        for bit in range(bits):
            field = {"kind": "coord_digit", "coordinate": name, "base": 2, "place": bit}
            if name == "squared_norm":
                field["origin"] = CENTER
            table = np.zeros((2, WIDTH))
            table[1, feature] = 2**bit
            a[f"e{len(fields)}"] = table
            fields.append(field)

    # The score scale is large enough for a unique winner to receive exactly
    # one under stable FP64 softmax. sqrt(4) compensates attention scaling.
    for i, outputs in enumerate((4, 1)):
        prefix = f"l{i}.h0."
        q = a[prefix + "q"] = np.zeros((WIDTH, 4))
        k = a[prefix + "k"] = np.zeros((WIDTH, 4))
        v = a[prefix + "v"] = np.zeros((WIDTH, outputs))
        o = a[prefix + "o"] = np.zeros((outputs, WIDTH))
        q[ROW, 0] = q[COL, 1] = 2
        q[ONE_REACHED, :3] = (-2 * CENTER, -2 * CENTER, 1)
        k[ROW, 0] = k[COL, 1] = 1
        k[NORM, 2] = -1
        if i == 0:
            # Spatial tie order, with utilities +1/8,+1/32,-1/32,-1/8.
            q[ONE_REACHED, :2] += (1 / 8, -1 / 32)
            q[[ONE_REACHED, TILE], 3] = (1.5, -1.5)
            k[ONE_REACHED, 3] = 1 / EPS
            # The residual accumulates an offset only when another cell wins.
            for j, (src, dst) in enumerate(((ROW, DR), (COL, DC), (NORM, DN),
                                            (ONE_REACHED, ONE_REACHED))):
                v[src, j] = o[j, dst] = 1
        else:
            k[DR, 0] = k[DC, 1] = 1
            k[DN, 2] = -1
            # Distinct sender positions break ties between identical parent
            # addresses. The entire perturbation is less than a quarter cell.
            k[[ROW, COL], 2] = (31 / 8192, 1 / 8192)
            q[ONE_REACHED, 3] = 0.5
            k[MARK, 3] = 1
            v[[ROW, DR, COL, DC, MARK], 0] = (62, 62, 2, 2, 1)
            o[0, TEMP] = 1
        q *= 2 * SCALE

    for i in range(2):
        prefix = f"l{i}."
        w1 = a[prefix + "w1"] = np.zeros((WIDTH, 4))
        w2 = a[prefix + "w2"] = np.zeros((4, WIDTH))
        b1 = a[prefix + "b1"] = np.zeros(4)
        b2 = a[prefix + "b2"] = np.zeros(WIDTH)
        if i == 0:
            for unit, (src, dst) in enumerate(((ROW, DR), (COL, DC), (NORM, DN))):
                w1[src, unit] = 1
                w2[unit, dst] = -1
            # The post-attention field is 2, 2+EPS, or 2+2*EPS. Restore the
            # leading constant and saturate the reached increment at EPS.
            w1[ONE_REACHED, 3] = 1
            b1[3] = -(2 + EPS)
            w2[3, ONE_REACHED] = b2[ONE_REACHED] = -1
        else:
            # On integers, relu(z)-2*relu(z-1)+relu(z-2) equals 1 iff z=1.
            # A cell already on the path is moved outside the equality window.
            w1[TEMP, :3] = 1
            w1[ROW, :3] = -62
            w1[COL, :3] = -2
            w1[MARK, :3] = 4096
            b1[:3] = (0, -1, -2)
            w2[:3, MARK] = (1, -2, 1)
            w1[TEMP, 3] = 1
            w2[3, TEMP] = -1

    out = a["readout"] = np.zeros((WIDTH, len(ALPHABET)))
    bias = a["bias"] = np.zeros(len(ALPHABET))
    for j, ch in enumerate(ALPHABET):
        center = TILE_VALUES[" " if ch == "o" else ch]
        out[TILE, j] = 2 * center
        bias[j] = -(center**2)
        if ch == "o":
            out[MARK, j] = 1 / 64
            bias[j] -= 1 / 128

    spec = {"fields": fields, "alphabet": ALPHABET, "layers": [1, 1], "rounds_per_token": 2}
    return spec, a


if __name__ == "__main__":
    spec, arrays = build()
    root = Path(__file__).resolve().parent
    (root / "network.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    np.savez(root / "weights.npz", **arrays)
    print(f"{sum(x.size for x in arrays.values())} weights; "
          f"{sum(np.count_nonzero(x) for x in arrays.values())} nonzero")
