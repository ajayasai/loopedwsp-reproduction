"""Analytic Checker Jumping transformer for symmetric boards of up to 40 per side.

The motion phases have lengths 1..N, N, N..1 and alternate piece colors.
Their forward and reflected halves share the same triangular-number circuit.
Each output token names an absolute [source, destination] pair.
"""

import numpy as np


MAX_SIZE = 40
DEV_SIZES = (1, 2, 3)
SIZES = tuple(range(1, MAX_SIZE + 1))


def build():
    width = 18
    (one, prompt, input_n, input_moves, ordinal, n, moves, short_time,
     mirror, ended, blue, slide, phase_offset, gap, gap_step, code, next_n, next_moves) = range(width)
    fields = [{"kind": "role"}, {"kind": "size"}]
    divisors = (1, 8, 64, 512)
    fields.extend({"kind": "position_digit", "base": 8, "divisor": d} for d in divisors)
    arrays = {}

    def matrix(key, *shape):
        arrays[key] = np.zeros(shape, dtype=np.float64)
        return arrays[key]

    def mlp(layer, hidden):
        prefix = f"l{layer}."
        return (matrix(prefix+"w1", width, hidden), matrix(prefix+"b1", hidden),
                matrix(prefix+"w2", hidden, width), matrix(prefix+"b2", width))

    role = matrix("e0", 2, width)
    role[:, one], role[0, prompt] = 1, 1
    sizes = matrix("e1", MAX_SIZE+1, width)
    ns = np.arange(MAX_SIZE+1, dtype=np.float64)
    sizes[:, input_n], sizes[:, input_moves] = ns, ns*(ns+2)
    for i, divisor in enumerate(divisors):
        pos = matrix(f"e{i+2}", 8, width)
        pos[:, ordinal] = divisor*np.arange(8)

    limit = 16384  # Larger than any decoded ordinal or gap in this position range.
    # Fold the move index around the middle of the symmetric schedule:
    # k = t - ReLU(2t-M+1), where M is the total number of moves.
    w, b, o, _ = mlp(0, 9)
    w[ordinal, 0], o[0, short_time] = 1, 1
    for i in (1, 2):
        w[ordinal, i], w[moves, i], b[i] = 2, -1, 2-i
    o[1, short_time], o[1, mirror], o[2, mirror] = -1, 1, -1
    for i in (3, 4):
        w[ordinal, i], w[moves, i], b[i] = 1, -1, 4-i
    o[3, ended], o[4, ended] = 1, -1
    w[short_time, 5:7], o[5:7, short_time] = (1, -1), (-1, 1)
    w[mirror, 7], o[7, mirror] = 1, -1
    w[ended, 8], o[8, ended] = 1, -1

    # The p-th phase starts at T_p=p(p+1)/2. Differences of adjacent ReLUs
    # give exact integer steps or an equality indicator at T_p-1 (a slide).
    w, b, o, _ = mlp(1, 3*MAX_SIZE+4)
    for p in range(1, MAX_SIZE+1):
        boundary = p*(p+1)//2
        row = 3*(p-1)
        w[short_time, row:row+3] = 1
        b[row:row+3] = (2-boundary, 1-boundary, -boundary)
        o[row:row+3, slide] = (1, -2, 1)
        # These step increments telescope to p mod 2 and (-1)^p*p*(p+2).
        blue_change = 1 if p % 2 else -1
        offset_now = (-1)**p * p*(p+2)
        offset_before = (-1)**(p-1) * (p-1)*(p+1)
        offset_change = offset_now-offset_before
        o[row+1:row+3, blue] = (blue_change, -blue_change)
        o[row+1:row+3, phase_offset] = (offset_change, -offset_change)
    base = 3*MAX_SIZE
    w[slide, base], o[base, slide] = 1, -1
    w[blue, base+1], o[base+1, blue] = 1, -1
    w[phase_offset, base+2:base+4], o[base+2:base+4, phase_offset] = (1, -1), (-1, 1)

    # Forward target gap is N + (-1)^p * (p*(p+2)-2*k).
    # The signed change of gap position is -2/+2 for a red/blue jump;
    # a slide has the same sign and half the magnitude.
    w, b, o, b2 = mlp(2, 10)
    for unit, sign in ((0, 1), (1, -1)):
        w[[n, phase_offset, short_time], unit] = (sign, sign, -2*sign)
        o[unit, gap] = sign
    w[short_time, 2], w[blue, 2], b[2], o[2, gap] = 1, limit, -limit, 4
    w[blue, 3], o[3, gap_step] = 1, 4
    w[slide, 4], o[4, gap_step] = 1, 1
    w[[blue, slide], 5], b[5], o[5, gap_step] = 1, -1, -2
    b2[gap_step] = -2
    w[gap, 6:8], o[6:8, gap] = (1, -1), (-1, 1)
    w[gap_step, 8:10], o[8:10, gap_step] = (1, -1), (-1, 1)

    # The reflected half reverses move order and board coordinates together.
    # Its destination is 2*N - (forward gap + forward gap step).
    stride = 2*MAX_SIZE+1
    w, b, o, b2 = mlp(3, 5)
    w[gap, 0], w[mirror, 0], o[0, code] = 1, -limit, 1
    w[[n, gap, gap_step, mirror], 1] = (2, -1, -1, limit)
    b[1], o[1, code] = -limit, 1
    w[blue, 2], o[2, code] = 1, 2*stride
    w[slide, 3], o[3, code] = 1, -stride
    w[code, 4], o[4, code] = 1, -1
    b2[code] = stride+1

    # Causal attention selects the original prompt. This layer is last so
    # the arithmetic first sees its prompt registers on the second loop.
    q, k = matrix("l4.h0.q", width, 1), matrix("l4.h0.k", width, 1)
    v, out = matrix("l4.h0.v", width, 2), matrix("l4.h0.o", 2, width)
    q[one, 0], k[prompt, 0] = 1024, 1
    v[input_n, 0], v[input_moves, 1] = 1, 1
    out[0, next_n], out[1, next_moves] = 1, 1
    w, b, o, _ = mlp(4, 4)
    for i, (buffer, register) in enumerate(((next_n, n), (next_moves, moves))):
        w[buffer, 2*i], o[2*i, [buffer, register]] = 1, (-1, 1)
        w[register, 2*i+1], o[2*i+1, register] = 1, -1

    # A token itself identifies both endpoints. Four families of coordinate
    # pairs share a nearest-integer classifier on the network-computed code.
    alphabet, centers = [None], [0]
    for mode, source_delta in enumerate((-1, -2, 1, 2)):
        for target in range(stride):
            source = target + source_delta
            if 0 <= source < stride:
                alphabet.append([source, target])
                centers.append(1 + stride*mode + target)
    readout, bias = matrix("readout", width, len(alphabet)), matrix("bias", len(alphabet))
    centers = np.asarray(centers, dtype=np.float64)
    readout[code], bias[:] = 2*centers, -centers**2
    readout[ended, 0], bias[0] = 2**30, -1
    spec = {"fields": fields, "layers": [0, 0, 0, 0, 1], "rounds": 2, "alphabet": alphabet}
    return spec, arrays
