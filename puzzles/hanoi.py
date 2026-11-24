"""Analytic tied causal transformer for the optimal three-peg Hanoi sequence.

The prompt specifies N, start peg 0, and target peg 2. Each prefix position p
predicts the move at t=p+1. Only the role, the prompt size and generic digits
of the position enter the embedding; past move tokens share one embedding.
"""

import numpy as np


MAX_N = 20
CONST, PROMPT, SIZE_SOURCE, PARITY_SOURCE, SIZE, N_ODD = range(6)
BITS = tuple(range(6, 27))                # binary digits of t, least significant first
MOD3 = tuple(range(27, 30))               # t modulo 3, one-hot
READY, DISK, DISK_SQUARED, DISK_ODD = range(30, 34)
PAIRS = tuple((src, dst) for src in range(3) for dst in range(3) if src != dst)
MOVE = {pair: 34 + i for i, pair in enumerate(PAIRS)}
EOS = 40
WIDTH = 41


def build():
    """Construct every scalar from the cyclic-disk recurrence, without training."""
    fields = [{"kind": "role"}, {"kind": "size"}]
    role = np.zeros((2, WIDTH))
    role[:, CONST] = 1
    role[0, PROMPT] = 1
    size = np.zeros((MAX_N + 1, WIDTH))
    n = np.arange(MAX_N + 1)
    size[:, SIZE_SOURCE] = n
    size[:, PARITY_SOURCE] = n % 2
    arrays = {"e0": role, "e1": size}
    for bit, feature in enumerate(BITS):
        fields.append({"kind": "position_digit", "offset": 1, "divisor": 2**bit, "base": 2})
        embed = np.zeros((2, WIDTH))
        embed[1, feature] = 1
        arrays[f"e{len(fields) - 1}"] = embed
    fields.append({"kind": "position_digit", "offset": 1, "divisor": 1, "base": 3})
    mod = np.zeros((3, WIDTH))
    for remainder in range(3):
        mod[remainder, MOD3[remainder]] = 1
    arrays[f"e{len(fields) - 1}"] = mod

    # Every causal prefix includes the unique prompt. Its stable softmax score
    # margin 2048 makes the prompt's probability exactly one in float64.
    q, k = np.zeros((WIDTH, 1)), np.zeros((WIDTH, 1))
    q[CONST, 0] = 2048
    k[PROMPT, 0] = 1
    v, o = np.zeros((WIDTH, 2)), np.zeros((2, WIDTH))
    for lane, (source, copied) in enumerate(((SIZE_SOURCE, SIZE), (PARITY_SOURCE, N_ODD))):
        # After the first block, the two prompt fields cancel and copying stops.
        v[source, lane], v[copied, lane], o[lane, copied] = 1, -1, 1
    arrays.update({"l0.h0.q": q, "l0.h0.k": k, "l0.h0.v": v, "l0.h0.o": o})

    ff = len(BITS) + 12 + 2
    w1, b1 = np.zeros((WIDTH, ff)), np.zeros(ff)
    w2, b2 = np.zeros((ff, WIDTH)), np.zeros(WIDTH)
    # On the first block exactly the least significant set bit fires. That
    # disk number is one plus the number of trailing zeroes in the move index.
    for j, bit_field in enumerate(BITS):
        disk = j + 1
        w1[bit_field, j] = 1
        w1[list(BITS[:j]), j] = -1
        w1[READY, j] = -1
        w2[j, [READY, DISK, DISK_SQUARED, DISK_ODD]] = (1, disk, disk**2, disk % 2)

    # Each disk moves in one cyclic direction. Its kth move (zero-based k)
    # occurs at t=(2k+1)*2**(disk-1). Reduction mod 3 gives the source and
    # destination below using only t mod 3, the size parity and disk parity.
    u = len(BITS)
    for n_odd in range(2):
        for d_odd in range(2):
            direction = -1 if n_odd == d_odd else 1
            for remainder in range(3):
                base = remainder if n_odd else -remainder
                pair = ((base + direction) % 3, (base - direction) % 3)
                flag = MOVE[pair]
                w1[[READY, MOD3[remainder]], u] = 1
                w1[N_ODD, u] = 1 if n_odd else -1
                w1[DISK_ODD, u] = 1 if d_odd else -1
                w1[flag, u] = -1
                # Four Boolean literals (including READY and the residue).
                b1[u] = (1 - n_odd) + (1 - d_odd) - 3
                w2[u, flag] = 1
                u += 1

    # On integer inputs the two rectifiers implement 1[disk>N]. For the
    # first such position t=2**N. Latching EOS keeps all later blocks fixed.
    w1[DISK, u:] = 1
    w1[SIZE, u:] = -1
    w1[EOS, u:] = -(MAX_N + 2)
    b1[u + 1] = -1
    w2[u:, EOS] = (1, -1)
    arrays.update({"l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": b2})

    alphabet = [None] + [[disk, src, dst] for disk in range(1, MAX_N + 1) for src, dst in PAIRS]
    readout, bias = np.zeros((WIDTH, len(alphabet))), np.zeros(len(alphabet))
    readout[EOS, 0], bias[0] = 2, -1
    for j, (disk, src, dst) in enumerate(alphabet[1:], 1):
        # The move receives 0.5 iff both the disk and the directed pair match.
        readout[[DISK, DISK_SQUARED, MOVE[(src, dst)]], j] = (2 * disk, -1, 1)
        bias[j] = -(disk**2) - 0.5
    arrays.update({"readout": readout, "bias": bias})
    return {"fields": fields, "layers": [1], "rounds": 2, "alphabet": alphabet}, arrays
