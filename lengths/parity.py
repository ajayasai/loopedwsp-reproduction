"""Parity of a bitstring by a parallel prefix scan in a tied transformer block."""

import numpy as np


SIZES = (8, 20, 21, 32, 64, 128, 255, 511, 1023, 2047, 4095)
DEV_SIZES = (8, 20)
COUNT = 64
TABLE_SIZE = 4096
CONST, POSITION, SQUARE, STRIDE, BIT = range(5)


def build():
    """Return the analytic FP64 embeddings, shared block, and last-token readout."""
    width = 5
    chars = np.zeros((3, width))  # BOS, '0', '1'
    chars[:, [CONST, STRIDE]] = 1
    chars[2, BIT] = 1

    ids = np.arange(TABLE_SIZE, dtype=np.float64)
    positions = np.zeros((TABLE_SIZE, width))
    positions[:, POSITION] = ids
    positions[:, SQUARE] = ids**2

    # The score for key j at query i is -1024*(j-(i-stride))**2 up to a
    # row-wise constant. A target below zero selects the zero-bit BOS token.
    q = np.zeros((width, 2))
    k = np.zeros((width, 2))
    q[[POSITION, STRIDE], 0] = (2, -2)
    q[CONST, 1] = -1
    q *= 1024 * np.sqrt(2)
    k[POSITION, 0] = k[SQUARE, 1] = 1
    v, o = np.zeros((width, 1)), np.zeros((1, width))
    v[BIT, 0] = o[0, BIT] = 1

    # After attention, BIT is the sum of two bits. Subtracting twice the
    # overflow at one is XOR. The other two units double and cap the stride.
    w1, w2 = np.zeros((width, 3)), np.zeros((3, width))
    w1[BIT, 0] = 1
    w2[0, BIT] = -2
    w1[STRIDE, 1:] = 1
    w2[1:, STRIDE] = (1, -2)
    b1 = np.array([-1.0, 0.0, -TABLE_SIZE / 2])

    readout = np.zeros((width, 2))
    readout[BIT, 1] = 1
    arrays = {
        "e0": chars, "e1": positions,
        "l0.h0.q": q, "l0.h0.k": k, "l0.h0.v": v, "l0.h0.o": o,
        "l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": np.zeros(width),
        "readout": readout, "bias": np.array([0.5, 0.0]),
    }
    spec = {"layers": [1], "fields": ["symbol", "position"], "round_rule": "log2_plus_one"}
    return spec, arrays


def sample(size, index, rng):
    """Draw a uniform bitstring conditional on the example's desired parity."""
    if not 1 <= size < TABLE_SIZE:
        raise ValueError(f"parity expects 1 to {TABLE_SIZE - 1} bits")
    bits = rng.integers(0, 2, size=size, dtype=np.uint8)
    # Alternating desired classes gives exactly balanced bins for even COUNT.
    bits[-1] = (int(bits[:-1].sum()) + index) & 1
    # Grade the generated string by integer popcount, separately from the
    # generator's arithmetic and the model's pointer-doubling computation.
    truth = int.from_bytes(np.packbits(bits).tobytes(), "big").bit_count() % 2
    tokens = np.empty((size + 1, 2), dtype=np.int64)
    tokens[0, 0] = 0
    tokens[1:, 0] = bits + 1
    tokens[:, 1] = np.arange(size + 1)
    return tokens, truth, {"bits": size}
