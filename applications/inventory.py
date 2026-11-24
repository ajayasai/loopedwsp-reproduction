"""Flag every negative balance in a stream of signed inventory transactions."""

import numpy as np


NAME = "Inventory shortfall detection"
UNIT = "events"
DEV_SIZES = (8, 16)
SIZES = (8, 16, 32, 64, 128)
COUNT = 64
MAX_EVENTS = 128
MAX_DELTA = 16
MAX_OPENING = 24
ONE, POSITION, SQUARE, STRIDE, BALANCE, INCOMING = range(6)


def build():
    """Build the shared parallel-prefix block. All entries are analytic FP64."""
    width = 6
    position_ids = np.arange(MAX_EVENTS, dtype=np.float64)
    position = np.zeros((MAX_EVENTS, width))
    position[:, [ONE, STRIDE]] = 1
    position[:, POSITION] = position_ids
    position[:, SQUARE] = position_ids**2
    delta = np.zeros((2 * MAX_DELTA + 1, width))
    delta[:, BALANCE] = np.arange(-MAX_DELTA, MAX_DELTA + 1)
    opening = np.zeros((2 * MAX_OPENING + 2, width))
    # ID zero means no new opening entry. Exactly the first event carries one.
    opening[1:, BALANCE] = np.arange(-MAX_OPENING, MAX_OPENING + 1)

    # For query i, the score for key j equals -1024*(j-(i-stride))**2
    # plus a row constant. Outside the prefix the nearest legal key is zero.
    q, k = np.zeros((width, 2)), np.zeros((width, 2))
    q[[POSITION, STRIDE], 0] = (2, -2)
    q[ONE, 1] = -1
    q *= 1024 * np.sqrt(2)
    k[POSITION, 0] = k[SQUARE, 1] = 1
    v, o = np.zeros((width, 1)), np.zeros((1, width))
    v[BALANCE, 0] = o[0, INCOMING] = 1

    w1, w2 = np.zeros((width, 6)), np.zeros((6, width))
    b1 = np.zeros(6)
    # For a signed incoming amount x, ReLU(bound+x)-ReLU(bound) equals x
    # when a left segment exists and zero otherwise. Every segment sum has
    # magnitude at most MAX_OPENING + MAX_EVENTS*MAX_DELTA. Using twice this
    # bound gives both gate branches a full integer unit of clearance.
    margin = 2 * (MAX_OPENING + MAX_EVENTS * MAX_DELTA + 1)
    w1[POSITION, :2] = margin
    w1[STRIDE, :2] = -margin
    b1[:2] = margin / 2
    w1[INCOMING, 0] = 1
    w2[:2, BALANCE] = (1, -1)
    # Reset the signed scratch value, then double the stride up to MAX_EVENTS.
    w1[INCOMING, 2:4] = (1, -1)
    w2[2:4, INCOMING] = (-1, 1)
    w1[STRIDE, 4:6] = 1
    b1[5] = -MAX_EVENTS / 2
    w2[4:6, STRIDE] = (1, -2)

    readout = np.zeros((width, 2))
    readout[BALANCE, 1] = -1
    arrays = {"e0": position, "e1": delta, "e2": opening,
              "l0.h0.q": q, "l0.h0.k": k, "l0.h0.v": v, "l0.h0.o": o,
              "l0.w1": w1, "l0.b1": b1, "l0.w2": w2, "l0.b2": np.zeros(width),
              "readout": readout, "bias": np.array([0.0, -0.5])}
    spec = {"fields": ["event_position", "signed_quantity", "opening_balance"],
            "layers": [1], "round_rule": "log2_plus_one"}
    return spec, arrays


def oracle(tokens):
    """Evaluate a sequential ledger and flag each post-event balance below zero."""
    if (tokens.ndim != 2 or tokens.shape[1] != 3 or len(tokens) == 0 or
            not 1 <= tokens[0, 2] <= 2 * MAX_OPENING + 1 or np.any(tokens[1:, 2] != 0)):
        raise ValueError("Expected an opening quantity on the first event only")
    balance = int(tokens[0, 2]) - (MAX_OPENING + 1)
    labels = np.empty(len(tokens), dtype=np.int64)
    for i, event in enumerate(tokens):
        balance += int(event[1]) - MAX_DELTA
        labels[i] = int(balance < 0)
    return labels


def sample(size, index, rng):
    """Draw an event stream with a mixture of random walks and known boundary cases."""
    if not 1 <= size <= MAX_EVENTS:
        raise ValueError(f"Expected 1 to {MAX_EVENTS} events")
    delta = rng.integers(-MAX_DELTA, MAX_DELTA + 1, size=size, dtype=np.int64)
    opening = int(rng.integers(-MAX_OPENING, MAX_OPENING + 1))
    if size >= 4 and index % 4 in (0, 1):
        # Include depletion to zero, a shortfall and a return to zero. Vary
        # quantities and later transactions separately for every sampled case.
        opening = int(rng.integers(1, MAX_DELTA + 1)) * (1 if index % 4 == 0 else -1)
        shortfall = int(rng.integers(1, MAX_DELTA + 1))
        surplus = int(rng.integers(1, MAX_DELTA + 1))
        delta[:4] = (-opening, -shortfall, shortfall, surplus)
    tokens = np.zeros((size, 3), dtype=np.int64)
    tokens[:, 0] = np.arange(size)
    tokens[:, 1] = delta + MAX_DELTA
    tokens[0, 2] = opening + MAX_OPENING + 1
    return tokens, oracle(tokens)
