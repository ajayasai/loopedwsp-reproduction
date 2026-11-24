"""A looped transformer that emits an alarm state after every sensor reading."""

import numpy as np


NAME = "Hysteresis alarm"
UNIT = "readings"
SIZES = (8, 16, 32, 64, 128)
DEV_SIZES = (8, 16)
COUNT = 64
LOW, HIGH = 30, 70


def oracle(tokens):
    """Replay the alarm specification in chronological order, starting off."""
    state = 0
    labels = []
    for reading, _ in tokens:
        if reading >= HIGH:
            state = 1
        elif reading <= LOW:
            state = 0
        labels.append(state)
    return np.asarray(labels, dtype=np.int64)


def sample(size, index, rng):
    """Generate raw readings and grade by the most recent trip/reset timestamp."""
    if not 8 <= size <= max(SIZES):
        raise ValueError("Telemetry cases support 8 through 128 readings")
    # Most cases contain extended stretches in the hold band. Eight case families
    # exercise both alarm levels, endpoint transitions, and the exact thresholds.
    x = rng.integers(LOW + 1, HIGH, size=size, dtype=np.int64)
    case = index % 8
    if case == 0:
        x[0] = HIGH
        x[rng.integers(size//3, 2*size//3 + 1)] = LOW
    elif case == 1:
        x[-1] = HIGH
    elif case == 2:
        x[0], x[-1] = HIGH, LOW
    elif case == 3:
        # A short pulse, recovery, a second pulse, and a later reset.
        i = int(rng.integers(0, size//3))
        j = int(rng.integers(size//2, size-2))
        x[i], x[i+1], x[j], x[-1] = HIGH, LOW, HIGH, LOW
    elif case == 4:
        x[0], x[-1] = LOW+1, HIGH-1
    elif case == 5:
        x[0], x[1], x[-1] = HIGH+1, LOW+1, HIGH-1
    elif case == 6:
        x = rng.integers(0, 101, size=size, dtype=np.int64)
        # Guarantee both range endpoints and each threshold's immediate neighbors.
        x[rng.choice(size, size=8, replace=False)] = [0,29,30,31,69,70,71,100]
    else:
        # Short bursts alternate with random holds; event values cover both tails.
        indices = np.sort(rng.choice(size, size=max(2, size//4), replace=False))
        x[indices[::2]] = rng.integers(HIGH, 101, size=len(indices[::2]))
        x[indices[1::2]] = rng.integers(0, LOW+1, size=len(indices[1::2]))
    positions = np.arange(size, dtype=np.int64)
    last_trip = np.maximum.accumulate(np.where(x >= HIGH, positions, -1))
    last_reset = np.maximum.accumulate(np.where(x <= LOW, positions, -1))
    labels = (last_trip > last_reset).astype(np.int64)
    return np.column_stack((x, positions)), labels


def build():
    pos, squared_pos, constant, reading, alarm, previous = range(6)
    width = 6
    e0 = np.zeros((101, width))
    e0[:, reading] = np.arange(101)
    e1 = np.zeros((max(SIZES), width))
    indices = np.arange(max(SIZES))
    e1[:, pos], e1[:, squared_pos], e1[:, constant] = indices, indices**2, 1

    # Nearest integer address to i-1, clamped to zero at the left edge.
    # The nearest alternative differs by at least 2048/sqrt(2) in logit,
    # so the standard FP64 softmax assigns it zero probability.
    q = np.zeros((width, 2))
    k = np.zeros_like(q)
    q[pos, 0], q[constant, 0], q[constant, 1] = 4096, -4096, -2048
    k[pos, 0] = k[squared_pos, 1] = 1
    v, o = np.zeros((width, 1)), np.zeros((1, width))
    v[alarm, 0] = o[0, previous] = 1

    # For integral x and bit p, ReLU(x-a+p)-ReLU(x-a) copies p exactly
    # when x >= a. The gates set at HIGH and retain p on LOW < x < HIGH.
    units = [
        ({reading: 1}, 1-HIGH, alarm, 1),
        ({reading: 1, previous: 1}, -(LOW+1), alarm, 1),
        ({reading: 1}, -(LOW+1), alarm, -1),
        ({reading: 1, previous: 1}, -HIGH, alarm, -1),
        ({alarm: 1}, 0, alarm, -1),
        ({previous: 1}, 0, previous, -1),
    ]
    w1, w2 = np.zeros((width, len(units))), np.zeros((len(units), width))
    b1 = np.zeros(len(units))
    for j, (inputs, bias, output, gain) in enumerate(units):
        for axis, coefficient in inputs.items():
            w1[axis,j] = coefficient
        b1[j], w2[j,output] = bias, gain
    readout = np.zeros((width, 2))
    readout[alarm, 1] = 2
    spec = {"fields": ["reading", "position"], "layers": [1], "round_rule":"tokens"}
    arrays = {"e0": e0, "e1": e1, "l0.h0.q": q, "l0.h0.k": k,
              "l0.h0.v": v, "l0.h0.o": o, "l0.w1": w1, "l0.b1": b1,
              "l0.w2": w2, "l0.b2": np.zeros(width),
              "readout": readout, "bias": np.array([1., 0.])}
    return spec, arrays
