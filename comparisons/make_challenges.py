"""Build paired, long-dependency checks without executing any of the candidate models."""
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "applications"))
from audit import reference
from experiment import SIZES, TASKS
import inventory
import scheduling
import telemetry


def make_pair(task, n, rng):
    pos = np.arange(n, dtype=np.int64)
    if task == "inventory":
        # The suffix nets to zero, so only the opening balance determines the last label.
        changes = rng.integers(-7, 8, size=n // 2)
        changes = np.stack([changes, -changes], axis=1).ravel() + 16
        left = np.column_stack((pos, changes, np.zeros_like(pos)))
        right = left.copy()
        amount = int(rng.integers(1, 17))
        left[0, 2], right[0, 2] = 25 - amount, 25 + amount
    elif task == "telemetry":
        left = np.column_stack((rng.integers(31, 70, size=n), pos))
        right = left.copy()
        left[0, 0], right[0, 0] = rng.integers(70, 101), rng.integers(0, 31)
    else:
        # All appointments start disjoint. Changing just the first booking makes
        # it overlap the final booking, a shuffled-index distance of n-1.
        origin = int(rng.integers(0, 1440 - 5*n))
        starts = (origin + 5*pos)[rng.permutation(n)]
        left = np.column_stack((starts, starts + rng.integers(1, 4, size=n), pos))
        right = left.copy()
        right[0, :2] = right[-1, :2]
    if rng.integers(2):
        return right, left
    return left, right


def main():
    bank = {}
    with np.load(HERE.parent / "applications/cases.npz", allow_pickle=False) as archived:
        for scenario, task in enumerate(TASKS):
            module = {"inventory": inventory, "scheduling": scheduling, "telemetry": telemetry}[task]
            for n in SIZES:
                rng = np.random.default_rng(np.random.SeedSequence([729031, scenario, n]))
                used = {r.tobytes() for r in archived[f"{task}_test_{n}_x"]}
                if n in (8, 16):
                    used.update(r.tobytes() for r in archived[f"{task}_dev_{n}_x"])
                records = []
                for _ in range(16):
                    for _ in range(1000):
                        pair = make_pair(task, n, rng)
                        enc = {row.tobytes() for row in pair}
                        if len(enc) == 2 and used.isdisjoint(enc):
                            break
                    else:
                        raise ValueError("Unable to generate distinct records")
                    used.update(enc)
                    records.extend(pair)
                x = np.stack(records).astype(np.int64)
                y = reference(task, x)
                if (not np.array_equal(x[::2, 1:], x[1::2, 1:]) or
                        not (y[::2, -1] != y[1::2, -1]).all() or
                        not np.array_equal(y, np.stack([module.oracle(row) for row in x]))):
                    raise ValueError(f"Invalid pair for {task}/{n}")
                bank[f"{task}_{n}_x"], bank[f"{task}_{n}_y"] = x, y
    np.savez_compressed(HERE / "challenge.npz", **bank)


if __name__ == "__main__":
    main()
