"""Apply a frozen application model to a JSON record. Read from a file or stdin."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lengths"))
from frame import Transformer, inspect


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")
    return value


def encode(task, data):
    field = {"inventory": "changes", "scheduling": "bookings", "telemetry": "readings"}[task]
    if not isinstance(data, dict) or not isinstance(data.get(field), list):
        raise ValueError(f"Expected a JSON object with a {field} array")
    rows = data[field]
    if not 1 <= len(rows) <= 128:
        raise ValueError(f"Expected 1 through 128 items in {field}")
    index = np.arange(len(rows), dtype=np.int64)
    if task == "inventory":
        opening = integer(data.get("opening"), -24, 24, "opening")
        changes = [integer(value, -16, 16, "quantity change") + 16 for value in rows]
        initial = np.zeros_like(index)
        initial[0] = opening + 25
        return np.column_stack((index, changes, initial))
    if task == "telemetry":
        readings = [integer(value, 0, 100, "reading") for value in rows]
        return np.column_stack((readings, index))
    spans = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("Each booking must be a [start_minute, end_minute] pair")
        start = integer(row[0], 0, 1439, "start_minute")
        end = integer(row[1], start + 1, 1440, "end_minute")
        spans.append((start, end))
    return np.column_stack((spans, index))


def predict(task, record, gpu=False):
    tokens = encode(task, record)
    folder = HERE / "models" / task
    spec = json.loads((folder / "network.json").read_bytes())
    frozen = json.loads((folder / "freeze.json").read_bytes())
    with np.load(folder / "weights.npz", allow_pickle=False) as archive:
        weights = {name: archive[name] for name in archive.files}
    if inspect(spec, weights)["model_sha256"] != frozen["model_sha256"]:
        raise ValueError("Frozen model fingerprint mismatch")
    model = Transformer(spec, weights, gpu=gpu)
    values, _ = model.evaluate_batch([tokens], multiples=(1,), per_token=True)
    label = {"inventory": "negative_balance", "scheduling": "conflict", "telemetry": "alarm_on"}[task]
    return {label: values[1][0].astype(bool).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("inventory", "scheduling", "telemetry"))
    parser.add_argument("input_file", type=Path, nargs="?", help="JSON file; omit to read a record from stdin")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA for inference")
    args = parser.parse_args()
    try:
        record = json.loads(args.input_file.read_bytes()) if args.input_file else json.load(sys.stdin)
        print(json.dumps(predict(args.task, record, args.gpu)))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
