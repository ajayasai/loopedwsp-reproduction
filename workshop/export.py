"""Export a predicted animation frame to PNG or a decoded music roll to WAV."""
import argparse
import json
from pathlib import Path
import struct
import sys
import wave
import zlib

import numpy as np


def save_frame(path, flat_class_ids, scale=32):
    """Enlarge the RGB4 frame with nearest-neighbor sampling and write a truecolor PNG."""
    values = np.asarray(flat_class_ids)
    if values.size != 192 or values.dtype.kind not in "iu" or values.min() < 0 or values.max() > 15:
        raise ValueError("Expected 8 × 8 × 3 four-bit channel values")
    if not isinstance(scale, int) or not 1 <= scale <= 64:
        raise ValueError("PNG scale must be an integer from 1 through 64")
    pixels = values.reshape(8, 8, 3).astype(np.uint8) * 17
    pixels = pixels.repeat(scale, axis=0).repeat(scale, axis=1)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    header = struct.pack(">IIBBBBB", pixels.shape[1], pixels.shape[0], 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                           + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def _write_notes(path, ids):
    # One beat is 0.25 s. Zero represents a rest or an unreadable frame.
    rate = 16000
    t = np.arange(rate // 4) / rate
    envelope = np.clip(t / .008, 0, 1) * np.clip((.225 - t) / .018, 0, 1)
    pitches = (60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79, 81, 83, 84)
    with wave.open(str(path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        for event in ids:
            sound = np.zeros_like(t)
            if event:
                frequency = 440 * 2 ** ((pitches[event - 1] - 69) / 12)
                sound = np.sin(2 * np.pi * frequency * t) * envelope
            out.writeframes(np.rint(sound * 12000).astype("<i2").tobytes())


def save_roll(path, flat_class_ids):
    values = np.asarray(flat_class_ids)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iu" or values.min() < 0 or values.max() > 144:
        raise ValueError("Expected one valid note/status class per frame")
    _write_notes(path, [0 if value == 144 else int(value % 16) for value in values])


def main():
    from experiment import MODELS, predict
    from roll import NOTES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("cels", "roll"))
    parser.add_argument("output", type=Path, help="Destination .png for cels or .wav for roll")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--example", action="store_true")
    src.add_argument("--input", type=Path, help="JSON input file, default stdin")
    args = parser.parse_args()
    expected = ".png" if args.task == "cels" else ".wav"
    if args.output.suffix.lower() != expected:
        parser.error(f"{args.task} requires a {expected} output path")
    if args.example:
        record = MODELS[args.task].example()
    elif args.input:
        record = json.loads(args.input.read_bytes())
    else:
        record = json.load(sys.stdin)
    answer = predict(args.task, record)
    if args.task == "cels":
        save_frame(args.output, answer["pixels"])
    else:
        events = answer["events"]
        _write_notes(args.output, [NOTES.index(event["note"]) if event["note"] is not None else 0 for event in events])
        print(f"{sum(e['status'] == 'corrected' for e in events)} corrected, "
              f"{sum(e['status'] == 'double_error' for e in events)} unreadable frames rendered as silence.")
    print(args.output)


if __name__ == "__main__":
    main()
