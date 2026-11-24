"""Download the two pinned Maze-Hard CSVs and verify their SHA-256 digests."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen


REVISION = "549754de3d67eebe57b6b22c7d226b9412d55b07"
BASE_URL = "https://huggingface.co/datasets/sapientinc/maze-30x30-hard-1k"
EXPECTED = {
    "train": "25f9e39b5aee16967da41ba63b2d1d75cf608c2360d8a1419d827f9859dbafa0",
    "test": "5120d52088fb3f8036e62e90103808a79ba58847d8601f9451f40991b0903a62",
}
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "data" / "maze"
CHUNK = 1024 * 1024


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")


def fetch(split, output_dir):
    target = output_dir / f"{split}.csv"
    expected = EXPECTED[split]
    if target.exists() or target.is_symlink():
        verify(target, expected)
        print(f"Verified existing {target}  {expected}")
        return

    request = Request(f"{BASE_URL}/resolve/{REVISION}/{target.name}?download=true",
                      headers={"User-Agent": "LoopedWSP-reproduction/1.0"})
    # The temporary file lives on the same volume as its final destination.
    with tempfile.TemporaryDirectory(prefix=".maze-download-", dir=output_dir) as temp:
        pending = Path(temp) / target.name
        with urlopen(request, timeout=60) as response, pending.open("xb") as file:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                file.write(chunk)
            file.flush()
            os.fsync(file.fileno())
        verify(pending, expected)
        try:
            if os.name == "nt":
                # Windows rename fails atomically if a concurrent writer won.
                os.rename(pending, target)
            else:
                # POSIX rename can replace a destination; link is atomic and exclusive.
                os.link(pending, target)
        except FileExistsError:
            verify(target, expected)
            print(f"Verified concurrently installed {target}  {expected}")
            return
    print(f"Downloaded and verified {target}  {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR,
                        help="Destination directory (default: data/maze in the project)")
    args = parser.parse_args()
    try:
        folder = args.output_dir.expanduser().resolve()
        folder.mkdir(parents=True, exist_ok=True)
        for split in (EXPECTED if args.split == "both" else (args.split,)):
            fetch(split, folder)
    except (OSError, URLError, ValueError) as exc:
        print(f"Maze-Hard download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
