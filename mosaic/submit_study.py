"""Package, run and verify the location, Braille and bridge models on Kaggle."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "checks.json", "quake.py", "braille.py", "bridge.py",
          "cases.json.gz", "context_pairs.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("mosaic-results.json", "mosaic-predictions.npz")
study = KaggleStudy(HERE, "mosaic", "LoopedWSP Quakes Braille and Bridge",
                    f"{OWNER}/loopedwsp-quakes-braille-and-bridge", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
