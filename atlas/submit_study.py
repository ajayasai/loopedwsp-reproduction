"""Package, submit and collect the three frozen application models on Kaggle."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "checks.json", "rings.py", "loom.py", "braid.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("atlas-results.json", "atlas-predictions.npz")
study = KaggleStudy(HERE, "atlas", "LoopedWSP Rings Loom and Braids",
                    f"{OWNER}/loopedwsp-rings-loom-and-braids", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
