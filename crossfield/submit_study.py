"""Authenticate, submit and collect the private cross-field study."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "checks.json", "inflection.py", "diffraction.py", "auction.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("crossfield-results.json", "crossfield-predictions.npz")
study = KaggleStudy(HERE, "crossfield", "LoopedWSP Crossfield Applications",
                    f"{OWNER}/loopedwsp-crossfield-applications", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
