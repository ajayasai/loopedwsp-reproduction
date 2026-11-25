"""Package, launch or collect the workshop application suite on Kaggle."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "mobile.py", "cels.py", "roll.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("workshop-results.json", "workshop-predictions.npz")
study = KaggleStudy(HERE, "workshop", "LoopedWSP Workshop Applications",
                    f"{OWNER}/loopedwsp-workshop-applications", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
