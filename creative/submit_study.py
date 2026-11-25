"""Package, submit and collect the three creative applications on a private Kaggle GPU."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

STAGE = "creative"
TITLE = "LoopedWSP Creative Applications"
KERNEL = f"{OWNER}/loopedwsp-creative-applications"
INPUTS = ("experiment.py", "folds.py", "typeset.py", "ballots.py", "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ["creative-results.json", "creative-predictions.npz"]

study = KaggleStudy(HERE, STAGE, TITLE, KERNEL, INPUTS, tuple(OUTPUTS))
main = study.main

if __name__ == "__main__":
    main()
