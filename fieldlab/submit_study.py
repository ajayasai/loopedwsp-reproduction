"""Submit and collect the private Kaggle GPU field study."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "basin.py", "transit.py", "bowling.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("fieldlab-results.json", "fieldlab-predictions.npz")
study = KaggleStudy(HERE, "fieldlab", "LoopedWSP Field Experiments",
                    f"{OWNER}/loopedwsp-field-experiments", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
