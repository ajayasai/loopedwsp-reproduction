"""Package, submit and collect three frozen application models on Kaggle."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "checks.json", "contours.py", "plumage.py", "clifford.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("frontier-results.json", "frontier-predictions.npz")
study = KaggleStudy(HERE, "frontier", "LoopedWSP Maps Plumage and Qubits",
                    f"{OWNER}/loopedwsp-maps-plumage-and-qubits", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
