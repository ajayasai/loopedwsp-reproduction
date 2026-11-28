"""Package, submit, and collect the isotope, Calendar Round and robot-arm experiments."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

INPUTS = ("experiment.py", "protocol.json", "checks.json", "isotopes.py", "calendar_round.py", "arm.py",
          "cases.json.gz", "weights.npz", "freeze.json")
OUTPUTS = ("prism-results.json", "prism-predictions.npz")
study = KaggleStudy(HERE, "prism", "LoopedWSP Isotopes Calendars and Arms",
                    f"{OWNER}/loopedwsp-isotopes-calendars-and-arms", INPUTS, OUTPUTS)

if __name__ == "__main__":
    study.main()
