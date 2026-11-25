"""Package the frozen graph-depth study or submit and collect its private Kaggle run."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from scripts.kaggle_study import KaggleStudy
from submit import OWNER

STAGE = "graph-depth"
TITLE = "LoopedWSP Pointer Reachability"
KERNEL = f"{OWNER}/loopedwsp-pointer-reachability"
INPUTS = ("experiment.py", "pointer.py", "reachability.py", "corpus.py", "protocol.json", "cases-manifest.json")
OUTPUTS = ["graph-depth-results.json", "graph-depth-predictions.npz"]

study = KaggleStudy(HERE, STAGE, TITLE, KERNEL, INPUTS, tuple(OUTPUTS))
build, push, pull_metadata, collect = study.build, study.push, study.pull_metadata, study.collect
main = study.main

if __name__ == "__main__":
    main()
