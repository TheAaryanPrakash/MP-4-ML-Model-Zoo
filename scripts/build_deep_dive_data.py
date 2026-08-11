"""
Aggregates results/feature_importance/*.json (feature selection + malicious-
vs-benign analysis, produced by feature_importance.py and
malicious_vs_benign.py) into frontend/deep-dive-data.json for the "Deep
Dive" dashboard tab. Kept separate from build_frontend_data.py's data.json
since this is a different, additive analysis rather than the core per-model
leaderboard data every tab already depends on.
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(ROOT, "results", "feature_importance")
FRONTEND_DIR = os.path.join(ROOT, "frontend")

DATASETS = ["mqttset", "datasense"]


def main():
    output = {"datasets": {}}
    for ds in DATASETS:
        fi_path = os.path.join(RESULTS_DIR, f"{ds}.json")
        mvb_path = os.path.join(RESULTS_DIR, f"{ds}_malicious_vs_benign.json")
        with open(fi_path) as f:
            fi = json.load(f)
        with open(mvb_path) as f:
            mvb = json.load(f)
        output["datasets"][ds] = {
            "feature_selection": fi,
            "malicious_vs_benign": mvb,
        }
        print(f"[{ds}] folded in {fi_path} + {mvb_path}")

    out_path = os.path.join(FRONTEND_DIR, "deep-dive-data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
