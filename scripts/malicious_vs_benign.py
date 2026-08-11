"""
Collapses each model's existing multiclass confusion matrix (already saved
in results/<dataset>/<key>.json) into a binary benign-vs-malicious view, for
every model that succeeded. No retraining -- this reuses the exact
predictions already produced by train_zoo.py, just re-grouped.

For each model: TP/FP/FN/TN (positive class = malicious), accuracy,
precision/recall/F1 on the malicious class, and false-positive /
false-negative rate on benign traffic -- the two numbers that actually
matter operationally for an IDS (falsely blocking legitimate traffic vs.
missing a real attack).

Output: results/feature_importance/<dataset>_malicious_vs_benign.json
"""
import glob
import json
import os

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_DIR = os.path.join(RESULTS_ROOT, "feature_importance")

BENIGN_LABEL = {"mqttset": "legitimate", "datasense": "benign"}


def collapse(cm, class_names, benign_name):
    benign_idx = class_names.index(benign_name)
    n = len(class_names)

    tn = cm[benign_idx][benign_idx]
    fp = sum(cm[benign_idx][j] for j in range(n) if j != benign_idx)  # benign predicted as attack
    fn = sum(cm[i][benign_idx] for i in range(n) if i != benign_idx)  # attack predicted as benign
    tp = sum(cm[i][j] for i in range(n) for j in range(n) if i != benign_idx and j != benign_idx)

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # = detection rate on real attacks
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0  # benign wrongly flagged as attack
    fnr = fn / (fn + tp) if (fn + tp) else 0.0  # attack wrongly passed as benign -- the dangerous one

    return {
        "tp_malicious_caught": tp, "tn_benign_passed": tn,
        "fp_benign_flagged_as_attack": fp, "fn_attack_missed_as_benign": fn,
        "accuracy": accuracy, "precision_malicious": precision, "recall_malicious": recall,
        "f1_malicious": f1, "false_positive_rate": fpr, "false_negative_rate": fnr,
    }


def run_dataset(dataset):
    benign_name = BENIGN_LABEL[dataset]
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, dataset, "*.json"))):
        r = json.load(open(path))
        if r.get("status") != "ok":
            continue
        binary = collapse(r["confusion_matrix"], r["class_names"], benign_name)
        rows.append({
            "key": r["key"], "name": r["name"], "family": r["family"],
            "f1_macro_multiclass": r["f1_macro"],
            "train_time_sec": r["train_time_sec"],
            **binary,
        })

    rows.sort(key=lambda r: r["f1_malicious"], reverse=True)

    out = {
        "dataset": dataset,
        "benign_class_name": benign_name,
        "n_models": len(rows),
        "models": rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{dataset}_malicious_vs_benign.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{dataset}: {len(rows)} models -> {out_path}")
    print(f"  best malicious-F1: {rows[0]['name']} ({rows[0]['f1_malicious']:.4f}), "
          f"FNR={rows[0]['false_negative_rate']:.4f}, FPR={rows[0]['false_positive_rate']:.4f}")
    worst = min(rows, key=lambda r: r["f1_malicious"])
    print(f"  worst malicious-F1: {worst['name']} ({worst['f1_malicious']:.4f})")
    return out


if __name__ == "__main__":
    for ds in ["mqttset", "datasense"]:
        run_dataset(ds)
