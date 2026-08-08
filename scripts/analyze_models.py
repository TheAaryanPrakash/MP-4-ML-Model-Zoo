"""
Multi-criteria decision-support analysis over every trained model in the zoo.

Scores each model on a composite of classification quality (F1-macro,
F1-weighted, and F1 on the rarest attack class -- weighted extra since
missing a rare attack is the failure mode that matters most for an IDS) and
deployment cost (inference latency, weighted higher than train time since
inference runs continuously in production while training is periodic).
Both halves are min-max normalized per dataset, then averaged across
datasets to also produce a cross-dataset consistency ("generalization")
score. Powers frontend/model-selection.html.

Usage:
    python analyze_models.py
"""
import glob
import json
import os

ROOT = "/Users/aaryan/Documents/University/Year 4/Semester 7/Major Project/v3-ml-model-zoo"
RESULTS_DIR = f"{ROOT}/results"
OUT_PATH = f"{ROOT}/frontend/model-selection-data.json"

QUALITY_WEIGHT = 0.70
COST_WEIGHT = 0.30
W_F1_MACRO = 0.55
W_F1_WEIGHTED = 0.20
W_RARE_CLASS = 0.25
W_INFER = 0.65
W_TRAIN = 0.35


def load(dataset):
    models = {}
    for path in glob.glob(f"{RESULTS_DIR}/{dataset}/*.json"):
        with open(path) as f:
            d = json.load(f)
        if d.get("status") == "ok":
            models[d["key"]] = d
    return models


def minmax_norm(values, invert=False):
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0

    def f(v):
        n = (v - lo) / span
        return (1 - n) if invert else n

    return f


def main():
    datasets = {"mqttset": load("mqttset"), "datasense": load("datasense")}

    rarest = {}
    for ds, models in datasets.items():
        any_model = next(iter(models.values()))
        pc = sorted(any_model["per_class"], key=lambda x: x["support"])
        rarest[ds] = pc[0]["class"]

    per_dataset_scores = {}
    for ds, models in datasets.items():
        keys = list(models.keys())
        rare_cls = rarest[ds]
        rare_f1_vals = {k: next(pc["f1"] for pc in models[k]["per_class"] if pc["class"] == rare_cls) for k in keys}

        n_f1m = minmax_norm([models[k]["f1_macro"] for k in keys])
        n_f1w = minmax_norm([models[k]["f1_weighted"] for k in keys])
        n_rare = minmax_norm(list(rare_f1_vals.values()))
        n_train = minmax_norm([models[k]["train_time_sec"] for k in keys], invert=True)
        n_infer = minmax_norm([models[k]["inference_time_per_sample_ms"] for k in keys], invert=True)

        scores = {}
        for k in keys:
            m = models[k]
            quality = W_F1_MACRO * n_f1m(m["f1_macro"]) + W_F1_WEIGHTED * n_f1w(m["f1_weighted"]) + W_RARE_CLASS * n_rare(rare_f1_vals[k])
            cost = W_INFER * n_infer(m["inference_time_per_sample_ms"]) + W_TRAIN * n_train(m["train_time_sec"])
            composite = QUALITY_WEIGHT * quality + COST_WEIGHT * cost
            scores[k] = {
                "name": m["name"], "family": m["family"],
                "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"], "accuracy": m["accuracy"],
                "rare_class_f1": rare_f1_vals[k], "rare_class": rare_cls,
                "train_time_sec": m["train_time_sec"], "infer_ms": m["inference_time_per_sample_ms"],
                "model_size": m["model_size"], "subsample_cap": m.get("subsample_cap"),
                "quality_score": round(quality * 100, 2), "cost_score": round(cost * 100, 2),
                "composite_score": round(composite * 100, 2),
            }
        per_dataset_scores[ds] = scores

    common_keys = set(per_dataset_scores["mqttset"]) & set(per_dataset_scores["datasense"])
    combined = []
    for k in common_keys:
        m = per_dataset_scores["mqttset"][k]
        d = per_dataset_scores["datasense"][k]
        combined.append({
            "key": k, "name": m["name"], "family": m["family"],
            "mqttset": m, "datasense": d,
            "avg_composite": round((m["composite_score"] + d["composite_score"]) / 2, 2),
            "avg_quality": round((m["quality_score"] + d["quality_score"]) / 2, 2),
            "avg_cost": round((m["cost_score"] + d["cost_score"]) / 2, 2),
            "consistency_gap": round(abs(m["composite_score"] - d["composite_score"]), 2),
        })
    combined.sort(key=lambda x: -x["avg_composite"])

    output = {
        "methodology": {
            "quality_weight": QUALITY_WEIGHT, "cost_weight": COST_WEIGHT,
            "w_f1_macro": W_F1_MACRO, "w_f1_weighted": W_F1_WEIGHTED, "w_rare_class": W_RARE_CLASS,
            "w_infer": W_INFER, "w_train": W_TRAIN,
        },
        "rarest": rarest,
        "per_dataset": per_dataset_scores,
        "combined": combined,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"{len(combined)} models scored -> {OUT_PATH}")
    print("\nTop 10 by combined composite score:")
    for r in combined[:10]:
        print(f"  {r['avg_composite']:6.2f}  {r['name']}")


if __name__ == "__main__":
    main()
