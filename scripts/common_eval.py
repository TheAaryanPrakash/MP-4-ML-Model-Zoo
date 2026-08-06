"""
Shared evaluation/result-schema helpers, adapted from v1/v2-datasense's
common_eval.py so every model in the zoo (successful or failed) produces a
directly comparable JSON record. RESULTS_DIR is parameterized per dataset
since v3 covers two datasets from a single repo.
"""
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

RESULTS_ROOT = "/Users/aaryan/Documents/University/Year 4/Semester 7/Major Project/v3-ml-model-zoo/results"


def build_multiclass_result(*, name, key, family, description, hyperparams,
                             y_test, y_pred, class_names,
                             train_time_sec, infer_time_sec, n_train, n_test,
                             n_features, model_size, subsample_cap=None, extra=None):
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names)))).tolist()
    report = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0,
        labels=list(range(len(class_names))),
    )

    per_class = []
    p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=list(range(len(class_names)))
    )
    for i, cn in enumerate(class_names):
        per_class.append({
            "class": cn,
            "precision": float(p_c[i]),
            "recall": float(r_c[i]),
            "f1": float(f1_c[i]),
            "support": int(sup_c[i]),
        })

    result = {
        "status": "ok",
        "name": name,
        "key": key,
        "category": "ml",
        "family": family,
        "description": description,
        "hyperparams": hyperparams,
        "subsample_cap": subsample_cap,
        "train_time_sec": train_time_sec,
        "inference_time_sec": infer_time_sec,
        "inference_time_per_sample_ms": (infer_time_sec / n_test) * 1000,
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "class_names": class_names,
        "per_class": per_class,
        "classification_report": report,
        "model_size": model_size,
        "n_train_samples": int(n_train),
        "n_test_samples": int(n_test),
        "n_features": int(n_features),
    }
    if extra:
        result.update(extra)

    print(f"[{name}] train={train_time_sec:.2f}s infer={infer_time_sec:.4f}s "
          f"acc={acc:.4f} f1_macro={f1_macro:.4f} f1_weighted={f1_weighted:.4f}")

    return result


def build_failed_result(*, name, key, family, description, hyperparams, error):
    print(f"[{name}] FAILED: {error}")
    return {
        "status": "failed",
        "name": name,
        "key": key,
        "category": "ml",
        "family": family,
        "description": description,
        "hyperparams": hyperparams,
        "error": str(error)[:500],
    }


def save_result(result, dataset, filename):
    out_dir = f"{RESULTS_ROOT}/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{filename}", "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> results/{dataset}/{filename}")
