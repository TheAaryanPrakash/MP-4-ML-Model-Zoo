"""
Feature-selection deep dive for the "Deep Dive" dashboard tab.

Retrains one representative model per major family (boosting/tree, bagging,
linear, instance-based, interpretable-GAM) on each dataset's existing
fixed-seed split, then computes:
  1. Permutation importance (model-agnostic, comparable across every family)
     on a fixed subsample of the held-out test set.
  2. Native impurity/coefficient-based importance where available, as a
     cross-check against permutation importance.

Also runs univariate filter-method feature selection (mutual information +
ANOVA F-test) against the binary benign-vs-malicious relabeling of each
dataset's target -- this answers "which raw features actually separate
malicious from benign" without training a new classifier for that question,
reusing the same train split every other model in the zoo sees.

Output: results/feature_importance/<dataset>.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "feature_importance")
MODERATE_CAP = 40000  # same cap model_zoo.py uses for EBM/RF-scale models
KNN_CAP = 15000  # KNN is lazy: permutation_importance re-runs predict() ~(n_features+1)*n_repeats
                  # times, and predict cost scales with the stored training-set size, so this
                  # analysis-only cap (tighter than train_zoo.py's, which doesn't cap knn5 at all)
                  # keeps runtime bounded
PERM_TEST_SUBSAMPLE = 2000  # keep permutation_importance runtime sane
PERM_REPEATS = 5

BALANCED = dict(class_weight="balanced")

BENIGN_LABEL = {"mqttset": "legitimate", "datasense": "benign"}


def subsample(X, y, cap, seed=SEED):
    if X.shape[0] <= cap:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx], y[idx]


def build_xgb(n_classes):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", random_state=SEED,
        tree_method="hist", n_jobs=1,
    )


def build_ebm():
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(interactions=0, random_state=SEED, n_jobs=1)


# every estimator pinned to n_jobs=1: permutation_importance itself parallelizes
# over features/repeats, and nesting joblib parallelism inside joblib parallelism
# (estimator n_jobs=-1 called from worker processes spawned by n_jobs=-1 above)
# deadlocks under the loky backend on macOS rather than just running slower.
def get_family_models(n_classes):
    return {
        "xgboost": ("boosting", build_xgb(n_classes), MODERATE_CAP),
        "random_forest": ("ensemble_bagging",
                           RandomForestClassifier(n_estimators=200, **BALANCED, random_state=SEED, n_jobs=1),
                           None),
        "logreg": ("linear", LogisticRegression(max_iter=1000, **BALANCED, random_state=SEED), None),
        "knn5": ("instance", KNeighborsClassifier(n_neighbors=5, n_jobs=1), KNN_CAP),
        "ebm": ("interpretable", build_ebm(), MODERATE_CAP),
    }


def native_importance(key, model, feature_names):
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=float)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        return coef
    if key == "ebm":
        try:
            data = model.explain_global().data()
            scores = np.asarray(data["scores"], dtype=float)
            # explain_global() labels terms "feature_0000", "feature_0001", ...
            # (we fit on a raw numpy array, not a DataFrame, so real column
            # names never reach the library) -- but with interactions=0 there
            # is exactly one term per input feature, in input column order, so
            # a positional zip against feature_names is exact, not a guess.
            if len(scores) != len(feature_names):
                return None
            return np.abs(scores)
        except Exception:
            return None
    return None


def top_k(names, values, k=15):
    values = np.asarray(values, dtype=float)
    total = values.sum()
    if total > 0:
        values = values / total
    order = np.argsort(values)[::-1][:k]
    return [{"feature": names[i], "importance": float(values[i])} for i in order]


def run_dataset(dataset):
    print(f"\n{'='*60}\n{dataset}\n{'='*60}", flush=True)
    if dataset == "mqttset":
        from preprocess_mqttset import load_data
    else:
        from preprocess_datasense import load_data

    data = load_data(scale=True)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]
    class_names = data["class_names"]
    n_classes = len(class_names)
    n_features = X_train.shape[1]

    benign_name = BENIGN_LABEL[dataset]
    benign_idx = class_names.index(benign_name)
    y_train_bin = (y_train != benign_idx).astype(int)  # 1 = malicious, 0 = benign

    rng = np.random.RandomState(SEED)
    perm_idx = rng.choice(X_test.shape[0], size=min(PERM_TEST_SUBSAMPLE, X_test.shape[0]), replace=False)
    X_perm, y_perm = X_test[perm_idx], y_test[perm_idx]

    models = get_family_models(n_classes)
    per_model = {}

    for key, (family, model, cap) in models.items():
        print(f"-- {key} ({family}) --", flush=True)
        Xtr, ytr = subsample(X_train, y_train, cap) if cap else (X_train, y_train)

        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_time = time.time() - t0

        entry = {"family": family, "fit_time_sec": fit_time}

        native = native_importance(key, model, feature_names)
        if native is not None:
            entry["native_importance_top15"] = top_k(feature_names, native)

        t0 = time.time()
        try:
            pi = permutation_importance(
                model, X_perm, y_perm, n_repeats=PERM_REPEATS,
                random_state=SEED, scoring="f1_macro", n_jobs=4,
            )
            perm_time = time.time() - t0
            importances = np.clip(pi.importances_mean, a_min=0, a_max=None)
            entry["permutation_importance_top15"] = top_k(feature_names, importances)
            entry["permutation_importance_runtime_sec"] = perm_time
        except Exception as e:
            entry["permutation_importance_error"] = str(e)[:300]

        per_model[key] = entry
        print(f"   fit={fit_time:.2f}s", flush=True)

    print("-- univariate filter-method feature selection (benign vs malicious) --", flush=True)
    t0 = time.time()
    mi = mutual_info_classif(X_train, y_train_bin, random_state=SEED, n_jobs=-1)
    mi_time = time.time() - t0
    f_stat, f_pval = f_classif(X_train, y_train_bin)

    benign_vs_malicious = {
        "benign_class_name": benign_name,
        "n_benign_train": int((y_train_bin == 0).sum()),
        "n_malicious_train": int((y_train_bin == 1).sum()),
        "mutual_info_top15": top_k(feature_names, mi),
        "mutual_info_runtime_sec": mi_time,
        "anova_f_top15": top_k(feature_names, np.nan_to_num(f_stat, nan=0.0)),
    }

    result = {
        "dataset": dataset,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": n_features,
        "feature_names": feature_names,
        "class_names": class_names,
        "per_model_feature_selection": per_model,
        "benign_vs_malicious_feature_selection": benign_vs_malicious,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{dataset}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out_path}", flush=True)
    return result


if __name__ == "__main__":
    for ds in ["mqttset", "datasense"]:
        run_dataset(ds)
