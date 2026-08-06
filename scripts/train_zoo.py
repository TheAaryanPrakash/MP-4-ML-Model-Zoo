"""
Train every model in the zoo (scripts/model_zoo.py) on one dataset and save
a common_eval-schema result JSON per model to results/<dataset>/<key>.json.

Each model is trained in its own subprocess with a hard wall-clock timeout:
some libsvm-backed / C-extension fits can't be interrupted with a plain
signal (the GIL only sees control back once the blocking C call returns), so
a timeout has to be enforced by killing the process, not by aborting in-line.

Usage:
    python train_zoo.py --dataset mqttset
    python train_zoo.py --dataset datasense
    python train_zoo.py --dataset mqttset --only random_forest,elm
"""
import argparse
import multiprocessing as mp
import sys
import time

import numpy as np

from model_zoo import build_registry
import common_eval as ce

SEED = 42
TIMEOUT_SEC = 900  # 15 min hard ceiling per model


def get_model_size(model, key):
    try:
        if hasattr(model, "n_support_"):
            return f"{int(np.sum(model.n_support_))} support vectors"
        if hasattr(model, "estimators_"):
            return f"{len(model.estimators_)} base estimators"
        if hasattr(model, "tree_"):
            return f"{model.tree_.node_count} nodes"
        if hasattr(model, "coef_"):
            return f"{np.asarray(model.coef_).size} weights"
        if key == "elm":
            return f"{model.beta_.size} readout weights"
        if key == "som_classifier":
            return f"{model.grid_size}x{model.grid_size} map units"
        if key in ("knn5", "knn15_distance"):
            return "lazy (stores full training set)"
        if key == "nearest_centroid":
            return f"{model.centroids_.size} centroid values"
    except Exception:
        pass
    return "n/a"


def subsample(X, y, cap, seed=SEED):
    if X.shape[0] <= cap:
        return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=cap, replace=False)
    return X[idx], y[idx]


def _worker(spec, dataset, X_train, y_train, X_test, y_test, class_names, n_classes, n_features, out_q):
    key, name, family = spec["key"], spec["name"], spec["family"]
    cap = spec["subsample_cap"]
    Xtr, ytr = (subsample(X_train, y_train, cap) if cap else (X_train, y_train))
    hyperparams = {"subsample_cap": cap, "seed": SEED}

    try:
        model = spec["build"](n_classes, n_features, SEED)

        def _json_safe(v):
            # some libraries (e.g. XGBoost's "missing" param) default to
            # float('nan'), which Python's json module writes as a bare NaN
            # token -- invalid JSON, so the browser's JSON.parse rejects it.
            if isinstance(v, float) and v != v:
                return None
            return v

        hyperparams.update({k: _json_safe(v) for k, v in getattr(model, "get_params", lambda: {})().items()
                             if isinstance(v, (int, float, str, bool)) or v is None})

        t0 = time.time()
        model.fit(Xtr, ytr)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test)
        infer_time = time.time() - t0

        result = ce.build_multiclass_result(
            name=name, key=key, family=family, description=spec["description"],
            hyperparams=hyperparams,
            y_test=y_test, y_pred=y_pred, class_names=class_names,
            train_time_sec=train_time, infer_time_sec=infer_time,
            n_train=Xtr.shape[0], n_test=X_test.shape[0], n_features=n_features,
            model_size=get_model_size(model, key),
            subsample_cap=cap,
        )
        ce.save_result(result, dataset, f"{key}.json")
        out_q.put(("ok", result["f1_macro"]))
    except Exception as e:
        result = ce.build_failed_result(
            name=name, key=key, family=family, description=spec["description"],
            hyperparams=hyperparams, error=e,
        )
        ce.save_result(result, dataset, f"{key}.json")
        out_q.put(("failed", None))


def run_with_timeout(spec, dataset, X_train, y_train, X_test, y_test, class_names, n_classes, n_features):
    # fork (not spawn): the model registry's build functions are lambdas,
    # which spawn can't pickle across the process boundary. fork copies the
    # parent's memory directly instead, so no serialization is needed.
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(spec, dataset, X_train, y_train, X_test, y_test,
                                           class_names, n_classes, n_features, q))
    p.start()
    p.join(TIMEOUT_SEC)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        result = ce.build_failed_result(
            name=spec["name"], key=spec["key"], family=spec["family"], description=spec["description"],
            hyperparams={"subsample_cap": spec["subsample_cap"], "seed": SEED},
            error=f"timed out after {TIMEOUT_SEC}s (killed)",
        )
        ce.save_result(result, dataset, f"{spec['key']}.json")
        return "failed", None
    try:
        return q.get_nowait()
    except Exception:
        return "failed", None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mqttset", "datasense"])
    parser.add_argument("--only", default=None, help="comma-separated list of model keys to run")
    args = parser.parse_args()

    if args.dataset == "mqttset":
        from preprocess_mqttset import load_data
    else:
        from preprocess_datasense import load_data

    print(f"Loading {args.dataset} data...", flush=True)
    data = load_data(scale=True)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    class_names = data["class_names"]
    n_classes = len(class_names)
    n_features = X_train.shape[1]
    print(f"X_train={X_train.shape} X_test={X_test.shape} classes={class_names}", flush=True)

    registry = build_registry(seed=SEED)
    if args.only:
        wanted = set(args.only.split(","))
        registry = [s for s in registry if s["key"] in wanted]

    n_total = len(registry)
    results_summary = []

    for i, spec in enumerate(registry, 1):
        print(f"\n=== [{i}/{n_total}] {spec['name']} ({spec['key']}) ===", flush=True)
        status, f1 = run_with_timeout(spec, args.dataset, X_train, y_train, X_test, y_test,
                                       class_names, n_classes, n_features)
        results_summary.append((spec["key"], status, f1))

    ok = sum(1 for _, status, _ in results_summary if status == "ok")
    print(f"\n\n=== DONE: {ok}/{n_total} models succeeded on {args.dataset} ===", flush=True)
    for key, status, f1 in results_summary:
        marker = f"f1_macro={f1:.4f}" if f1 is not None else "FAILED"
        print(f"  {key:30s} {status:8s} {marker}", flush=True)


if __name__ == "__main__":
    main()
