"""
Aggregates every model's results/<dataset>/*.json into a single
frontend/data.json consumed by the static dashboard, covering both datasets
(mqttset, datasense) so the frontend can switch between them.
"""
import glob
import json
import os

ROOT = "/Users/aaryan/Documents/University/Year 4/Semester 7/Major Project/v3-ml-model-zoo"
RESULTS_DIR = f"{ROOT}/results"
FRONTEND_DIR = f"{ROOT}/frontend"

DATASETS = {
    "mqttset": {
        "key": "mqttset",
        "name": "MQTTset",
        "note": "Per-packet MQTT/TCP header features from a simulated smart-home IoT network (Vaccari et al., 2020).",
    },
    "datasense": {
        "key": "datasense",
        "name": "DataSense (CIC IIoT 2025)",
        "note": "Per-device, 5-second-window aggregated sensor + network telemetry from a 40-device IIoT testbed (Firouzi et al., 2025).",
    },
}


def load_dataset_results(dataset_key):
    pattern = f"{RESULTS_DIR}/{dataset_key}/*.json"
    files = sorted(glob.glob(pattern))
    models = []
    for path in files:
        with open(path) as f:
            models.append(json.load(f))
    return models


def main():
    output = {"datasets": {}}

    for ds_key, ds_meta in DATASETS.items():
        models = load_dataset_results(ds_key)
        ok = [m for m in models if m.get("status") == "ok"]
        failed = [m for m in models if m.get("status") == "failed"]
        ok_sorted = sorted(ok, key=lambda r: -r["f1_macro"])

        ref = ok_sorted[0] if ok_sorted else None
        dataset_info = dict(ds_meta)
        dataset_info.update({
            "classes": ref["class_names"] if ref else [],
            "n_train": ref["n_train_samples"] if ref else None,
            "n_test": ref["n_test_samples"] if ref else None,
            "n_features": ref["n_features"] if ref else None,
            "n_models_attempted": len(models),
            "n_models_ok": len(ok),
            "n_models_failed": len(failed),
        })

        families = sorted({m["family"] for m in models})

        output["datasets"][ds_key] = {
            "dataset": dataset_info,
            "models_sorted": ok_sorted,
            "failed_models": failed,
            "families": families,
        }

        print(f"[{ds_key}] {len(ok)} ok / {len(failed)} failed / {len(models)} attempted")

    os.makedirs(FRONTEND_DIR, exist_ok=True)
    out_path = f"{FRONTEND_DIR}/data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
