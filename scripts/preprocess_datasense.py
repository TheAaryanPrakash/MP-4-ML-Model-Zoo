"""
Preprocessing for the DataSense (CIC IIoT Dataset 2025) dataset, reusing the
exact feature engineering and train/test split from the sibling v2-datasense
(MP-3-DataSense-ML-vs-DL) project so v3's model-zoo results stay directly
comparable to that project's 11-model leaderboard.

Data is read from the sibling v2-datasense repo's data/ directory rather
than duplicated here (the raw CSVs are gitignored in both repos anyway).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = "/Users/aaryan/Documents/University/Year 4/Semester 7/Major Project/v2-datasense/data"

DROP_COLS = [
    "device_mac", "label_full", "label1", "label3", "label4",
    "timestamp", "timestamp_start", "timestamp_end",
    "log_data-types",
    "network_ips_all", "network_ips_dst", "network_ips_src",
    "network_macs_all", "network_macs_dst", "network_macs_src",
    "network_ports_all", "network_ports_dst", "network_ports_src",
    "network_protocols_all", "network_protocols_dst", "network_protocols_src",
]

TARGET = "label2"
CATEGORICAL_COLS = ["device_name"]

TEST_FRAC = 0.3
SEED = 42


def _load_raw():
    attack = pd.read_csv(f"{DATA_DIR}/attack_samples_5sec.csv", low_memory=False)
    benign = pd.read_csv(f"{DATA_DIR}/benign_samples_5sec.csv", low_memory=False)
    return pd.concat([attack, benign], ignore_index=True)


def load_data(scale=True):
    df = _load_raw()
    df = df.drop(columns=DROP_COLS)

    y_raw = df.pop(TARGET)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix="device")

    feature_names = list(df.columns)
    X = df.to_numpy(dtype=np.float64)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRAC, random_state=SEED, stratify=y
    )

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "label_encoder": le, "scaler": scaler,
        "feature_names": feature_names,
        "class_names": list(le.classes_),
    }
