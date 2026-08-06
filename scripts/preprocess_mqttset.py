"""
Preprocessing for the MQTTset dataset, reusing the exact feature engineering
and train/test split from the sibling v1 (MP-2-ML-vs-DL) project so v3's
model-zoo results stay directly comparable to v1's 11-model leaderboard.

Data is read from the sibling v1 repo's data/ directory rather than
duplicated here (the raw CSVs are gitignored in both repos anyway).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = "/Users/aaryan/Documents/University/Year 4/Semester 7/Major Project/v1/data"

HEX_COLS = ["tcp.flags", "mqtt.conack.flags", "mqtt.conflags", "mqtt.hdrflags"]


def _hex_to_int(series):
    def conv(v):
        v = str(v)
        try:
            if v.startswith("0x"):
                return int(v, 16)
            return int(float(v))
        except ValueError:
            return 0
    return series.map(conv)


def _engineer(df):
    df = df.copy()
    for col in HEX_COLS:
        df[col] = _hex_to_int(df[col])
    df["mqtt.protoname"] = (df["mqtt.protoname"] == "MQTT").astype(int)
    df["mqtt_msg_len"] = df["mqtt.msg"].map(
        lambda v: 0 if pd.isna(v) or str(v) in ("0", "nan") else len(str(v)) // 2
    )
    df = df.drop(columns=["mqtt.msg"])
    df = df.fillna(0)
    return df


def load_data(scale=True):
    train_df = pd.read_csv(f"{DATA_DIR}/train70_reduced.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test30_reduced.csv")

    train_df = _engineer(train_df)
    test_df = _engineer(test_df)

    y_train_raw = train_df.pop("target")
    y_test_raw = test_df.pop("target")

    feature_names = list(train_df.columns)

    X_train = train_df.to_numpy(dtype=np.float64)
    X_test = test_df.to_numpy(dtype=np.float64)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": le,
        "scaler": scaler,
        "feature_names": feature_names,
        "class_names": list(le.classes_),
    }
