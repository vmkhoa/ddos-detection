#!/usr/bin/env python3
"""
train.py

Train a UDP-only DDoS detector on CICDDoS2019 parquet files.

- Loads train/test parquet files (UDP subset)
- Converts labels to binary: Attack (UDP + MSSQL) vs Benign
- Trains a Pipeline(StandardScaler + RandomForestClassifier)
- Prints classification metrics
- Saves:
    artifacts/udp_ddos_pipeline.pkl
    artifacts/udp_features.json

Usage (example):
    uv run train.py \
      --train-parquet /path/UDP-training.parquet \
      --test-parquet  /path/UDP-testing.parquet \
      --sample-n 30000
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train-parquet",
        type=str,
        required=True,
        help="Path to UDP-training.parquet",
    )
    p.add_argument(
        "--test-parquet",
        type=str,
        help="Path to UDP-testing.parquet (optional)",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=30000,
        help="Optional sample size for speed",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="artifacts",
        help="Directory to save model artifacts",
    )
    return p.parse_args()


def load_and_prepare(train_path: str, test_path: str | None, sample_n: int | None):
    dfs = []
    train_path = Path(train_path)
    if not train_path.exists():
        raise SystemExit(f"Train parquet not found: {train_path}")
    dfs.append(pd.read_parquet(train_path))

    if test_path:
        test_path = Path(test_path)
        if test_path.exists():
            dfs.append(pd.read_parquet(test_path))
        else:
            print(f"[WARN] Test parquet not found: {test_path}, continuing with train only")

    df = pd.concat(dfs, ignore_index=True)
    print("Raw shape:", df.shape)

    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"Sampled to {df.shape[0]} rows")

    # Label prep: treat UDP + MSSQL as Attack, Benign as normal
    df = df.copy()
    df["Label"] = df["Label"].replace({"UDP": "Attack", "MSSQL": "Attack"})
    df["y"] = (df["Label"] != "Benign").astype(int)

    print("Label distribution (0=Benign, 1=Attack):")
    print(df["y"].value_counts())

    # Use all numeric features except target
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "y" in num_cols:
        num_cols.remove("y")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df[num_cols]
    y = df["y"]

    return X, y, num_cols


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data…")
    X, y, features = load_and_prepare(args.train_parquet, args.test_parquet, args.sample_n)
    print(f"Using {len(features)} numeric features")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Building pipeline (StandardScaler + RandomForest)…")
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("Training model…")
    pipe.fit(X_train, y_train)

    print("Evaluating on hold-out set…")
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save artifacts
    model_path = out_dir / "udp_ddos_pipeline.pkl"
    features_path = out_dir / "udp_features.json"
    joblib.dump(pipe, model_path)
    json.dump(features, open(features_path, "w"))

    print(f"Saved model to {model_path}")
    print(f"Saved features to {features_path}")


if __name__ == "__main__":
    main()
