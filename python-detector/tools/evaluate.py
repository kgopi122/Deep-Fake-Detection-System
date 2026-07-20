from __future__ import annotations

import argparse
import csv
from dataclasses import asdict

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from veritrue_detector import DetectorConfig, detect_image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV with columns: path,label (0 real, 1 ai)")
    p.add_argument("--threshold", type=int, default=65)
    args = p.parse_args()

    cfg = DetectorConfig()

    y_true: list[int] = []
    y_score: list[float] = []

    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            path = row["path"]
            label = int(row["label"])
            rep = detect_image(path, cfg)
            y_true.append(label)
            y_score.append(float(rep.ai_likelihood_0_100) / 100.0)

    y_true_np = np.array(y_true)
    y_score_np = np.array(y_score)
    y_pred = (y_score_np >= (args.threshold / 100.0)).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true_np, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true_np, y_pred)

    try:
        roc = roc_auc_score(y_true_np, y_score_np)
    except Exception:
        roc = float("nan")

    fpr, tpr, _ = roc_curve(y_true_np, y_score_np)
    roc_auc = auc(fpr, tpr)

    print("Samples:", len(y_true))
    print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)


if __name__ == "__main__":
    main()
