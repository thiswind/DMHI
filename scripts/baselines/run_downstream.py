#!/usr/bin/env python3
"""
Downstream mortality prediction task for tab:downstream.

Trains a 2-layer LSTM classifier on imputed time series from each baseline method,
evaluates AUC-ROC and F1 on physionet2012 and mimic test sets.

Reads:
  runs/baselines/<method>_<dataset>_<rate>_<pattern>_imputed.npy
  data/processed/<dataset>/y_test.npy

Writes:
  runs/baselines/downstream_results.json

Usage:
    python run_downstream.py --out_dir runs/baselines
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from common import load_dataset, DATA_ROOT, imputed_filename

DOWNSTREAM_DATASETS = ["physionet2012"]
DOWNSTREAM_RATE = 0.3
DOWNSTREAM_PATTERN = "mcar"

METHODS = ["mean", "linear", "locf", "saits", "brits", "csdi", "imputeformer",
           "timemixer", "psw_i"]


# ---------------------------------------------------------------------------
# LSTM classifier
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        logits = self.classifier(h_n[-1])
        return logits.squeeze(-1)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_eval(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   device: torch.device,
                   epochs: int = 50, batch_size: int = 64,
                   lr: float = 1e-3) -> dict:
    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    D = X_tr.shape[2]
    model = LSTMClassifier(input_dim=D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size,
                        shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_te = model(X_te.to(device)).cpu().numpy()

    probs = torch.sigmoid(torch.tensor(logits_te)).numpy()
    preds = (probs >= 0.5).astype(int)
    y_np = y_te.numpy().astype(int)

    auc = float(roc_auc_score(y_np, probs)) if len(np.unique(y_np)) > 1 else 0.0
    f1 = float(f1_score(y_np, preds, zero_division=0))
    return {"auc_roc": round(auc, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/baselines")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[downstream] Using device: {device}")

    all_results = {}

    for dataset in DOWNSTREAM_DATASETS:
        X_train, M_train, y_train = load_dataset(dataset, "train")
        X_test,  M_test,  y_test  = load_dataset(dataset, "test")

        if y_train is None or y_test is None:
            print(f"[downstream] No labels for {dataset}, skipping.")
            continue

        for method in METHODS:
            imputed_path = out_dir / imputed_filename(
                method, dataset, DOWNSTREAM_RATE, DOWNSTREAM_PATTERN)

            if not imputed_path.exists():
                print(f"[downstream] Missing: {imputed_path}, skipping {method}/{dataset}")
                continue

            X_imputed_test = np.load(str(imputed_path))
            X_imputed_train = X_train.copy()

            key = f"{method}/{dataset}"
            print(f"[downstream] Training LSTM: {key} ...")
            scores = train_and_eval(
                X_imputed_train, y_train,
                X_imputed_test, y_test,
                device=device,
                epochs=args.epochs,
            )
            all_results[key] = scores
            print(f"  {key}: AUC={scores['auc_roc']:.4f}  F1={scores['f1']:.4f}")

    out_path = out_dir / "downstream_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[downstream] Saved: {out_path}")


if __name__ == "__main__":
    main()
