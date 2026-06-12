#!/usr/bin/env python3
"""
Leakage-aware GriffinGNN experiment for Bitcoin fraud detection.

Article: When Graph Connectivity Becomes Leakage: A Leakage-Aware Evaluation
of Graph Neural Networks for Bitcoin Fraud Detection

This script is a GitHub-ready version of the original Colab GriffinGNN v26 run.
It trains a hybrid heterogeneous GNN + tabular model under a chronological split:
Train = D1+D2+D3, Validation = D4, Test = D5+D6.

Expected inputs:
  1) SQLite database with tx_inputs and tx_outputs tables
  2) CSV dataset containing tx_hash, snapshot_id, label_final, and tabular features

Example:
python experiments/run_leakage_aware_griffin_gnn.py \
  --db data/extended/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db \
  --labels data/processed_v21_final/dataset_v21_full.csv \
  --out-dir results/leakage_aware_griffin_gnn
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sqlite3
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


DEFAULT_FEATURE_COLS = [
    "input_count",
    "output_count",
    "input_addr_count",
    "coinbase_flag",
    "has_witness",
    "script_type_encoded",
    "input_addr_concentration",
    "io_count_ratio",
    "tx_weight",
    "avg_input_value",
    "total_input_scaled",
    "log_output_value",
    "fee_ratio",
    "prev_addr_seen_ratio",
    "prev_addr_seen_count",
]

SNAPSHOT_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6"]
TRAIN_SNAPS = ["D1", "D2", "D3"]
VAL_SNAPS = ["D4"]
TEST_SNAPS = ["D5", "D6"]

NUM_NEIGHBORS = {
    ("address", "input_to_tx", "transaction"): [20, 15],
    ("address", "output_to_tx", "transaction"): [20, 15],
    ("transaction", "tx_to_input", "address"): [15, 10],
    ("transaction", "tx_to_output", "address"): [15, 10],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware GriffinGNN experiment")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--labels", required=True, help="Path to labelled CSV dataset")
    parser.add_argument("--out-dir", default="results/leakage_aware_griffin_gnn", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_inputs_outputs(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = sqlite3.connect(db_path)
    try:
        inputs_df = pd.read_sql_query("SELECT tx_hash, input_address FROM tx_inputs", conn)
        outputs_df = pd.read_sql_query("SELECT tx_hash, output_address FROM tx_outputs", conn)
    finally:
        conn.close()
    return inputs_df, outputs_df


def load_transactions(labels_path: str, feature_cols: List[str]) -> pd.DataFrame:
    labels_df = pd.read_csv(labels_path)
    required = ["tx_hash", "snapshot_id", "label_final"] + feature_cols
    missing = [c for c in required if c not in labels_df.columns]
    if missing:
        raise ValueError(f"Missing columns in labels CSV: {missing}")

    tx = labels_df[required].copy()
    tx = tx[tx["snapshot_id"].isin(SNAPSHOT_ORDER)].copy()
    tx["snapshot_rank"] = tx["snapshot_id"].map({s: i for i, s in enumerate(SNAPSHOT_ORDER)})
    return tx.sort_values(["snapshot_rank", "tx_hash"]).reset_index(drop=True)


def make_masks(transactions: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = transactions["snapshot_id"].isin(TRAIN_SNAPS).values
    val_mask = transactions["snapshot_id"].isin(VAL_SNAPS).values
    test_mask = transactions["snapshot_id"].isin(TEST_SNAPS).values
    y = transactions["label_final"].astype(int).values
    snap_rank = transactions["snapshot_rank"].values
    return train_mask, val_mask, test_mask, y, snap_rank


def scale_transaction_features(transactions: pd.DataFrame, feature_cols: List[str], train_mask: np.ndarray,
                               val_mask: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    x_raw = (
        transactions[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .values.astype(np.float32)
    )
    scaler = StandardScaler()
    x_scaled = np.zeros_like(x_raw, dtype=np.float32)
    x_scaled[train_mask] = scaler.fit_transform(x_raw[train_mask])
    x_scaled[val_mask] = scaler.transform(x_raw[val_mask])
    x_scaled[test_mask] = scaler.transform(x_raw[test_mask])
    return x_scaled


def prepare_edges(inputs_df: pd.DataFrame, outputs_df: pd.DataFrame, transactions: pd.DataFrame):
    tx_hashes = transactions["tx_hash"].values
    tx_hash_to_idx = {h: i for i, h in enumerate(tx_hashes)}
    tx_hash_set = set(tx_hashes)

    inputs_df = inputs_df[inputs_df["tx_hash"].isin(tx_hash_set)].dropna().copy()
    outputs_df = outputs_df[outputs_df["tx_hash"].isin(tx_hash_set)].dropna().copy()
    inputs_df["tx_idx"] = inputs_df["tx_hash"].map(tx_hash_to_idx)
    outputs_df["tx_idx"] = outputs_df["tx_hash"].map(tx_hash_to_idx)
    inputs_df["input_address"] = inputs_df["input_address"].astype(str)
    outputs_df["output_address"] = outputs_df["output_address"].astype(str)

    all_addresses = pd.concat([inputs_df["input_address"], outputs_df["output_address"]]).dropna().unique()
    addr_to_idx = {a: i for i, a in enumerate(all_addresses)}
    inputs_df["addr_idx"] = inputs_df["input_address"].map(addr_to_idx)
    outputs_df["addr_idx"] = outputs_df["output_address"].map(addr_to_idx)
    inputs_df = inputs_df.dropna(subset=["tx_idx", "addr_idx"]).copy()
    outputs_df = outputs_df.dropna(subset=["tx_idx", "addr_idx"]).copy()
    inputs_df[["tx_idx", "addr_idx"]] = inputs_df[["tx_idx", "addr_idx"]].astype(np.int64)
    outputs_df[["tx_idx", "addr_idx"]] = outputs_df[["tx_idx", "addr_idx"]].astype(np.int64)
    return inputs_df, outputs_df, len(all_addresses)


def build_address_features(inputs_df: pd.DataFrame, outputs_df: pd.DataFrame, num_addr: int,
                           history_mask: np.ndarray, labels: np.ndarray,
                           snap_rank_array: np.ndarray) -> np.ndarray:
    hist_idx = set(np.where(history_mask)[0].tolist())
    hist_fraud = set(np.where(history_mask & (labels == 1))[0].tolist())

    inp = inputs_df[inputs_df["tx_idx"].isin(hist_idx)].copy()
    out = outputs_df[outputs_df["tx_idx"].isin(hist_idx)].copy()
    rank_map = dict(zip(np.where(history_mask)[0], snap_rank_array[history_mask]))
    inp["weight"] = (inp["tx_idx"].map(rank_map).fillna(0) + 1).astype(np.float32)
    out["weight"] = (out["tx_idx"].map(rank_map).fillna(0) + 1).astype(np.float32)

    in_deg = np.bincount(inp["addr_idx"].values, minlength=num_addr)
    out_deg = np.bincount(out["addr_idx"].values, minlength=num_addr)
    total_deg = in_deg + out_deg
    active_both = ((in_deg > 0) & (out_deg > 0)).astype(np.float32)
    in_out_ratio = (in_deg + 1.0) / (out_deg + 1.0)
    recency_in = np.bincount(inp["addr_idx"].values, weights=inp["weight"].values, minlength=num_addr)
    recency_out = np.bincount(out["addr_idx"].values, weights=out["weight"].values, minlength=num_addr)
    recency_score = recency_in + recency_out

    inp_f = inputs_df[inputs_df["tx_idx"].isin(hist_fraud)]
    out_f = outputs_df[outputs_df["tx_idx"].isin(hist_fraud)]
    total_fraud = (
        np.bincount(inp_f["addr_idx"].values, minlength=num_addr)
        + np.bincount(out_f["addr_idx"].values, minlength=num_addr)
    )
    fraud_ratio = total_fraud / (total_deg + 1e-8)

    addr_tx = pd.concat([inp[["addr_idx", "tx_idx"]], out[["addr_idx", "tx_idx"]]])
    if len(addr_tx) > 0:
        unique_tx = addr_tx.groupby("addr_idx")["tx_idx"].nunique().reindex(range(num_addr), fill_value=0).values
        nb_links = addr_tx.groupby("addr_idx")["tx_idx"].count().reindex(range(num_addr), fill_value=0).values
    else:
        unique_tx = nb_links = np.zeros(num_addr)
    activity_ratio = (total_deg + 1.0) / (unique_tx + 1.0)

    return np.vstack([
        np.log1p(in_deg), np.log1p(out_deg), np.log1p(total_deg),
        np.log1p(in_out_ratio), active_both,
        np.log1p(unique_tx), np.log1p(nb_links), np.log1p(activity_ratio),
        fraud_ratio, np.log1p(total_fraud), np.log1p(recency_score),
    ]).T.astype(np.float32)


def compute_connectivity_mask(inputs_df: pd.DataFrame, outputs_df: pd.DataFrame, train_mask: np.ndarray,
                              val_mask: np.ndarray, test_mask: np.ndarray, num_tx: int) -> np.ndarray:
    train_val_ids = set(np.where(train_mask | val_mask)[0].tolist())
    test_ids = set(np.where(test_mask)[0].tolist())
    inp_tv = inputs_df[inputs_df["tx_idx"].isin(train_val_ids)]
    out_tv = outputs_df[outputs_df["tx_idx"].isin(train_val_ids)]
    known_addrs = set(inp_tv["addr_idx"].values) | set(out_tv["addr_idx"].values)
    inp_test = inputs_df[inputs_df["tx_idx"].isin(test_ids)]
    out_test = outputs_df[outputs_df["tx_idx"].isin(test_ids)]
    connected_tx = (
        set(inp_test[inp_test["addr_idx"].isin(known_addrs)]["tx_idx"].values)
        | set(out_test[out_test["addr_idx"].isin(known_addrs)]["tx_idx"].values)
    )
    mask = np.zeros(num_tx, dtype=bool)
    mask[list(connected_tx)] = True
    return mask


def make_hetero_data(x_scaled: np.ndarray, addr_x: np.ndarray, labels: np.ndarray,
                     inp: pd.DataFrame, out: pd.DataFrame) -> HeteroData:
    data = HeteroData()
    data["transaction"].x = torch.tensor(x_scaled, dtype=torch.float32)
    data["transaction"].y = torch.tensor(labels, dtype=torch.long)
    data["address"].x = torch.tensor(addr_x, dtype=torch.float32)
    data["address", "input_to_tx", "transaction"].edge_index = torch.tensor(
        np.vstack([inp["addr_idx"].values, inp["tx_idx"].values]), dtype=torch.long
    )
    data["address", "output_to_tx", "transaction"].edge_index = torch.tensor(
        np.vstack([out["addr_idx"].values, out["tx_idx"].values]), dtype=torch.long
    )
    data["transaction", "tx_to_input", "address"].edge_index = torch.tensor(
        np.vstack([inp["tx_idx"].values, inp["addr_idx"].values]), dtype=torch.long
    )
    data["transaction", "tx_to_output", "address"].edge_index = torch.tensor(
        np.vstack([out["tx_idx"].values, out["addr_idx"].values]), dtype=torch.long
    )
    return data


def build_data_split(x_scaled, addr_x, labels, inputs_df, outputs_df, allowed_mask):
    allowed = set(np.where(allowed_mask)[0].tolist())
    return make_hetero_data(
        x_scaled,
        addr_x,
        labels,
        inputs_df[inputs_df["tx_idx"].isin(allowed)],
        outputs_df[outputs_df["tx_idx"].isin(allowed)],
    )


def build_data_test(x_scaled, addr_x, labels, inputs_df, outputs_df, train_mask, val_mask, test_mask):
    train_val_ids = set(np.where(train_mask | val_mask)[0].tolist())
    test_ids = set(np.where(test_mask)[0].tolist())
    inp_tv = inputs_df[inputs_df["tx_idx"].isin(train_val_ids)]
    out_tv = outputs_df[outputs_df["tx_idx"].isin(train_val_ids)]
    known_addrs = set(inp_tv["addr_idx"].values) | set(out_tv["addr_idx"].values)
    inp_test = inputs_df[inputs_df["tx_idx"].isin(test_ids)]
    out_test = outputs_df[outputs_df["tx_idx"].isin(test_ids)]
    inp_kept = inp_test[inp_test["addr_idx"].isin(known_addrs)]
    out_kept = out_test[out_test["addr_idx"].isin(known_addrs)]
    return make_hetero_data(
        x_scaled,
        addr_x,
        labels,
        pd.concat([inp_tv, inp_kept], ignore_index=True),
        pd.concat([out_tv, out_kept], ignore_index=True),
    )


class GriffinLeakageAwareGNN(nn.Module):
    """Hybrid heterogeneous GNN + tabular model used in the leakage-aware experiment."""

    def __init__(self, tx_in_dim: int, addr_in_dim: int, hidden_dim: int = 192, heads: int = 4,
                 dropout: float = 0.15):
        super().__init__()
        assert hidden_dim % heads == 0
        head_dim = hidden_dim // heads
        self.dropout_val = dropout

        self.tx_proj = nn.Sequential(nn.Linear(tx_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.addr_proj = nn.Sequential(nn.Linear(addr_in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))

        self.conv1 = HeteroConv({
            ("address", "input_to_tx", "transaction"): GATConv((hidden_dim, hidden_dim), head_dim, heads=heads, dropout=dropout, add_self_loops=False),
            ("address", "output_to_tx", "transaction"): GATConv((hidden_dim, hidden_dim), head_dim, heads=heads, dropout=dropout, add_self_loops=False),
            ("transaction", "tx_to_input", "address"): GATConv((hidden_dim, hidden_dim), head_dim, heads=heads, dropout=dropout, add_self_loops=False),
            ("transaction", "tx_to_output", "address"): GATConv((hidden_dim, hidden_dim), head_dim, heads=heads, dropout=dropout, add_self_loops=False),
        }, aggr="sum")
        self.norm_tx1 = nn.LayerNorm(hidden_dim)
        self.norm_ad1 = nn.LayerNorm(hidden_dim)

        self.conv2 = HeteroConv({
            ("address", "input_to_tx", "transaction"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("address", "output_to_tx", "transaction"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("transaction", "tx_to_input", "address"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("transaction", "tx_to_output", "address"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr="sum")
        self.norm_tx2 = nn.LayerNorm(hidden_dim)

        self.tab_fc1 = nn.Linear(tx_in_dim, hidden_dim)
        self.tab_bn1 = nn.BatchNorm1d(hidden_dim)
        self.tab_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tab_bn2 = nn.BatchNorm1d(hidden_dim)
        self.tab_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.tab_bn3 = nn.BatchNorm1d(hidden_dim)
        self.tab_fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.tab_bn4 = nn.BatchNorm1d(hidden_dim)
        self.tab_skip = nn.Linear(tx_in_dim, hidden_dim)

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 2), nn.Softmax(dim=-1),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.skip_proj = nn.Sequential(nn.Linear(tx_in_dim, hidden_dim // 4), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.GELU(), nn.Dropout(dropout / 2), nn.Linear(hidden_dim // 4, 2),
        )

    def _tab_forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = F.dropout(F.gelu(self.tab_bn1(self.tab_fc1(x))), p=self.dropout_val, training=self.training)
        h1 = F.dropout(F.gelu(self.tab_bn2(self.tab_fc2(h0))), p=self.dropout_val, training=self.training) + h0
        h2 = F.dropout(F.gelu(self.tab_bn3(self.tab_fc3(h1))), p=self.dropout_val, training=self.training) + h1
        h3 = F.dropout(F.gelu(self.tab_bn4(self.tab_fc4(h2))), p=self.dropout_val, training=self.training) + h2
        return h3 + F.gelu(self.tab_skip(x))

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_tx = data["transaction"].x
        h_tx = self.tx_proj(x_tx)
        h_addr = self.addr_proj(data["address"].x)
        h1 = self.conv1({"transaction": h_tx, "address": h_addr}, data.edge_index_dict)
        h_tx = F.dropout(F.gelu(self.norm_tx1(h_tx + h1.get("transaction", h_tx))), p=self.dropout_val, training=self.training)
        h_addr = F.dropout(F.gelu(self.norm_ad1(h_addr + h1.get("address", h_addr))), p=self.dropout_val, training=self.training)
        h2 = self.conv2({"transaction": h_tx, "address": h_addr}, data.edge_index_dict)
        h_gnn = F.dropout(F.gelu(self.norm_tx2(h_tx + h2.get("transaction", h_tx))), p=self.dropout_val, training=self.training)
        h_tab = self._tab_forward(x_tx)
        gate = self.fusion_gate(torch.cat([h_gnn, h_tab], dim=-1))
        h_fuse = gate[:, 0:1] * h_gnn + gate[:, 1:2] * h_tab
        h_fuse = F.dropout(F.gelu(self.fusion_norm(h_fuse)), p=self.dropout_val, training=self.training)
        return self.classifier(torch.cat([h_fuse, self.skip_proj(x_tx)], dim=-1))


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma_pos: float = 2.0, gamma_neg: float = 1.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        gamma = torch.where(
            targets == 1,
            torch.tensor(self.gamma_pos, device=logits.device),
            torch.tensor(self.gamma_neg, device=logits.device),
        )
        return (((1 - pt) ** gamma) * ce).mean()


@torch.no_grad()
def predict_loader(model: nn.Module, loader: NeighborLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all, true_all = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        bs = batch["transaction"].batch_size
        logits = torch.nan_to_num(out[:bs], nan=0.0, posinf=20.0, neginf=-20.0)
        yb = batch["transaction"].y[:bs]
        probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs_all.append(np.nan_to_num(probs, nan=0.0))
        true_all.append(yb.detach().cpu().numpy())
        del batch, out, logits, yb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(probs_all), np.concatenate(true_all)


def metrics_at_threshold(probs: np.ndarray, true: np.ndarray, th: float) -> Dict[str, float]:
    pred = (np.nan_to_num(probs, nan=0.0) >= th).astype(int)
    return {
        "AUC_ROC": float(roc_auc_score(true, probs)),
        "AUC_PR": float(average_precision_score(true, probs)),
        "F1": float(f1_score(true, pred, zero_division=0)),
        "Precision": float(precision_score(true, pred, zero_division=0)),
        "Recall": float(recall_score(true, pred, zero_division=0)),
        "Accuracy": float(accuracy_score(true, pred)),
    }


def best_threshold_fbeta(probs: np.ndarray, true: np.ndarray, beta: float = 1.0) -> Tuple[float, float]:
    prec, rec, thresholds = precision_recall_curve(true, probs)
    fb = np.where((beta ** 2 * prec + rec) > 0, (1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec + 1e-8), 0.0)
    idx = int(np.argmax(fb[:-1]))
    return float(thresholds[idx]), float(fb[idx])


def bootstrap_ci(probs: np.ndarray, true: np.ndarray, theta: float, seed: int, n_iter: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    boot = {"AUC_ROC": [], "AUC_PR": [], "F1": [], "Precision": [], "Recall": []}
    n = len(true)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        bt, bp = true[idx], probs[idx]
        if bt.sum() == 0 or bt.sum() == len(bt):
            continue
        m = metrics_at_threshold(bp, bt, theta)
        for k in boot:
            boot[k].append(m[k])
    summary = {}
    for k, vals in boot.items():
        arr = np.array(vals)
        lo, hi = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        summary[k] = {"mean": float(arr.mean()), "ci_2.5": float(lo), "ci_97.5": float(hi), "width": float(hi - lo)}
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {device}")

    inputs_df, outputs_df = load_inputs_outputs(args.db)
    transactions = load_transactions(args.labels, DEFAULT_FEATURE_COLS)
    train_mask, val_mask, test_mask, y, snap_rank = make_masks(transactions)

    print("Snapshot distribution:")
    print(transactions.groupby("snapshot_id")["label_final"].agg(["count", "sum", "mean"]))

    x_scaled = scale_transaction_features(transactions, DEFAULT_FEATURE_COLS, train_mask, val_mask, test_mask)
    inputs_df, outputs_df, num_addr = prepare_edges(inputs_df, outputs_df, transactions)
    num_tx = len(transactions)
    print(f"Graph: {num_tx:,} transactions | {num_addr:,} addresses | {len(inputs_df):,} input edges | {len(outputs_df):,} output edges")

    addr_raw = build_address_features(inputs_df, outputs_df, num_addr, train_mask, y, snap_rank)
    train_tx_set = set(np.where(train_mask)[0].tolist())
    train_addr_idx = np.unique(pd.concat([
        inputs_df[inputs_df["tx_idx"].isin(train_tx_set)]["addr_idx"],
        outputs_df[outputs_df["tx_idx"].isin(train_tx_set)]["addr_idx"],
    ]).values)
    addr_scaler = StandardScaler()
    addr_scaler.fit(addr_raw[train_addr_idx])
    addr_x = addr_scaler.transform(addr_raw).astype(np.float32)

    connectivity_mask = compute_connectivity_mask(inputs_df, outputs_df, train_mask, val_mask, test_mask, num_tx)
    print(f"Test connected transactions: {connectivity_mask[test_mask].mean():.2%}")

    data_train = build_data_split(x_scaled, addr_x, y, inputs_df, outputs_df, train_mask)
    data_val = build_data_split(x_scaled, addr_x, y, inputs_df, outputs_df, train_mask | val_mask)
    data_test = build_data_test(x_scaled, addr_x, y, inputs_df, outputs_df, train_mask, val_mask, test_mask)

    train_idx = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
    val_idx = torch.tensor(np.where(val_mask)[0], dtype=torch.long)
    test_idx = torch.tensor(np.where(test_mask)[0], dtype=torch.long)

    train_loader = NeighborLoader(data_train, input_nodes=("transaction", train_idx), num_neighbors=NUM_NEIGHBORS, batch_size=args.batch_size, shuffle=True)
    val_loader = NeighborLoader(data_val, input_nodes=("transaction", val_idx), num_neighbors=NUM_NEIGHBORS, batch_size=args.batch_size, shuffle=False)
    test_loader = NeighborLoader(data_test, input_nodes=("transaction", test_idx), num_neighbors=NUM_NEIGHBORS, batch_size=args.batch_size, shuffle=False)

    model = GriffinLeakageAwareGNN(len(DEFAULT_FEATURE_COLS), addr_x.shape[1], args.hidden_dim, args.heads, args.dropout).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_counts = np.bincount(y[train_mask], minlength=2)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * class_counts + 1e-8), dtype=torch.float32, device=device)
    criterion = AsymmetricFocalLoss(class_weights=class_weights, gamma_pos=2.0, gamma_neg=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=args.warmup),
            CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6),
        ],
        milestones=[args.warmup],
    )

    top_checkpoints = []
    score_window = deque(maxlen=3)
    best_window_score = -1.0
    wait = args.patience
    training_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            out = model(batch)
            bs = batch["transaction"].batch_size
            loss = criterion(out[:bs], batch["transaction"].y[:bs]) / args.accum_steps
            loss.backward()
            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            losses.append(float(loss.item() * args.accum_steps))
            del batch, out, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        val_probs, val_true = predict_loader(model, val_loader, device)
        val_ap = average_precision_score(val_true, val_probs)
        _, val_f1 = best_threshold_fbeta(val_probs, val_true, beta=1.0)
        combined = val_ap * 0.6 + val_f1 * 0.4
        score_window.append(combined)
        window_avg = float(np.mean(score_window))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val_auc_pr": float(val_ap), "val_f1": float(val_f1), "combined": float(combined), "lr": optimizer.param_groups[0]["lr"]}
        training_log.append(row)
        print(row)

        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top_checkpoints.append((combined, val_ap, val_f1, state))
        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
        top_checkpoints = top_checkpoints[:3]

        if window_avg > best_window_score + 1e-5:
            best_window_score = window_avg
            wait = args.patience
        else:
            wait -= 1
            if wait <= 0:
                print(f"Early stopping at epoch {epoch}")
                break

    best_ckpt = top_checkpoints[0]
    model.load_state_dict(best_ckpt[3])
    model.to(device)

    val_probs, val_true = predict_loader(model, val_loader, device)
    theta_f1, score_f1 = best_threshold_fbeta(val_probs, val_true, beta=1.0)
    theta_fb05, score_fb05 = best_threshold_fbeta(val_probs, val_true, beta=0.5)
    theta_fb2, score_fb2 = best_threshold_fbeta(val_probs, val_true, beta=2.0)

    test_probs, test_true = predict_loader(model, test_loader, device)
    theta_list = sorted(set([0.50, 0.70, 0.90, 0.95, 0.97, 0.99, round(theta_f1, 4), round(theta_fb05, 4), round(theta_fb2, 4)]))
    results = [{"threshold": float(th), **metrics_at_threshold(test_probs, test_true, th)} for th in theta_list]
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(out_dir / "test_metrics_by_threshold.csv", index=False)

    theta_op = round(theta_f1, 4)
    pred = (test_probs >= theta_op).astype(int)
    report = classification_report(test_true, pred, digits=4, zero_division=0, output_dict=True)
    with open(out_dir / "classification_report_f1_threshold.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    test_conn_mask = connectivity_mask[test_mask]
    exposure_rows = []
    for label, mask in [("connected", test_conn_mask), ("isolated", ~test_conn_mask)]:
        if mask.sum() > 0 and test_true[mask].sum() > 0:
            exposure_rows.append({"subset": label, "n": int(mask.sum()), "fraud": int(test_true[mask].sum()), **metrics_at_threshold(test_probs[mask], test_true[mask], theta_op)})
    pd.DataFrame(exposure_rows).to_csv(out_dir / "connectivity_decomposition.csv", index=False)

    boot_summary = bootstrap_ci(test_probs, test_true, theta_op, args.seed, args.bootstrap_iters)
    with open(out_dir / "bootstrap_ci.json", "w", encoding="utf-8") as f:
        json.dump(boot_summary, f, indent=2)

    np.save(out_dir / "val_probs.npy", val_probs)
    np.save(out_dir / "val_true.npy", val_true)
    np.save(out_dir / "test_probs.npy", test_probs)
    np.save(out_dir / "test_true.npy", test_true)
    pd.DataFrame(training_log).to_csv(out_dir / "training_log.csv", index=False)

    torch.save({
        "experiment": "leakage_aware_griffin_gnn",
        "article": "When Graph Connectivity Becomes Leakage",
        "snapshot_order": SNAPSHOT_ORDER,
        "train_snapshots": TRAIN_SNAPS,
        "validation_snapshots": VAL_SNAPS,
        "test_snapshots": TEST_SNAPS,
        "feature_cols": DEFAULT_FEATURE_COLS,
        "hidden_dim": args.hidden_dim,
        "theta_f1": float(theta_op),
        "theta_fb05": float(round(theta_fb05, 4)),
        "theta_fb2": float(round(theta_fb2, 4)),
        "best_val_combined": float(best_ckpt[0]),
        "best_val_auc_pr": float(best_ckpt[1]),
        "best_val_f1": float(best_ckpt[2]),
        "model_state_dict": model.state_dict(),
        "top3_checkpoints_scores": [(float(c[0]), float(c[1]), float(c[2])) for c in top_checkpoints],
    }, out_dir / "leakage_aware_griffin_gnn_checkpoint.pt")

    print("\nSaved outputs to:", out_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
