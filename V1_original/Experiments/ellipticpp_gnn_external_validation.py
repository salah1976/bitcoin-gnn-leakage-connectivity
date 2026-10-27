import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class MLP_Elliptic(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class GraphSAGE_Elliptic(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.tab = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, 2), nn.Softmax(dim=-1))
        self.clf = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.gelu(self.norm1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h_gnn = F.gelu(self.norm2(h))
        h_tab = self.tab(x)
        g = self.gate(torch.cat([h_gnn, h_tab], dim=1))
        h_fused = g[:, 0:1] * h_gnn + g[:, 1:2] * h_tab
        return self.clf(h_fused)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(data_dir):
    feat_df = pd.read_csv(os.path.join(data_dir, "txs_features.csv"))
    class_df = pd.read_csv(os.path.join(data_dir, "txs_classes.csv"))
    edges_df = pd.read_csv(os.path.join(data_dir, "txs_edgelist.csv"))

    df = feat_df.merge(class_df, on="txId", how="left")
    df = df[df["class"].isin([1, 2])].copy().reset_index(drop=True)
    df["label"] = (df["class"] == 1).astype(int)

    local_feat_cols = [c for c in df.columns if c.startswith("Local_feature_")]
    btc_feat_cols = [c for c in df.columns if "in_BTC" in c or "out_BTC" in c]
    feat_cols = local_feat_cols + btc_feat_cols

    return df, edges_df, feat_cols


def build_edge_index(edges_sub, txid_to_idx):
    src = edges_sub["txId1"].map(txid_to_idx)
    dst = edges_sub["txId2"].map(txid_to_idx)
    mask = src.notna() & dst.notna()
    src = src[mask].astype(int).values
    dst = dst[mask].astype(int).values

    if len(src) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    src_bi = np.concatenate([src, dst])
    dst_bi = np.concatenate([dst, src])
    return torch.tensor(np.vstack([src_bi, dst_bi]), dtype=torch.long)


def best_threshold_from_val(y_val, p_val):
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_i = int(np.argmax(f1s[:-1]))
    return float(thresholds[best_i]), float(f1s[best_i])


def compute_metrics(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "AUC_ROC": roc_auc_score(y_true, probs),
        "AUC_PR": average_precision_score(y_true, probs),
        "F1": f1_score(y_true, pred),
        "Precision": np.sum((pred == 1) & (y_true == 1)) / max(np.sum(pred == 1), 1),
        "Recall": np.sum((pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1),
        "threshold": threshold,
    }


def get_class_weight(y_tensor, train_idx, device):
    y_train = y_tensor[train_idx].cpu().numpy()
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    return torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)


def print_result(result):
    print(f"\n{result['protocol']}")
    print(f"AUC-ROC   = {result['AUC_ROC']:.4f}")
    print(f"AUC-PR    = {result['AUC_PR']:.4f}")
    print(f"F1        = {result['F1']:.4f}")
    print(f"Precision = {result['Precision']:.4f}")
    print(f"Recall    = {result['Recall']:.4f}")
    print(f"Threshold = {result['threshold']:.4f}")
    print(f"Val AP    = {result['best_val_AP']:.4f}")


def train_mlp(protocol_name, x_tensor, y_tensor, train_idx, val_idx, test_idx, in_dim, device, epochs, lr):
    model = MLP_Elliptic(in_dim).to(device)
    x_gpu = x_tensor.to(device)
    y_gpu = y_tensor.to(device)
    weights = get_class_weight(y_tensor, train_idx, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_ap = -1.0
    best_state = None
    bad = 0
    patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_gpu)
        loss = F.cross_entropy(logits[train_idx], y_gpu[train_idx], weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                probs_val = F.softmax(model(x_gpu)[val_idx], dim=1)[:, 1].cpu().numpy()
                val_ap = average_precision_score(y_tensor[val_idx].numpy(), probs_val)
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    with torch.no_grad():
        probs_val = F.softmax(model(x_gpu)[val_idx], dim=1)[:, 1].cpu().numpy()
        probs_test = F.softmax(model(x_gpu)[test_idx], dim=1)[:, 1].cpu().numpy()

    threshold, val_f1 = best_threshold_from_val(y_tensor[val_idx].numpy(), probs_val)
    metrics = compute_metrics(y_tensor[test_idx].numpy(), probs_test, threshold)
    metrics.update({
        "protocol": protocol_name,
        "best_val_AP": best_val_ap,
        "val_F1_at_threshold": val_f1,
        "removed_edges": 0,
        "removed_edges_pct": 0.0,
    })
    print_result(metrics)
    return metrics


def train_gnn(protocol_name, edge_index, removed_edges, removed_edges_pct, x_tensor, y_tensor, train_idx, val_idx, test_idx, in_dim, device, epochs, lr):
    model = GraphSAGE_Elliptic(in_dim).to(device)
    x_gpu = x_tensor.to(device)
    y_gpu = y_tensor.to(device)
    ei_gpu = edge_index.to(device)
    weights = get_class_weight(y_tensor, train_idx, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_ap = -1.0
    best_state = None
    bad = 0
    patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_gpu, ei_gpu)
        loss = F.cross_entropy(logits[train_idx], y_gpu[train_idx], weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                probs_val = F.softmax(model(x_gpu, ei_gpu)[val_idx], dim=1)[:, 1].cpu().numpy()
                val_ap = average_precision_score(y_tensor[val_idx].numpy(), probs_val)
            if val_ap > best_val_ap:
                best_val_ap = val_ap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device).eval()

    with torch.no_grad():
        probs_val = F.softmax(model(x_gpu, ei_gpu)[val_idx], dim=1)[:, 1].cpu().numpy()
        probs_test = F.softmax(model(x_gpu, ei_gpu)[test_idx], dim=1)[:, 1].cpu().numpy()

    threshold, val_f1 = best_threshold_from_val(y_tensor[val_idx].numpy(), probs_val)
    metrics = compute_metrics(y_tensor[test_idx].numpy(), probs_test, threshold)
    metrics.update({
        "protocol": protocol_name,
        "best_val_AP": best_val_ap,
        "val_F1_at_threshold": val_f1,
        "removed_edges": removed_edges,
        "removed_edges_pct": removed_edges_pct,
    })
    print_result(metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", default="results/elliptic_gnn_corrected")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bitfraud-standard-ap", type=float, default=0.8372)
    parser.add_argument("--bitfraud-isolated-ap", type=float, default=0.0581)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df, edges_df, feat_cols = load_data(args.data_dir)
    print(f"Labeled: {len(df):,}")
    print(f"Fraud rate: {df['label'].mean() * 100:.2f}%")
    print(f"Features: {len(feat_cols)}")

    train_ts = list(range(1, 30))
    val_ts = list(range(30, 35))
    test_ts = list(range(35, 50))

    train_df = df[df["Time step"].isin(train_ts)].copy()
    val_df = df[df["Time step"].isin(val_ts)].copy()
    test_df = df[df["Time step"].isin(test_ts)].copy()

    print("\nTemporal split:")
    print(f"Train: {len(train_df):,} | fraud={train_df['label'].sum():,} | rate={train_df['label'].mean() * 100:.2f}%")
    print(f"Val  : {len(val_df):,} | fraud={val_df['label'].sum():,} | rate={val_df['label'].mean() * 100:.2f}%")
    print(f"Test : {len(test_df):,} | fraud={test_df['label'].sum():,} | rate={test_df['label'].mean() * 100:.2f}%")

    txid_to_idx = {txid: i for i, txid in enumerate(df["txId"].values)}
    train_idx = torch.tensor([txid_to_idx[t] for t in train_df["txId"]], dtype=torch.long)
    val_idx = torch.tensor([txid_to_idx[t] for t in val_df["txId"]], dtype=torch.long)
    test_idx = torch.tensor([txid_to_idx[t] for t in test_df["txId"]], dtype=torch.long)

    train_ids = set(train_df["txId"].values)
    val_ids = set(val_df["txId"].values)
    test_ids = set(test_df["txId"].values)
    hist_ids = train_ids | val_ids

    x_raw = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(x_raw[train_idx.numpy()])
    x_tensor = torch.tensor(scaler.transform(x_raw).astype(np.float32), dtype=torch.float32)
    y_tensor = torch.tensor(df["label"].values.astype(np.int64), dtype=torch.long)

    edges_standard = edges_df.copy()
    edge_index_standard = build_edge_index(edges_standard, txid_to_idx)
    standard_directed_edges = edge_index_standard.shape[1]

    protocol_edges = {
        "GNN Standard - all edges": edges_standard,
        "GNN No test-test edges": edges_df[~(edges_df["txId1"].isin(test_ids) & edges_df["txId2"].isin(test_ids))].copy(),
        "GNN No history-test edges": edges_df[~(
            (edges_df["txId1"].isin(hist_ids) & edges_df["txId2"].isin(test_ids)) |
            (edges_df["txId1"].isin(test_ids) & edges_df["txId2"].isin(hist_ids))
        )].copy(),
        "GNN Fully isolated test": edges_df[~(edges_df["txId1"].isin(test_ids) | edges_df["txId2"].isin(test_ids))].copy(),
    }

    results = []
    results.append(train_mlp(
        "MLP baseline - no graph", x_tensor, y_tensor, train_idx, val_idx, test_idx,
        len(feat_cols), device, args.epochs, args.lr,
    ))

    for name, edge_df in protocol_edges.items():
        edge_index = build_edge_index(edge_df, txid_to_idx)
        removed = standard_directed_edges - edge_index.shape[1]
        removed_pct = 100 * removed / standard_directed_edges if standard_directed_edges else 0.0
        print(f"\n{name}: directed_edges={edge_index.shape[1]:,}, removed={removed:,} ({removed_pct:.1f}%)")
        results.append(train_gnn(
            name, edge_index, removed, removed_pct, x_tensor, y_tensor,
            train_idx, val_idx, test_idx, len(feat_cols), device, args.epochs, args.lr,
        ))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_results = pd.DataFrame(results)
    result_cols = [
        "protocol", "AUC_ROC", "AUC_PR", "F1", "Precision", "Recall",
        "threshold", "best_val_AP", "val_F1_at_threshold", "removed_edges", "removed_edges_pct",
    ]
    df_results = df_results[result_cols]
    results_path = os.path.join(args.out_dir, "elliptic_external_validation_corrected.csv")
    df_results.to_csv(results_path, index=False)

    standard_ap = df_results.loc[df_results["protocol"] == "GNN Standard - all edges", "AUC_PR"].iloc[0]
    isolated_ap = df_results.loc[df_results["protocol"] == "GNN Fully isolated test", "AUC_PR"].iloc[0]
    elliptic_delta = standard_ap - isolated_ap
    elliptic_rel_drop = 100 * elliptic_delta / standard_ap if standard_ap > 0 else 0.0

    bitfraud_delta = args.bitfraud_standard_ap - args.bitfraud_isolated_ap
    bitfraud_rel_drop = 100 * bitfraud_delta / args.bitfraud_standard_ap if args.bitfraud_standard_ap > 0 else 0.0

    comparison = pd.DataFrame({
        "dataset": ["BitFraud", "Elliptic++"],
        "standard_AUC_PR": [args.bitfraud_standard_ap, standard_ap],
        "fully_isolated_AUC_PR": [args.bitfraud_isolated_ap, isolated_ap],
        "delta_AUC_PR": [bitfraud_delta, elliptic_delta],
        "relative_drop_pct": [bitfraud_rel_drop, elliptic_rel_drop],
    })
    comparison_path = os.path.join(args.out_dir, "comparison_bitfraud_elliptic_corrected.csv")
    comparison.to_csv(comparison_path, index=False)

    print("\nFINAL RESULTS — Elliptic++ External Validation")
    print(df_results.to_string(index=False))
    print("\nBITFRAUD vs ELLIPTIC++")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    main()
