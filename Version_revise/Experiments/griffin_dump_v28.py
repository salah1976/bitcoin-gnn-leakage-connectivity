# ============================================================
# GriffinGNN v28 — GPU FIRST, CPU FALLBACK
# Détecte automatiquement GPU/CPU et adapte les hyperparamètres
# GPU : hidden=192, neighbors=[20,15], batch=512  → ~5min/epoch
# CPU : hidden=128, neighbors=[5,3],  batch=4096  → ~8min/epoch
# ============================================================


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc, random, shutil, sqlite3, time
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve,
)
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv

# ============================================================
# 0) DEVICE DETECTION — GPU FIRST
# ============================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    GPU_MODE = True
    print("="*50)
    print("✅  GPU détecté — mode GPU activé")
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print("="*50)
else:
    device = torch.device("cpu")
    GPU_MODE = False
    torch.set_num_threads(os.cpu_count())
    print("="*50)
    print("⚠️  Aucun GPU — mode CPU activé")
    print(f"   Threads CPU: {torch.get_num_threads()}")
    print("   💡 Pour activer GPU : Runtime → Change runtime type → T4 GPU")
    print("="*50)

# ============================================================
# HYPERPARAMÈTRES ADAPTATIFS GPU vs CPU
# ============================================================

if GPU_MODE:
    HIDDEN      = 192
    BATCH_SIZE  = 512
    NUM_NEIGHBORS = {
        ("address","input_to_tx","transaction"):  [20, 15],
        ("address","output_to_tx","transaction"): [20, 15],
        ("transaction","tx_to_input","address"):  [15, 10],
        ("transaction","tx_to_output","address"): [15, 10],
    }
    EPOCHS      = 80
    PATIENCE    = 25
    WARMUP      = 5
    LR_MAX      = 1e-4
    LR_START    = 5e-6
    ACCUM_STEPS = 2
    N_WORKERS   = 0
    BOOT_ITERS  = 1000
else:
    HIDDEN      = 128
    BATCH_SIZE  = 4096
    NUM_NEIGHBORS = {
        ("address","input_to_tx","transaction"):  [5, 3],
        ("address","output_to_tx","transaction"): [5, 3],
        ("transaction","tx_to_input","address"):  [3, 2],
        ("transaction","tx_to_output","address"): [3, 2],
    }
    EPOCHS      = 30
    PATIENCE    = 10
    WARMUP      = 2
    LR_MAX      = 3e-4
    LR_START    = 2e-5
    ACCUM_STEPS = 1
    N_WORKERS   = 2
    BOOT_ITERS  = 500

print(f"\nConfig : hidden={HIDDEN} | batch={BATCH_SIZE} | "
      f"neighbors={NUM_NEIGHBORS[('address','input_to_tx','transaction')]} | "
      f"epochs={EPOCHS}")

SEED = 42                     # <-- 42, 101, 202, 303, 404
FEATURE_MODE = "full"         # <-- "full" | "no_fraud" (retire fraud_ratio et log1p(total_fraud))
RUN_SWEEP = False             # <-- True pour lancer aussi le balayage lambda complet (long)
LOAD_PT = None                # <-- chemin d'un .pt existant (ex: ".../griffin_v26_seed123_gpu.pt") = INFERENCE SEULE, sans reentrainement
RUN_TAG = f"{FEATURE_MODE}_seed{SEED}" + ("_infer" if LOAD_PT else "")
if LOAD_PT is None:
    raise SystemExit("STOP : griffin_dump_v28.py n'entraine jamais. Lance-le uniquement avec run_queue_dump.py (qui fournit LOAD_PT = checkpoint). Aucun fichier n'a ete modifie.")
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if GPU_MODE: torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# PATHS
# ============================================================

from google.colab import drive
drive.mount("/content/drive", force_remount=True)

ROOT       = "/content/drive/MyDrive/bitcoin_snapshots_project"
DRIVE_DB   = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
LOCAL_DB   = "/content/all_snapshots_v21.db"
LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
OUT_DIR    = f"{ROOT}/05_results/runs/{RUN_TAG}"   # un dossier par run: pas de reutilisation de vieux JSON
os.makedirs(OUT_DIR, exist_ok=True)

MODE_TAG   = "gpu" if GPU_MODE else "cpu"
SAVE_PATH = f"{OUT_DIR}/griffin_v26_seed{SEED}_{MODE_TAG}.pt"

shutil.copy2(DRIVE_DB, LOCAL_DB)
print(f"\nDB copiée → {LOCAL_DB}")

SNAPSHOT_ORDER = ["D1","D2","D3","D4","D5","D6"]
TRAIN_SNAPS    = ["D1","D2","D3"]
VAL_SNAPS      = ["D4"]
TEST_SNAPS     = ["D5","D6"]

feature_cols = [
    "input_count","output_count","input_addr_count",
    "coinbase_flag","has_witness","script_type_encoded",
    "input_addr_concentration","io_count_ratio","tx_weight",
    "avg_input_value","total_input_scaled","log_output_value",
    "fee_ratio","prev_addr_seen_ratio","prev_addr_seen_count",
]

# ============================================================
# 1) LOAD DATA
# ============================================================

print("\nLoading SQLite edges...")
conn       = sqlite3.connect(LOCAL_DB)
inputs_df  = pd.read_sql_query("SELECT tx_hash, input_address  FROM tx_inputs",  conn)
outputs_df = pd.read_sql_query("SELECT tx_hash, output_address FROM tx_outputs", conn)
conn.close()
print(f"  in-edges: {len(inputs_df):,} | out-edges: {len(outputs_df):,}")

labels_df    = pd.read_csv(LABELS_CSV)
keep_cols    = ["tx_hash","snapshot_id","label_final"] + feature_cols
transactions = labels_df[keep_cols].copy()
transactions = transactions[
    transactions["snapshot_id"].isin(SNAPSHOT_ORDER)].copy()
transactions["snapshot_rank"] = transactions["snapshot_id"].map(
    {s:i for i,s in enumerate(SNAPSHOT_ORDER)})
transactions = transactions.sort_values(
    ["snapshot_rank","tx_hash"]).reset_index(drop=True)

train_mask = transactions["snapshot_id"].isin(TRAIN_SNAPS).values
val_mask   = transactions["snapshot_id"].isin(VAL_SNAPS).values
test_mask  = transactions["snapshot_id"].isin(TEST_SNAPS).values
y          = transactions["label_final"].astype(int).values
snap_rank  = transactions["snapshot_rank"].values

print(f"  Transactions: {len(transactions):,}")
print(transactions.groupby("snapshot_id")["label_final"]
      .agg(["count","sum","mean"]).to_string())

print("\nSplit:")
for nm,mk in [("Train",train_mask),("Val",val_mask),("Test",test_mask)]:
    yy=y[mk]
    print(f"  {nm}: {mk.sum():>7,} | fraud={yy.sum():>5,} | rate={yy.mean():.4%}")

# Scale
X_raw = (transactions[feature_cols]
         .replace([np.inf,-np.inf],np.nan).fillna(0)
         .values.astype(np.float32))
scaler   = StandardScaler()
X_scaled = np.zeros_like(X_raw, dtype=np.float32)
X_scaled[train_mask] = scaler.fit_transform(X_raw[train_mask])
X_scaled[val_mask]   = scaler.transform(X_raw[val_mask])
X_scaled[test_mask]  = scaler.transform(X_raw[test_mask])

# ============================================================
# 2) NODE INDICES
# ============================================================

tx_hashes      = transactions["tx_hash"].values
tx_hash_to_idx = {h:i for i,h in enumerate(tx_hashes)}
tx_hash_set    = set(tx_hashes)

inputs_df  = inputs_df[inputs_df["tx_hash"].isin(tx_hash_set)].dropna().copy()
outputs_df = outputs_df[outputs_df["tx_hash"].isin(tx_hash_set)].dropna().copy()

inputs_df["tx_idx"]        = inputs_df["tx_hash"].map(tx_hash_to_idx)
outputs_df["tx_idx"]       = outputs_df["tx_hash"].map(tx_hash_to_idx)
inputs_df["input_address"] = inputs_df["input_address"].astype(str)
outputs_df["output_address"]= outputs_df["output_address"].astype(str)

all_addresses = pd.concat([
    inputs_df["input_address"],
    outputs_df["output_address"]
]).dropna().unique()
addr_to_idx = {a:i for i,a in enumerate(all_addresses)}

inputs_df["addr_idx"]  = inputs_df["input_address"].map(addr_to_idx)
outputs_df["addr_idx"] = outputs_df["output_address"].map(addr_to_idx)
inputs_df  = inputs_df.dropna(subset=["tx_idx","addr_idx"]).copy()
outputs_df = outputs_df.dropna(subset=["tx_idx","addr_idx"]).copy()
inputs_df[["tx_idx","addr_idx"]]  = inputs_df[["tx_idx","addr_idx"]].astype(np.int64)
outputs_df[["tx_idx","addr_idx"]] = outputs_df[["tx_idx","addr_idx"]].astype(np.int64)

num_tx   = len(transactions)
num_addr = len(all_addresses)
print(f"\nGraph: {num_tx:,} tx | {num_addr:,} addr")

# ============================================================
# 3) ADDRESS FEATURES — 11 dimensions
# ============================================================

def build_address_features(history_mask):
    hist_idx   = set(np.where(history_mask)[0].tolist())
    hist_fraud = set(np.where(history_mask & (y==1))[0].tolist())
    inp = inputs_df[inputs_df["tx_idx"].isin(hist_idx)].copy()
    out = outputs_df[outputs_df["tx_idx"].isin(hist_idx)].copy()
    rank_map      = dict(zip(np.where(history_mask)[0], snap_rank[history_mask]))
    inp["weight"] = (inp["tx_idx"].map(rank_map).fillna(0)+1).astype(np.float32)
    out["weight"] = (out["tx_idx"].map(rank_map).fillna(0)+1).astype(np.float32)
    in_deg   = np.bincount(inp["addr_idx"].values, minlength=num_addr)
    out_deg  = np.bincount(out["addr_idx"].values, minlength=num_addr)
    tot_deg  = in_deg + out_deg
    act_both = ((in_deg>0)&(out_deg>0)).astype(np.float32)
    io_ratio = (in_deg+1.)/(out_deg+1.)
    r_in  = np.bincount(inp["addr_idx"].values,
                        weights=inp["weight"].values, minlength=num_addr)
    r_out = np.bincount(out["addr_idx"].values,
                        weights=out["weight"].values, minlength=num_addr)
    recency  = r_in + r_out
    inp_f    = inputs_df[inputs_df["tx_idx"].isin(hist_fraud)]
    out_f    = outputs_df[outputs_df["tx_idx"].isin(hist_fraud)]
    tot_fr   = (np.bincount(inp_f["addr_idx"].values, minlength=num_addr)+
                np.bincount(out_f["addr_idx"].values, minlength=num_addr))
    fr_ratio = tot_fr/(tot_deg+1e-8)
    atx      = pd.concat([inp[["addr_idx","tx_idx"]], out[["addr_idx","tx_idx"]]])
    uniq_tx  = atx.groupby("addr_idx")["tx_idx"].nunique().reindex(
        range(num_addr),fill_value=0).values
    nb_lnk   = atx.groupby("addr_idx")["tx_idx"].count().reindex(
        range(num_addr),fill_value=0).values
    act_r    = (tot_deg+1.)/(uniq_tx+1.)
    return np.vstack([
        np.log1p(in_deg),    np.log1p(out_deg),   np.log1p(tot_deg),
        np.log1p(io_ratio),  act_both,
        np.log1p(uniq_tx),   np.log1p(nb_lnk),    np.log1p(act_r),
        fr_ratio,            np.log1p(tot_fr),     np.log1p(recency),
    ]).T.astype(np.float32)

print("\nBuilding 11-dim address features...")
addr_x_raw     = build_address_features(train_mask)
if FEATURE_MODE == "no_fraud":
    addr_x_raw[:, [8, 9]] = 0.0   # colonne 8 = fraud_ratio, 9 = log1p(total_fraud)
print(f"FEATURE_MODE={FEATURE_MODE} | SEED={SEED} | OUT_DIR={OUT_DIR}")
train_tx_set   = set(np.where(train_mask)[0].tolist())
train_addr_idx = np.unique(pd.concat([
    inputs_df[inputs_df["tx_idx"].isin(train_tx_set)]["addr_idx"],
    outputs_df[outputs_df["tx_idx"].isin(train_tx_set)]["addr_idx"],
]).values)
addr_scaler = StandardScaler()
addr_scaler.fit(addr_x_raw[train_addr_idx])
addr_x = addr_scaler.transform(addr_x_raw).astype(np.float32)
print(f"Address feature dim: {addr_x.shape[1]} ✓")

# ============================================================
# 4) GRAPH CONSTRUCTION — Known-Address (λ=24.27%)
# ============================================================

def make_hetero(inp, out):
    data = HeteroData()
    data["transaction"].x = torch.tensor(X_scaled, dtype=torch.float32)
    data["transaction"].y = torch.tensor(y,        dtype=torch.long)
    data["address"].x     = torch.tensor(addr_x,   dtype=torch.float32)
    data["address","input_to_tx","transaction"].edge_index = torch.tensor(
        np.vstack([inp["addr_idx"].values, inp["tx_idx"].values]),dtype=torch.long)
    data["address","output_to_tx","transaction"].edge_index = torch.tensor(
        np.vstack([out["addr_idx"].values, out["tx_idx"].values]),dtype=torch.long)
    data["transaction","tx_to_input","address"].edge_index = torch.tensor(
        np.vstack([inp["tx_idx"].values, inp["addr_idx"].values]),dtype=torch.long)
    data["transaction","tx_to_output","address"].edge_index = torch.tensor(
        np.vstack([out["tx_idx"].values, out["addr_idx"].values]),dtype=torch.long)
    return data

def build_split(mask):
    allowed = set(np.where(mask)[0].tolist())
    return make_hetero(
        inputs_df[inputs_df["tx_idx"].isin(allowed)],
        outputs_df[outputs_df["tx_idx"].isin(allowed)])

def build_test_known():
    tv_ids = set(np.where(train_mask|val_mask)[0].tolist())
    te_ids = set(np.where(test_mask)[0].tolist())
    inp_tv = inputs_df[inputs_df["tx_idx"].isin(tv_ids)]
    out_tv = outputs_df[outputs_df["tx_idx"].isin(tv_ids)]
    known  = set(inp_tv["addr_idx"].values)|set(out_tv["addr_idx"].values)
    inp_te = inputs_df[inputs_df["tx_idx"].isin(te_ids)]
    out_te = outputs_df[outputs_df["tx_idx"].isin(te_ids)]
    ik = inp_te[inp_te["addr_idx"].isin(known)]
    ok = out_te[out_te["addr_idx"].isin(known)]
    te = pd.concat([ik[["tx_idx","addr_idx"]],ok[["tx_idx","addr_idx"]]])
    fc = te.groupby("addr_idx")["tx_idx"].nunique()
    lk = te[te["addr_idx"].isin(fc[fc>1].index)]["tx_idx"].nunique()
    print(f"  Known-Address λ = {lk/len(te_ids):.4%}  (target ~24.27%)")
    return make_hetero(
        pd.concat([inp_tv,ik],ignore_index=True),
        pd.concat([out_tv,ok],ignore_index=True))

def build_test_topk(K):
    """Filtrage Top-K des arêtes partagées test-to-test (inférence seule)."""
    tv_ids = set(np.where(train_mask|val_mask)[0].tolist())
    te_ids = set(np.where(test_mask)[0].tolist())
    inp_tv = inputs_df[inputs_df["tx_idx"].isin(tv_ids)]
    out_tv = outputs_df[outputs_df["tx_idx"].isin(tv_ids)]
    known  = set(inp_tv["addr_idx"].values)|set(out_tv["addr_idx"].values)
    inp_te = inputs_df[inputs_df["tx_idx"].isin(te_ids)]
    out_te = outputs_df[outputs_df["tx_idx"].isin(te_ids)]
    ik = inp_te[inp_te["addr_idx"].isin(known)].copy()
    ok = out_te[out_te["addr_idx"].isin(known)].copy()

    tv_addr_per_tx = pd.concat([inp_tv[["tx_idx","addr_idx"]], out_tv[["tx_idx","addr_idx"]]])
    exposure_score = tv_addr_per_tx.groupby("tx_idx")["addr_idx"].nunique()

    te_edges = pd.concat([ik[["tx_idx","addr_idx"]], ok[["tx_idx","addr_idx"]]]).drop_duplicates()
    addr_counts  = te_edges.groupby("addr_idx")["tx_idx"].nunique()
    shared_addrs = set(addr_counts[addr_counts > 1].index)

    te_edges["score"] = te_edges["tx_idx"].map(exposure_score).fillna(0)
    te_edges["rank"]  = te_edges.groupby("addr_idx")["score"].rank(
        ascending=False, method="first")


    keep_pairs_df = te_edges[
        (~te_edges["addr_idx"].isin(shared_addrs)) | (te_edges["rank"]<=K)
    ][["tx_idx","addr_idx"]].copy()
    ik_f = ik.merge(keep_pairs_df, on=["tx_idx","addr_idx"], how="inner")
    ok_f = ok.merge(keep_pairs_df, on=["tx_idx","addr_idx"], how="inner")

    te_f = pd.concat([ik_f[["tx_idx","addr_idx"]], ok_f[["tx_idx","addr_idx"]]])
    fc = te_f.groupby("addr_idx")["tx_idx"].nunique()
    connected_tx_ids = set(te_f[te_f["addr_idx"].isin(fc[fc>1].index)]["tx_idx"].values)
    lk = len(connected_tx_ids)
    lam = lk/len(te_ids)
    print(f"  Top-K={K}: λ = {lam:.4%}")

    conn_mask_topk = np.zeros(num_tx, dtype=bool)
    for i in connected_tx_ids: conn_mask_topk[i] = True


    result = make_hetero(
        pd.concat([inp_tv,ik_f],ignore_index=True),
        pd.concat([out_tv,ok_f],ignore_index=True))
    del inp_te, out_te, ik, ok, tv_addr_per_tx, exposure_score
    del te_edges, addr_counts, shared_addrs, keep_pairs_df, te_f, fc
    gc.collect()
    return result, lam, conn_mask_topk

print("\nBuilding graphs...")
data_train = build_split(train_mask)
data_val   = build_split(train_mask|val_mask)
data_test  = build_test_known()

# Connectivity decomposition
tv_ids = set(np.where(train_mask|val_mask)[0].tolist())
te_ids = set(np.where(test_mask)[0].tolist())
inp_tv = inputs_df[inputs_df["tx_idx"].isin(tv_ids)]
out_tv = outputs_df[outputs_df["tx_idx"].isin(tv_ids)]
known  = set(inp_tv["addr_idx"].values)|set(out_tv["addr_idx"].values)
inp_te = inputs_df[inputs_df["tx_idx"].isin(te_ids)]
out_te = outputs_df[outputs_df["tx_idx"].isin(te_ids)]
conn_tx= (set(inp_te[inp_te["addr_idx"].isin(known)]["tx_idx"].values)|
          set(out_te[out_te["addr_idx"].isin(known)]["tx_idx"].values))
conn_mask = np.zeros(num_tx, dtype=bool)
for i in conn_tx: conn_mask[i]=True

conn_nodes = torch.tensor(np.where(test_mask & conn_mask)[0],  dtype=torch.long)
isol_nodes = torch.tensor(np.where(test_mask & ~conn_mask)[0], dtype=torch.long)
print(f"  Connected: {conn_mask[test_mask].sum():,} ({conn_mask[test_mask].mean():.2%})"
      f" | Isolated: {(~conn_mask[test_mask]).sum():,}")
print(f"  Fraud connected: {y[test_mask & conn_mask].mean():.4%} "
      f"| Fraud isolated: {y[test_mask & ~conn_mask].mean():.4%}")

train_idx_t = torch.tensor(np.where(train_mask)[0], dtype=torch.long)
val_idx_t   = torch.tensor(np.where(val_mask)[0],   dtype=torch.long)
test_idx_t  = torch.tensor(np.where(test_mask)[0],  dtype=torch.long)

# ============================================================
# 5) LOADERS
# ============================================================

def make_loader(data, nodes, shuffle=False):
    return NeighborLoader(data,
        input_nodes=("transaction", nodes),
        num_neighbors=NUM_NEIGHBORS,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=N_WORKERS)

train_loader = make_loader(data_train, train_idx_t, shuffle=True)
val_loader   = make_loader(data_val,   val_idx_t)
test_loader  = make_loader(data_test,  test_idx_t)
conn_loader  = make_loader(data_test,  conn_nodes)
isol_loader  = make_loader(data_test,  isol_nodes)

print(f"\nLoaders: {len(train_idx_t)//BATCH_SIZE} batches/epoch | "
      f"batch={BATCH_SIZE} | workers={N_WORKERS}")

# ============================================================
# 6) MODEL — GriffinGNN full architecture
# ============================================================

class GriffinGNN(nn.Module):
    def __init__(self, tx_in, addr_in, hidden=192, heads=4, drop=0.15):
        super().__init__()
        assert hidden % heads == 0
        hd = hidden // heads; self.drop = drop

        # Projections
        self.tx_proj   = nn.Sequential(
            nn.Linear(tx_in,   hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(drop))
        self.addr_proj = nn.Sequential(
            nn.Linear(addr_in, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(drop))

        # Layer 1 — GAT (4 edge types)
        self.conv1 = HeteroConv({
            ("address","input_to_tx","transaction"):
                GATConv((hidden,hidden),hd,heads=heads,dropout=drop,add_self_loops=False),
            ("address","output_to_tx","transaction"):
                GATConv((hidden,hidden),hd,heads=heads,dropout=drop,add_self_loops=False),
            ("transaction","tx_to_input","address"):
                GATConv((hidden,hidden),hd,heads=heads,dropout=drop,add_self_loops=False),
            ("transaction","tx_to_output","address"):
                GATConv((hidden,hidden),hd,heads=heads,dropout=drop,add_self_loops=False),
        }, aggr="sum")
        self.n_tx1 = nn.LayerNorm(hidden)
        self.n_ad1 = nn.LayerNorm(hidden)

        # Layer 2 — GraphSAGE (4 edge types)
        self.conv2 = HeteroConv({
            ("address","input_to_tx","transaction"):  SAGEConv((hidden,hidden),hidden),
            ("address","output_to_tx","transaction"): SAGEConv((hidden,hidden),hidden),
            ("transaction","tx_to_input","address"):  SAGEConv((hidden,hidden),hidden),
            ("transaction","tx_to_output","address"): SAGEConv((hidden,hidden),hidden),
        }, aggr="sum")
        self.n_tx2 = nn.LayerNorm(hidden)

        # Tabular branch — 4-layer residual MLP
        self.t1=nn.Linear(tx_in,hidden);  self.b1=nn.BatchNorm1d(hidden)
        self.t2=nn.Linear(hidden,hidden);  self.b2=nn.BatchNorm1d(hidden)
        self.t3=nn.Linear(hidden,hidden);  self.b3=nn.BatchNorm1d(hidden)
        self.t4=nn.Linear(hidden,hidden);  self.b4=nn.BatchNorm1d(hidden)
        self.tskip = nn.Linear(tx_in, hidden)

        # Attention fusion gate
        self.gate = nn.Sequential(
            nn.Linear(hidden*2, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Linear(hidden//2, 2), nn.Softmax(dim=-1))
        self.gn = nn.LayerNorm(hidden)

        # Skip + classifier
        self.skip = nn.Sequential(nn.Linear(tx_in, hidden//4), nn.GELU())
        self.clf  = nn.Sequential(
            nn.Linear(hidden+hidden//4, hidden//2),
            nn.LayerNorm(hidden//2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden//2, hidden//4), nn.GELU(),
            nn.Dropout(drop/2), nn.Linear(hidden//4, 2))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _tab(self, x):
        h =F.gelu(self.b1(self.t1(x))); h =F.dropout(h, self.drop,self.training)
        h1=F.gelu(self.b2(self.t2(h)));  h1=h1+h;  h1=F.dropout(h1,self.drop,self.training)
        h2=F.gelu(self.b3(self.t3(h1))); h2=h2+h1; h2=F.dropout(h2,self.drop,self.training)
        h3=F.gelu(self.b4(self.t4(h2))); h3=h3+h2; h3=F.dropout(h3,self.drop,self.training)
        return h3 + F.gelu(self.tskip(x))

    def forward(self, data):
        x   = data["transaction"].x
        h_t = self.tx_proj(x)
        h_a = self.addr_proj(data["address"].x)
        # GAT
        o1  = self.conv1({"transaction":h_t,"address":h_a}, data.edge_index_dict)
        h_t = F.gelu(self.n_tx1(h_t+o1.get("transaction",h_t)))
        h_a = F.gelu(self.n_ad1(h_a+o1.get("address",h_a)))
        h_t = F.dropout(h_t,self.drop,self.training)
        h_a = F.dropout(h_a,self.drop,self.training)
        # SAGE
        o2  = self.conv2({"transaction":h_t,"address":h_a}, data.edge_index_dict)
        hg  = F.gelu(self.n_tx2(h_t+o2.get("transaction",h_t)))
        hg  = F.dropout(hg,self.drop,self.training)
        # Tabular
        ht  = self._tab(x)
        # Fusion
        g   = self.gate(torch.cat([hg,ht],dim=-1))
        hf  = g[:,0:1]*hg + g[:,1:2]*ht
        hf  = F.gelu(self.gn(hf)); hf=F.dropout(hf,self.drop,self.training)
        return self.clf(torch.cat([hf,self.skip(x)],dim=-1))


model = GriffinGNN(
    tx_in=len(feature_cols), addr_in=addr_x.shape[1],
    hidden=HIDDEN, heads=4, drop=0.15
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: {n_params:,} parameters | hidden={HIDDEN} | device={device}")

# ============================================================
# 7) LOSS + OPTIMIZER + SCHEDULER
# ============================================================

counts  = np.bincount(y[train_mask], minlength=2)
w_t     = torch.tensor(counts.sum()/(2.*counts+1e-8),
                        dtype=torch.float32, device=device)
print(f"Class weights: w-={w_t[0]:.4f} | w+={w_t[1]:.2f}")

class AsymFocalLoss(nn.Module):
    def __init__(self, w, gp=2., gn=1.):
        super().__init__(); self.w=w; self.gp=gp; self.gn=gn
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits,targets,weight=self.w,reduction="none")
        pt = torch.exp(-ce)
        g  = torch.where(targets==1,
                         torch.full_like(ce,self.gp),
                         torch.full_like(ce,self.gn))
        return (((1-pt)**g)*ce).mean()

criterion = AsymFocalLoss(w_t)
# APRÈS :
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR_MAX, weight_decay=2e-4,
    fused=GPU_MODE  # fused kernel dispo uniquement sur CUDA
)

warmup_sched = LinearLR(optimizer,
                         start_factor=LR_START/LR_MAX,
                         end_factor=1., total_iters=WARMUP)
cosine_sched = CosineAnnealingLR(optimizer,
                                  T_max=EPOCHS-WARMUP, eta_min=1e-6)
scheduler    = SequentialLR(optimizer,
                              schedulers=[warmup_sched,cosine_sched],
                              milestones=[WARMUP])

# AMP only on GPU
use_amp = GPU_MODE
if use_amp:
    amp_sc = torch.amp.GradScaler("cuda")

# ============================================================
# 8) HELPERS
# ============================================================

@torch.no_grad()
def predict(loader):
    model.eval(); ps, ts = [], []
    for batch in loader:
        batch  = batch.to(device)
        bs     = batch["transaction"].batch_size
        logits = torch.nan_to_num(model(batch)[:bs],
                                   nan=0.,posinf=20.,neginf=-20.)
        ps.append(F.softmax(logits,dim=-1)[:,1].cpu().numpy())
        ts.append(batch["transaction"].y[:bs].cpu().numpy())
        del batch, logits
        if GPU_MODE: torch.cuda.empty_cache()
        else: gc.collect()
    return np.concatenate(ps), np.concatenate(ts)

def strata_report(model_p, y_true, rule_score, connected):
    """Modele vs regle sans modele, global et par strate. S1: adresse deja signalee; S2: connectee sans adresse signalee; S3: isolee."""
    model_p = np.asarray(model_p, float); y_true = np.asarray(y_true).astype(int)
    rule_score = np.asarray(rule_score, float); connected = np.asarray(connected).astype(bool)
    def block(sel):
        yy = y_true[sel]
        out = {"n": int(sel.sum()), "pos": int(yy.sum()), "neg": int((1-yy).sum()),
               "prevalence": float(yy.mean()) if sel.sum() else None}
        if 0 < yy.sum() < len(yy):
            out["model_AUC_PR"]  = round(float(average_precision_score(yy, model_p[sel])), 4)
            out["model_AUC_ROC"] = round(float(roc_auc_score(yy, model_p[sel])), 4)
            out["rule_AUC_PR"]   = round(float(average_precision_score(yy, rule_score[sel])), 4)
            out["rule_AUC_ROC"]  = round(float(roc_auc_score(yy, rule_score[sel])), 4)
        else:
            out["note"] = "pas de positifs ou pas de negatifs: metriques non definies"
        return out
    s1 = rule_score > 0
    s2 = (~s1) & connected
    s3 = (~s1) & (~connected)
    assert (s1 | s2 | s3).all()
    return {"all": block(np.ones(len(y_true), bool)), "S1_reported_address": block(s1),
            "S2_connected_not_reported": block(s2), "S3_isolated": block(s3)}

def best_th(p, y_true, beta=1.):
    pr,rc,thr = precision_recall_curve(y_true,p)
    num = (1+beta**2)*pr[:-1]*rc[:-1]
    den = beta**2*pr[:-1]+rc[:-1]+1e-12
    return float(thr[np.argmax(num/den)])

def mets(p, y_true, th):
    pred=(p>=th).astype(int)
    return {
        "AUC-ROC": round(roc_auc_score(y_true,p),4),
        "AUC-PR":  round(average_precision_score(y_true,p),4),
        "F1":      round(f1_score(y_true,pred,zero_division=0),4),
        "P":       round(precision_score(y_true,pred,zero_division=0),4),
        "R":       round(recall_score(y_true,pred,zero_division=0),4),
    }

# ============================================================
# 9) TRAINING LOOP
# ============================================================

# ============================================================
# 9) TRAINING LOOP (avec option de reprise sans réentraînement)
# ============================================================

RESUME_FROM_CHECKPOINT = False  # <-- mets False pour réentraîner depuis zéro
CKPT_PATH = f"{OUT_DIR}/griffin_v26_seed{SEED}_pretrain_checkpoint.pt"
if LOAD_PT:
    RESUME_FROM_CHECKPOINT = True
    CKPT_PATH = LOAD_PT

if RESUME_FROM_CHECKPOINT and os.path.exists(CKPT_PATH):
    print(f"⏭️  Reprise sans entraînement — chargement de {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)

    # On recalcule les métriques de validation manquantes
    # (elles n'ont pas été sauvegardées séparément la dernière fois)
    vp_resume, vt_resume = predict(val_loader)
    th_resume = best_th(vp_resume, vt_resume)
    vap_resume = average_precision_score(vt_resume, vp_resume)
    vf1_resume = f1_score(vt_resume, (vp_resume >= th_resume).astype(int), zero_division=0)
    comb_resume = 0.7 * vap_resume + 0.3 * vf1_resume

    best = (comb_resume, vap_resume, vf1_resume,
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    top_ckpts = [best]

    print(f"   Recalculé : comb={comb_resume:.4f} | AP={vap_resume:.4f} | F1={vf1_resume:.4f}")

else:
    top_ckpts=[]; sw=deque(maxlen=3); best_ws=-1.; wait=PATIENCE

    print(f"\n{'Ep':>3} | {'Loss':>8} | {'AP':>7} | {'F1':>7} | "
          f"{'Comb':>7} | {'LR':>9} | {'Time':>6}")
    print("-"*62)

    for epoch in range(1, EPOCHS+1):
        t0 = time.time()
        model.train(); losses=[]; optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch   = batch.to(device)
            bs      = batch["transaction"].batch_size

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(batch)[:bs]
                    loss   = criterion(logits, batch["transaction"].y[:bs]) / ACCUM_STEPS
                amp_sc.scale(loss).backward()
                if (step+1) % ACCUM_STEPS == 0:
                    amp_sc.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    amp_sc.step(optimizer); amp_sc.update()
                    optimizer.zero_grad()
            else:
                logits = model(batch)[:bs]
                loss   = criterion(logits, batch["transaction"].y[:bs])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad()

            losses.append(loss.item() * (ACCUM_STEPS if use_amp else 1))
            del batch, logits
            if GPU_MODE: torch.cuda.empty_cache()
            else: gc.collect()

        if use_amp:
            amp_sc.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_sc.step(optimizer); amp_sc.update()
            optimizer.zero_grad()

        scheduler.step()

        vp,vt = predict(val_loader)
        vap   = average_precision_score(vt,vp)
        th_v  = best_th(vp,vt)
        vf1   = f1_score(vt,(vp>=th_v).astype(int),zero_division=0)
        comb  = 0.7*vap + 0.3*vf1
        lr    = optimizer.param_groups[0]["lr"]
        dt    = time.time()-t0

        print(f"{epoch:3d} | {np.mean(losses):8.4f} | {vap:7.4f} | "
              f"{vf1:7.4f} | {comb:7.4f} | {lr:9.2e} | {dt:5.0f}s")

        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        top_ckpts.append((comb,vap,vf1,state))
        top_ckpts.sort(key=lambda x:x[0],reverse=True)
        top_ckpts = top_ckpts[:3]

        sw.append(comb); wa=np.mean(sw)
        if wa > best_ws + 1e-5:
            best_ws=wa; wait=PATIENCE
            print(f"    >>> Best window={wa:.4f}")
        else:
            wait -= 1
            if wait <= 0:
                print(f"Early stopping epoch {epoch}."); break

    best = top_ckpts[0]
    model.load_state_dict(best[3]); model.to(device)
    print(f"\nTop-3:")
    for i,(sc,ap,f1,_) in enumerate(top_ckpts):
        print(f"  #{i+1}: comb={sc:.4f} | AP={ap:.4f} | F1={f1:.4f}")
# ============================================================
# ============================================================
# 10) THRESHOLD CALIBRATION
# ============================================================

vp,vt  = predict(val_loader)
th_f1  = best_th(vp,vt,beta=1.0)
th_fp  = best_th(vp,vt,beta=0.5)
th_fr  = best_th(vp,vt,beta=2.0)
print(f"\nThresholds: θ_F1={th_f1:.4f} | θ_F0.5={th_fp:.4f} | θ_F2={th_fr:.4f}")

if not RESUME_FROM_CHECKPOINT:
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"✅ Checkpoint pré-sweep sauvegardé → {CKPT_PATH}")
# ============================================================
# REVISION : SAUVEGARDE DES PROBABILITES DE TEST (sans reentrainement ; checkpoint charge via LOAD_PT)
# Trois graphes de test : Known-Address (protocole principal), graphe naturel complet, isolation K=1.
# Sortie : test_probs_<RUN_TAG>.npz (hash des tx, snapshot, etiquette, probabilites, seuil theta_F1)
# ============================================================
_te_arr = np.where(test_mask)[0]
_p_known, _y_known = predict(make_loader(data_test, test_idx_t))
_p_nat, _ = predict(make_loader(make_hetero(inputs_df, outputs_df), test_idx_t))
_g1, _lam1, _ = build_test_topk(1)
_p_k1, _ = predict(make_loader(_g1, test_idx_t))
assert len(_p_known) == len(_te_arr) == len(_p_nat) == len(_p_k1)
assert (_y_known == y[_te_arr]).all()
np.savez_compressed(f"{OUT_DIR}/test_probs_{RUN_TAG}.npz",
                    tx_hash=np.asarray(transactions["tx_hash"].values[_te_arr], dtype="U80"),
                    snap=np.asarray(transactions["snapshot_id"].values[_te_arr], dtype="U8"),
                    y=y[_te_arr].astype(np.int8), th_f1=np.float64(th_f1), lam_k1=np.float64(_lam1),
                    p_known=_p_known.astype(np.float32), p_natural=_p_nat.astype(np.float32), p_k1=_p_k1.astype(np.float32))
print(f"Ecrit : {OUT_DIR}/test_probs_{RUN_TAG}.npz | n_test={len(_te_arr):,} | lambda(K=1)={_lam1:.4f}")
import sys; sys.exit(0)

# ============================================================
# 11) TEST EVALUATION
# ============================================================

tp,tt = predict(test_loader)
cp,ct = predict(conn_loader)
ip,it = predict(isol_loader)

print("\n=== FINAL TEST D5+D6 ===")
for name,th in [("F1",th_f1),("F0.5",th_fp),("F2",th_fr)]:
    m = mets(tp,tt,th)
    print(f"  [θ_{name}={th:.4f}] AUC-PR={m['AUC-PR']} | "
          f"AUC-ROC={m['AUC-ROC']} | F1={m['F1']} | "
          f"P={m['P']} | R={m['R']}")

m_main = mets(tp,tt,th_f1)
print(f"\nClassification report (θ_F1={th_f1:.4f}):")
print(classification_report(tt,(tp>=th_f1).astype(int),digits=4,zero_division=0))

print("=== STRUCTURAL DECOMPOSITION ===")
for nm,p_,t_ in [("Connected",cp,ct),("Isolated",ip,it)]:
    if t_.sum()==0: print(f"  {nm}: 0 fraud cases"); continue
    m = mets(p_,t_,th_f1)
    print(f"  {nm} ({len(t_):,} tx | fraud={t_.sum()}) | "
          f"AUC-PR={m['AUC-PR']} | AUC-ROC={m['AUC-ROC']} | F1={m['F1']}")

# ===== REVISION : strates (modele vs regle sans modele) + sauvegarde des metriques de la run =====
import json
te_arr = np.where(test_mask)[0]                       # meme ordre que tp / tt (shuffle=False)
hist_f = set(np.where(train_mask & (y == 1))[0].tolist())
_fi = inputs_df[inputs_df["tx_idx"].isin(hist_f)]; _fo = outputs_df[outputs_df["tx_idx"].isin(hist_f)]
tot_fr_rule = (np.bincount(_fi["addr_idx"].values, minlength=num_addr) +
               np.bincount(_fo["addr_idx"].values, minlength=num_addr))
_e = pd.concat([inp_te[["tx_idx","addr_idx"]], out_te[["tx_idx","addr_idx"]]]).drop_duplicates()
rule = (_e.assign(s=tot_fr_rule[_e["addr_idx"].values]).groupby("tx_idx")["s"].max()
          .reindex(te_arr, fill_value=0).values.astype(float))
assert len(rule) == len(tp) == len(tt)
strata = strata_report(tp, tt, rule, conn_mask[te_arr])
print("\n=== STRATES (modele vs regle sans modele) ===")
print(json.dumps(strata, indent=1))
run_metrics = {"seed": SEED, "feature_mode": FEATURE_MODE, "th_f1": th_f1, "test_known_graph": m_main,
               "connected": mets(cp, ct, th_f1), "isolated": mets(ip, it, th_f1), "strata": strata}
json.dump(run_metrics, open(f"{OUT_DIR}/run_metrics_{RUN_TAG}.json", "w"), indent=1)
print(f"Metriques sauvegardees -> {OUT_DIR}/run_metrics_{RUN_TAG}.json")

print("\n=== COMPARISON vs BASELINES ===")
rows = [
    ("LightGBM (tabular)",       0.7657, 0.9643, 0.7557, "0%"),
    ("GriffinGNN Standard λ=95%",0.8372, 0.9830, 0.7517, "94.99%"),
    ("GriffinGNN Isolated λ≈0%", 0.0581, 0.8426, 0.0237, "~0%"),
    ("GriffinGNN v26 Known-Addr",
     m_main["AUC-PR"], m_main["AUC-ROC"], m_main["F1"], "24.27%"),
]
print(f"{'Model':<32} | {'AUC-PR':>7} | {'AUC-ROC':>8} | {'F1':>7} | λ")
print("-"*70)
for nm,ap,auc,f1,lk in rows:
    print(f"{nm:<32} | {ap:7.4f} | {auc:8.4f} | {f1:7.4f} | {lk}")

# ============================================================
# 11bis) λ SWEEP v28 — CIBLAGE PAR RECHERCHE DICHOTOMIQUE
# Résolution fine (0.5%) dans la zone de transition 15%-30%
# + points plus espacés jusqu'à λ élevé (~95%+)
# ============================================================

import json
import gc

print("\n================ λ SWEEP v28 (ciblé) ================\n")

# Cache pour ne jamais recalculer deux fois le même K
lambda_cache = {}

def eval_K(K):
    """Construit le graphe Top-K, évalue, retourne (lam, metrics)."""
    if K in lambda_cache:
        return lambda_cache[K]
    data_topk, lam, _ = build_test_topk(K)
    loader = make_loader(data_topk, test_idx_t)
    probs, true = predict(loader)
    m = mets(probs, true, th_f1)
    result = (lam, m)
    lambda_cache[K] = result
    del loader, data_topk, probs, true
    gc.collect()
    if GPU_MODE: torch.cuda.empty_cache()
    return result

def find_K_for_lambda(target, K_lo=1, K_hi=20000, tol=0.0015, max_iter=16):
    """Recherche dichotomique de K tel que λ(K) ≈ target.
       λ est croissante (non strictement) avec K."""
    lam_lo, _ = eval_K(K_lo)
    lam_hi, _ = eval_K(K_hi)
    if target <= lam_lo:
        return K_lo
    if target >= lam_hi:
        return K_hi
    lo, hi = K_lo, K_hi
    best_K = K_hi
    for _ in range(max_iter):
        mid = (lo + hi) // 2
        if mid == lo:
            break
        lam_mid, _ = eval_K(mid)
        if abs(lam_mid - target) <= tol:
            return mid
        if lam_mid < target:
            lo = mid
        else:
            hi = mid
            best_K = mid
    return best_K

# -------------------------------------------------
# Grille de cibles λ
# -------------------------------------------------
targets = (
    list(np.round(np.arange(0.02, 0.15, 0.02), 4)) +   # 2%  -> 13%  : grossier, avant la zone
    list(np.round(np.arange(0.15, 0.31, 0.005), 4)) +  # 15% -> 30%  : FIN (pas de 0.5%) -> couvre 19/20/22/23/24/25%
    list(np.round(np.arange(0.32, 0.50, 0.02), 4)) +   # 32% -> 48%
    list(np.round(np.arange(0.50, 0.96, 0.05), 4)) +   # 50% -> 95%
    [0.99]
)
targets = sorted(set(targets))

# -------------------------------------------------
# Borne supérieure de K
# -------------------------------------------------
K_UPPER_BOUND = 20000
lam_max_check, _ = eval_K(K_UPPER_BOUND)
print(f"λ atteignable avec K={K_UPPER_BOUND} : {lam_max_check*100:.2f}%")
if lam_max_check < 0.90:
    K_UPPER_BOUND = 100000
    lam_max_check, _ = eval_K(K_UPPER_BOUND)
    print(f"Borne relevée -> K={K_UPPER_BOUND} : λ={lam_max_check*100:.2f}%")

# -------------------------------------------------
# Resume si checkpoint existant
# -------------------------------------------------
# ===== REVISION : ancrages calcules (Standard, Known-Address, Isole K=1) pour CETTE run =====
anchors = {"seed": SEED, "feature_mode": FEATURE_MODE, "load_pt": LOAD_PT}
anchors["known_address"] = dict(m_main)
_ps, _ts = predict(make_loader(make_hetero(inputs_df, outputs_df), test_idx_t))
anchors["standard"] = mets(_ps, _ts, th_f1)
_lam1, _m1 = eval_K(1)
anchors["isolated_K1"] = dict(_m1, lam=float(_lam1))
del _ps, _ts; gc.collect()

# ===== REVISION (R3-1) : isolation qui GARDE l'historique =====
# Les messages test -> adresse -> test sont supprimes en retirant les aretes tx -> adresse issues des transactions de TEST.
# Les aretes adresse -> tx (vers les transactions de test) et tout le voisinage train/validation sont conserves :
# lambda (test-test) = 0, rho (exposition historique) inchange.
def build_test_hist_preserved(base="known"):
    d = build_test_known() if base == "known" else make_hetero(inputs_df, outputs_df)
    te_flag = torch.zeros(num_tx, dtype=torch.bool)
    te_flag[torch.tensor(np.where(test_mask)[0], dtype=torch.long)] = True
    for et in (("transaction", "tx_to_input", "address"), ("transaction", "tx_to_output", "address")):
        ei = d[et].edge_index
        d[et].edge_index = ei[:, ~te_flag[ei[0]]]
    return d

for _base in ("known", "standard"):
    _g = build_test_hist_preserved(_base)
    _pp, _tt = predict(make_loader(_g, test_idx_t))
    anchors[f"hist_preserved_{_base}"] = mets(_pp, _tt, th_f1)
    del _g, _pp, _tt; gc.collect()
print("\n=== ANCRES de cette run ===")
print(json.dumps(anchors, indent=1))
json.dump(anchors, open(f"{OUT_DIR}/anchors_{RUN_TAG}.json", "w"), indent=1)

if not RUN_SWEEP:
    print("RUN_SWEEP=False : arret avant le balayage lambda (ancres, strates et metriques sauvegardees).")
    import sys; sys.exit(0)

checkpoint_file = f"{OUT_DIR}/lambda_sweep_v28_checkpoint.json"

if os.path.exists(checkpoint_file):
    with open(checkpoint_file) as f:
        sweep_results = json.load(f)
    done_targets = {round(x["target_lambda"], 4) for x in sweep_results}
    print(f"Checkpoint chargé ({len(done_targets)} points déjà calculés).")
else:
    sweep_results = []
    done_targets = set()

print(f"\n{'cible λ':>9} | {'K':>7} | {'λ réel':>8} | {'AUC-PR':>8} | {'F1':>8} | {'Prec':>8} | {'Rec':>8}")
print("-"*72)

for target in targets:
    if target in done_targets:
        continue
    K_found = find_K_for_lambda(target, K_lo=1, K_hi=K_UPPER_BOUND)
    lam, m = eval_K(K_found)
    print(f"{target*100:8.2f}% | {K_found:7d} | {lam*100:7.2f}% | "
          f"{m['AUC-PR']:8.4f} | {m['F1']:8.4f} | {m['P']:8.4f} | {m['R']:8.4f}")
    sweep_results.append({
        "target_lambda": float(target),
        "K": int(K_found),
        "lambda": float(lam),
        "AUC_PR": m["AUC-PR"],
        "AUC_ROC": m["AUC-ROC"],
        "F1": m["F1"],
        "Precision": m["P"],
        "Recall": m["R"],
    })
    with open(checkpoint_file, "w") as f:
        json.dump(sweep_results, f, indent=2)

sweep_results = sorted(sweep_results, key=lambda x: x["lambda"])
final_file = f"{OUT_DIR}/lambda_sweep_v28_final.json"
with open(final_file, "w") as f:
    json.dump(sweep_results, f, indent=2)

print("\n========================================")
print("λ Sweep v28 (ciblé) terminé.")
print(f"Résultats sauvegardés : {final_file}")
print("========================================")
# ============================================================
# 12) BOOTSTRAP CI
# ============================================================

print(f"\n=== BOOTSTRAP CI ({BOOT_ITERS} iter) ===")
rng  = np.random.default_rng(SEED)
boot = {k:[] for k in ["AUC-ROC","AUC-PR","F1","P","R"]}
for _ in range(BOOT_ITERS):
    idx = rng.integers(0,len(tt),size=len(tt))
    bt,bp = tt[idx],tp[idx]
    if bt.sum()==0 or bt.sum()==len(bt): continue
    for k,v in mets(bp,bt,th_f1).items():
        boot[k].append(v)
print(f"{'Metric':<8} | {'Mean':>7} | {'CI 2.5%':>9} | {'CI 97.5%':>9}")
print("-"*40)
for k,vals in boot.items():
    arr=np.array(vals); lo,hi=np.percentile(arr,2.5),np.percentile(arr,97.5)
    print(f"{k:<8} | {arr.mean():7.4f} | {lo:9.4f} | {hi:9.4f}")

# ============================================================
# 13) SAVE — Complete .pt
# ============================================================

print(f"\nSaving → {SAVE_PATH}")
torch.save({
    "model_state_dict":  model.state_dict(),
    "version":           f"griffin_v26_{MODE_TAG}",
    "hidden_dim":        HIDDEN,
    "heads":             4,
    "dropout":           0.15,
    "feature_cols":      feature_cols,
    "addr_feature_dim":  addr_x.shape[1],
    "device_mode":       MODE_TAG,
    "theta_f1":          float(th_f1),
    "theta_fb05":        float(th_fp),
    "theta_fb2":         float(th_fr),
    "best_val_combined": float(best[0]),
    "best_val_ap":       float(best[1]),
    "best_val_f1":       float(best[2]),
    "val_probs":  vp,  "val_true":  vt,
    "test_probs": tp,  "test_true": tt,
    "conn_probs": cp,  "conn_true": ct,
    "isol_probs": ip,  "isol_true": it,
    "boot_metrics": {k:np.array(v) for k,v in boot.items()},
    "connectivity_mask": conn_mask[test_mask],
    "lambda_sweep": sweep_results,
    "test_ap":  m_main["AUC-PR"],
    "test_auc": m_main["AUC-ROC"],
    "test_f1":  m_main["F1"],
    "baselines": {
        "LightGBM":    {"AUC_PR":0.7657,"AUC_ROC":0.9643,"F1":0.7557},
        "V21_Standard":{"AUC_PR":0.8372,"AUC_ROC":0.9830,"F1":0.7517},
        "V23_Isolated":{"AUC_PR":0.0581,"AUC_ROC":0.8426,"F1":0.0237},
    },
}, SAVE_PATH)

print(f"✅ Saved: {SAVE_PATH}")
print(f"   Exists: {os.path.exists(SAVE_PATH)}")
print(f"\n=== GriffinGNN v26 ({MODE_TAG.upper()}) — DONE ===")