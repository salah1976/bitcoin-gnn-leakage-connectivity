# ============================================================
# structure_sweep.py -- balayage Top-K SANS MODELE (CPU) : ce que le balayage change reellement dans le graphe
# Reponses visees : biais de selection du Top-K (R : "suppression aleatoire a lambda egal"), confusion lambda / rho.
# Pour chaque K : lambda (test-test), rho_K (part des transactions de test qui gardent au moins une adresse deja vue en D1-D4),
#                 regle sans modele (AUC-PR / AUC-ROC) calculee sur les aretes conservees.
# Trois ordres de selection : 'orig' (celui de build_test_topk), 'random' (ordre aleatoire, 3 tirages), et verification que le
# score d'exposition utilise par build_test_topk est constant pour les transactions de test.
# Sortie : 05_results/structure_sweep.json et tableau imprime.
# ============================================================
import os, json, shutil, sqlite3, time, gc, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
from google.colab import drive
drive.mount("/content/drive")

ROOT       = "/content/drive/MyDrive/bitcoin_snapshots_project"
DRIVE_DB   = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
LOCAL_DB   = "/content/all_snapshots_v21.db"
LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
OUT_JSON   = f"{ROOT}/05_results/structure_sweep.json"
SNAPSHOT_ORDER = ["D1","D2","D3","D4","D5","D6"]
TRAIN_SNAPS = ["D1","D2","D3"]; VAL_SNAPS = ["D4"]; TEST_SNAPS = ["D5","D6"]
feature_cols = [
    "input_count","output_count","input_addr_count",
    "coinbase_flag","has_witness","script_type_encoded",
    "input_addr_concentration","io_count_ratio","tx_weight",
    "avg_input_value","total_input_scaled","log_output_value",
    "fee_ratio","prev_addr_seen_ratio","prev_addr_seen_count",
]
K_LIST = [1, 2, 3, 5, 10, 25, 100, 1000, 100000]
N_SHUFFLES = 3

# ---------- chargement (copie de la v28) ----------
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
# TABLE DES ARETES DE TEST (adresses deja vues en D1-D4), dans le meme ordre que build_test_topk
# ============================================================
tv_ids = set(np.where(train_mask | val_mask)[0].tolist())
te_ids = set(np.where(test_mask)[0].tolist())
te_arr = np.where(test_mask)[0]
inp_tv = inputs_df[inputs_df["tx_idx"].isin(tv_ids)]
out_tv = outputs_df[outputs_df["tx_idx"].isin(tv_ids)]
known  = set(inp_tv["addr_idx"].values) | set(out_tv["addr_idx"].values)
inp_te = inputs_df[inputs_df["tx_idx"].isin(te_ids)]
out_te = outputs_df[outputs_df["tx_idx"].isin(te_ids)]
ik = inp_te[inp_te["addr_idx"].isin(known)]
ok = out_te[out_te["addr_idx"].isin(known)]
te_edges = pd.concat([ik[["tx_idx","addr_idx"]], ok[["tx_idx","addr_idx"]]]).drop_duplicates().reset_index(drop=True)
print(f"\nAretes test -> adresse connue : {len(te_edges):,} | transactions de test : {len(te_ids):,}")

# verification : score d'exposition de build_test_topk (calcule sur les transactions TRAIN/VAL uniquement)
tv_addr_per_tx = pd.concat([inp_tv[["tx_idx","addr_idx"]], out_tv[["tx_idx","addr_idx"]]])
exposure_score = tv_addr_per_tx.groupby("tx_idx")["addr_idx"].nunique()
sc = te_edges["tx_idx"].map(exposure_score)
print(f"Score d'exposition de build_test_topk : defini pour {sc.notna().sum():,} aretes de test sur {len(te_edges):,} "
      f"(les autres recoivent 0) | valeurs distinctes : {sorted(sc.fillna(0).unique().tolist())[:5]}")
del tv_addr_per_tx, exposure_score, sc

addr_counts  = te_edges.groupby("addr_idx")["tx_idx"].nunique()
shared_addrs = set(addr_counts[addr_counts > 1].index)
is_shared = te_edges["addr_idx"].isin(shared_addrs).values

# regle sans modele : score = max (sur les aretes conservees) du nombre de liens de l'adresse avec des tx frauduleuses d'entrainement
hist_f = set(np.where(train_mask & (y == 1))[0].tolist())
_fi = inputs_df[inputs_df["tx_idx"].isin(hist_f)]; _fo = outputs_df[outputs_df["tx_idx"].isin(hist_f)]
tot_fr = np.bincount(_fi["addr_idx"].values, minlength=num_addr) + np.bincount(_fo["addr_idx"].values, minlength=num_addr)
yte = y[te_arr]
pos_in_te = pd.Series(np.arange(len(te_arr)), index=te_arr)

def evaluate(keep):
    """keep : masque booleen sur te_edges. Renvoie lambda, rho, S1 retenue, regle."""
    e = te_edges[keep]
    fc = e.groupby("addr_idx")["tx_idx"].nunique()
    connected = e[e["addr_idx"].isin(fc[fc > 1].index)]["tx_idx"].nunique()
    lam = connected / len(te_ids)
    rho = e["tx_idx"].nunique() / len(te_ids)
    s = pd.Series(tot_fr[e["addr_idx"].values], index=e["tx_idx"].values).groupby(level=0).max()
    score = np.zeros(len(te_arr)); score[pos_in_te.reindex(s.index).values.astype(int)] = s.values
    return {"lambda": float(lam), "rho": float(rho), "n_edges": int(len(e)),
            "rule_AUC_PR": round(float(average_precision_score(yte, score)), 4),
            "rule_AUC_ROC": round(float(roc_auc_score(yte, score)), 4),
            "fraud_with_reported_addr": round(float(((score > 0) & (yte == 1)).sum() / yte.sum()), 4)}

def keep_mask(order_rank, K):
    return (~is_shared) | (order_rank <= K)

# ordre d'origine : rang = ordre d'apparition dans te_edges au sein de chaque adresse (method="first", scores tous egaux)
rank_orig = te_edges.groupby("addr_idx").cumcount().values + 1
res = {"orig": {}, "random": {}}
print(f"\n{'K':>7} | {'ordre':>8} | {'lambda':>8} | {'rho':>7} | {'aretes':>9} | {'regle AUC-PR':>12} | {'AUC-ROC':>8}")
print("-" * 78)
full = evaluate(np.ones(len(te_edges), bool))
print(f"{'natural':>7} | {'-':>8} | {full['lambda']:8.4f} | {full['rho']:7.4f} | {full['n_edges']:9,d} | {full['rule_AUC_PR']:12.4f} | {full['rule_AUC_ROC']:8.4f}")
res["known_all_edges"] = full
for K in K_LIST:
    r = evaluate(keep_mask(rank_orig, K)); res["orig"][K] = r
    print(f"{K:7d} | {'orig':>8} | {r['lambda']:8.4f} | {r['rho']:7.4f} | {r['n_edges']:9,d} | {r['rule_AUC_PR']:12.4f} | {r['rule_AUC_ROC']:8.4f}")
    rr = []
    for sd in range(N_SHUFFLES):
        perm = np.random.default_rng(sd).permutation(len(te_edges))
        shuffled = te_edges.iloc[perm].reset_index(drop=True)
        rk = np.empty(len(te_edges), int)
        rk[perm] = shuffled.groupby("addr_idx").cumcount().values + 1
        rr.append(evaluate(keep_mask(rk, K)))
    avg = {k: float(np.mean([x[k] for x in rr])) for k in rr[0]}
    sd_ = {k: float(np.std([x[k] for x in rr])) for k in rr[0]}
    res["random"][K] = {"mean": avg, "std": sd_}
    print(f"{K:7d} | {'random':>8} | {avg['lambda']:8.4f} | {avg['rho']:7.4f} | {int(avg['n_edges']):9,d} | {avg['rule_AUC_PR']:12.4f} | {avg['rule_AUC_ROC']:8.4f}")

# ============================================================
# BALAYAGE "HISTORIQUE CONSERVE" (sans modele) : on retire la capacite d'une fraction q de transactions de test a EMETTRE des messages
# (equivaut a supprimer leurs aretes tx -> adresse, comme l'isolation avec historique) ; rho reste constant.
# lambda_hp(q) = part des transactions de test qui partagent une adresse avec une AUTRE transaction de test dont l'arete est conservee.
# ============================================================
print("\n=== Balayage historique conserve : lambda(q) avec rho constant ===")
res["hist_preserved"] = {}
rng = np.random.default_rng(0)
order_tx = rng.permutation(te_arr)
for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
    removed = set(order_tx[: int(q * len(order_tx))].tolist())
    sender_ok = ~te_edges["tx_idx"].isin(removed).values            # aretes tx -> adresse conservees
    es = te_edges[sender_ok]
    cnt = es.groupby("addr_idx")["tx_idx"].nunique()
    # une transaction t est "connectee" si une adresse de t porte au moins un AUTRE emetteur conserve
    m = te_edges.merge(cnt.rename("n_send"), left_on="addr_idx", right_index=True, how="left").fillna({"n_send": 0})
    m["self_send"] = (~m["tx_idx"].isin(removed)).astype(int)
    m["others"] = m["n_send"] - m["self_send"]
    lam_hp = m[m["others"] > 0]["tx_idx"].nunique() / len(te_ids)
    res["hist_preserved"][q] = {"lambda": float(lam_hp), "rho": full["rho"]}
    print(f"  q={q:4.2f} : lambda={lam_hp:7.4f} | rho={full['rho']:7.4f} (inchange)")

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(res, open(OUT_JSON, "w"), indent=1, default=float)
print("\nEcrit :", OUT_JSON)
