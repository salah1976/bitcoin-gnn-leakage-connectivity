# ============================================================
# LightGBM multi-graines (reference tabulaire et "informee par le graphe") -- CPU
# Reutilise EXACTEMENT le chargement, le split et les attributs d'adresse de griffin_v28_revision.py.
# Variantes :
#   A_tabular        : 15 attributs de transaction seuls (reference du Tableau 9)
#   B_graph_nolabel  : A + agregats d'adresses calcules sur l'historique D1-D3, SANS etiquettes
#   C_graph_label    : B + agregats avec etiquettes (fraud_ratio, log1p total_fraud, nb d'adresses signalees)
#   C_loo            : comme C, mais chaque transaction d'entrainement ne compte pas sa propre etiquette (leave-one-out)
# Protocoles de test : natural (toutes les aretes), known (adresses vues en D1-D4), isolated (aucune adresse)
# Sorties : 05_results/runs_lgbm/<variante>_seed<N>.json  +  tableau final moyenne +- ecart-type sur les 6 graines
# ============================================================
import os, json, shutil, sqlite3, time, gc, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    os.system("pip install -q lightgbm"); import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve
from google.colab import drive
drive.mount("/content/drive")

SEEDS = [42, 123, 2024, 7, 101, 202]
VARIANTS = os.environ.get("LGBM_VARIANTS", "A_tabular,B_graph_nolabel,C_graph_label,C_loo,A_no_prev,B_no_prev").split(",")
# A_no_prev / B_no_prev : retirent prev_addr_seen_ratio et prev_addr_seen_count (attributs deja lies a l'historique des adresses)
FEATURE_MODE = "full"          # utilise par le code copie de la v28 ; on lit addr_x_raw non modifie

ROOT       = "/content/drive/MyDrive/bitcoin_snapshots_project"
DRIVE_DB   = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
LOCAL_DB   = "/content/all_snapshots_v21.db"
LABELS_CSV = os.environ.get("LGBM_LABELS_CSV", f"{ROOT}/01_data/processed/dataset_v21_full.csv")
OUT_DIR    = os.environ.get("LGBM_OUT_DIR", f"{ROOT}/05_results/runs_lgbm")
os.makedirs(OUT_DIR, exist_ok=True)
SNAPSHOT_ORDER = ["D1","D2","D3","D4","D5","D6"]
TRAIN_SNAPS = ["D1","D2","D3"]; VAL_SNAPS = ["D4"]; TEST_SNAPS = ["D5","D6"]
feature_cols = [
    "input_count","output_count","input_addr_count",
    "coinbase_flag","has_witness","script_type_encoded",
    "input_addr_concentration","io_count_ratio","tx_weight",
    "avg_input_value","total_input_scaled","log_output_value",
    "fee_ratio","prev_addr_seen_ratio","prev_addr_seen_count",
]

# ---------- chargement + attributs d'adresse (copie de la v28) ----------
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

# ============================================================
# AGREGATS PAR TRANSACTION A PARTIR DES ATTRIBUTS D'ADRESSE (historique D1-D3 uniquement)
# ============================================================
A = addr_x_raw                               # (num_addr, 11) valeurs brutes
tot_deg = np.expm1(A[:, 2])                  # degre total sur l'historique
tot_fr  = np.expm1(A[:, 9])                  # nb de liens avec des transactions frauduleuses d'entrainement
recency = A[:, 10]
known_tv = np.zeros(num_addr, bool)          # adresses vues en D1-D4
_tv = set(np.where(train_mask | val_mask)[0].tolist())
for d_ in (inputs_df, outputs_df):
    known_tv[d_[d_["tx_idx"].isin(_tv)]["addr_idx"].values] = True

E_all = pd.concat([inputs_df[["tx_idx","addr_idx"]], outputs_df[["tx_idx","addr_idx"]]], ignore_index=True)
E = E_all.groupby(["tx_idx","addr_idx"]).size().reset_index(name="mult")      # aretes uniques + multiplicite
del E_all; gc.collect()
is_train_tx = train_mask.astype(float)

def aggregate(edges, loo=False):
    """features du graphe par transaction, calculees sur 'edges' (tx_idx, addr_idx, mult). Renvoie un DataFrame indexe 0..num_tx-1."""
    t, a, m = edges["tx_idx"].values, edges["addr_idx"].values, edges["mult"].values.astype(float)
    deg = tot_deg[a]; fr = tot_fr[a]
    if loo:                                   # retire la contribution propre des transactions d'ENTRAINEMENT
        fr  = fr  - y[t] * is_train_tx[t] * m
        deg = deg - is_train_tx[t] * m
        fr  = np.maximum(fr, 0); deg = np.maximum(deg, 0)
    df = pd.DataFrame({
        "tx": t,
        "g_n_addr": 1.0,
        "g_n_known": (deg > 0).astype(float),
        "g_max_logdeg": np.log1p(deg),
        "g_max_recency": recency[a],
        "l_max_fr_ratio": fr / (deg + 1e-8),
        "l_max_logfr": np.log1p(fr),
        "l_n_fraud_addr": (fr > 0).astype(float),
    })
    agg = df.groupby("tx").agg({"g_n_addr":"sum","g_n_known":"sum","g_max_logdeg":"max","g_max_recency":"max",
                                "l_max_fr_ratio":"max","l_max_logfr":"max","l_n_fraud_addr":"sum"})
    return agg.reindex(range(num_tx), fill_value=0.0)

G_COLS = ["g_n_addr","g_n_known","g_max_logdeg","g_max_recency"]
L_COLS = ["l_max_fr_ratio","l_max_logfr","l_n_fraud_addr"]

te_set = set(np.where(test_mask)[0].tolist())
E_te_all   = E[E["tx_idx"].isin(te_set)]
E_nontest  = E[~E["tx_idx"].isin(te_set)]
E_te_known = E_te_all[known_tv[E_te_all["addr_idx"].values]]
print("\nAgregats : natural / known / (isolated = zeros)")
AGG = {"natural": aggregate(E), "natural_loo": aggregate(E, loo=True)}
AGG["known"] = AGG["natural"].copy()
_k = aggregate(pd.concat([E_nontest, E_te_known], ignore_index=True))
te_idx = np.where(test_mask)[0]
AGG["known"].iloc[te_idx] = _k.iloc[te_idx].values
AGG["isolated"] = AGG["natural"].copy(); AGG["isolated"].iloc[te_idx] = 0.0
AGG["known_loo"] = AGG["natural_loo"].copy(); AGG["known_loo"].iloc[te_idx] = _k.iloc[te_idx].values   # test inchange par loo (pas de train)
AGG["isolated_loo"] = AGG["natural_loo"].copy(); AGG["isolated_loo"].iloc[te_idx] = 0.0
del _k; gc.collect()

# strates (meme definition que la v28) pour le test
rule = AGG["natural"]["l_max_logfr"].values                      # >0 <=> au moins une adresse deja signalee (S1)
conn = (AGG["known"]["g_n_addr"].values > 0)
s1 = rule[te_idx] > 0
conn_te = conn[te_idx]
STRATA = {"all": np.ones(len(te_idx), bool), "S1_reported": s1, "S2_connected_not_reported": (~s1) & conn_te, "S3_isolated": (~s1) & (~conn_te)}

X_tab = X_raw                                                    # attributs de transaction bruts (comme la v28 avant mise a l'echelle)
yv, ytr, yte = y[val_mask], y[train_mask], y[test_mask]

def design(variant, proto):
    """matrice d'attributs pour une variante et un protocole de test ; train/val identiques entre protocoles."""
    Xb = X_tab
    if variant.endswith("_no_prev"):
        keep = [i for i, c in enumerate(feature_cols) if c not in ("prev_addr_seen_ratio", "prev_addr_seen_count")]
        Xb = X_tab[:, keep]
    if variant in ("A_tabular", "A_no_prev"):
        return Xb
    loo = (variant == "C_loo")
    cols = G_COLS + (L_COLS if variant not in ("B_graph_nolabel", "B_no_prev") else [])
    key = lambda p: (p + "_loo") if loo else p
    Xg = AGG[key("natural")][cols].values.copy()
    if proto != "natural":
        Xg[te_idx] = AGG[key(proto)][cols].values[te_idx]
    return np.hstack([Xb, Xg]).astype(np.float32)

def mets(p, yt, th):
    pr = (p >= th).astype(int)
    return {"AUC-PR": round(float(average_precision_score(yt, p)), 4), "AUC-ROC": round(float(roc_auc_score(yt, p)), 4),
            "F1": round(float(f1_score(yt, pr, zero_division=0)), 4), "P": round(float(precision_score(yt, pr, zero_division=0)), 4),
            "R": round(float(recall_score(yt, pr, zero_division=0)), 4)}

def best_th(p, yt):
    pr, rc, thr = precision_recall_curve(yt, p)
    f = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(thr[np.argmax(f)])

def strata_block(p):
    out = {}
    for nm, sel in STRATA.items():
        yy = yte[sel]
        out[nm] = {"n": int(sel.sum()), "pos": int(yy.sum())}
        if 0 < yy.sum() < len(yy):
            out[nm]["AUC-PR"] = round(float(average_precision_score(yy, p[sel])), 4)
            out[nm]["AUC-ROC"] = round(float(roc_auc_score(yy, p[sel])), 4)
    return out

# ============================================================
# ENTRAINEMENT : 4 variantes x 6 graines
# ============================================================
results = {}
t_all = time.time()
for variant in VARIANTS:
    for seed in SEEDS:
        f_out = f"{OUT_DIR}/{variant}_seed{seed}.json"
        if os.path.exists(f_out):
            results[(variant, seed)] = json.load(open(f_out)); print(f"[skip] {variant} seed{seed}"); continue
        t0 = time.time()
        Xn = design(variant, "natural")
        Xtr, Xva = Xn[train_mask], Xn[val_mask]
        params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=20, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0, n_estimators=3000, random_state=seed,
                      n_jobs=-1, verbose=-1)
        clf = lgb.LGBMClassifier(**params)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yv)], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(150, verbose=False)])
        pv = clf.predict_proba(Xva)[:, 1]; th = best_th(pv, yv)
        res = {"variant": variant, "seed": seed, "best_iter": int(clf.best_iteration_ or params["n_estimators"]),
               "val_AUC-PR": round(float(average_precision_score(yv, pv)), 4), "th_f1": th}
        for proto in ("natural", "known", "isolated"):
            Xp = design(variant, proto)[test_mask]
            pt = clf.predict_proba(Xp)[:, 1]
            res[proto] = mets(pt, yte, th)
            if proto in ("natural", "known"):
                res[proto]["strata"] = strata_block(pt)
        json.dump(res, open(f_out, "w"), indent=1)
        results[(variant, seed)] = res
        print(f"{variant:16s} seed{seed:<5d} iter={res['best_iter']:<5d} val={res['val_AUC-PR']:.4f} | "
              f"natural={res['natural']['AUC-PR']:.4f} known={res['known']['AUC-PR']:.4f} isolated={res['isolated']['AUC-PR']:.4f} "
              f"({(time.time()-t0)/60:.1f} min)")
        del clf, Xn, Xtr, Xva; gc.collect()

# ============================================================
# RESUME : moyenne +- ecart-type (AUC-PR) sur les graines
# ============================================================
print("\n=== LightGBM, AUC-PR test D5-D6 (moyenne +- ecart-type, %d graines) ===" % len(SEEDS))
print(f"{'variante':18s} | {'natural':>14s} | {'known':>14s} | {'isolated':>14s} | {'S1 known':>14s} | {'S2 known AUC-ROC':>16s}")
summary = {}
for variant in VARIANTS:
    R = [results[(variant, s)] for s in SEEDS if (variant, s) in results]
    if not R: continue
    def ms(fn):
        v = np.array([fn(r) for r in R], float); return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
    nat, kn, iso = ms(lambda r: r["natural"]["AUC-PR"]), ms(lambda r: r["known"]["AUC-PR"]), ms(lambda r: r["isolated"]["AUC-PR"])
    s1 = ms(lambda r: r["known"]["strata"]["S1_reported"]["AUC-PR"])
    s2 = ms(lambda r: r["known"]["strata"]["S2_connected_not_reported"].get("AUC-ROC", np.nan))
    summary[variant] = {"n": len(R), "natural": nat, "known": kn, "isolated": iso, "S1_known": s1, "S2_known_AUC-ROC": s2}
    print(f"{variant:18s} | {nat[0]:.3f}+-{nat[1]:.3f} | {kn[0]:.3f}+-{kn[1]:.3f} | {iso[0]:.3f}+-{iso[1]:.3f} | {s1[0]:.3f}+-{s1[1]:.3f} | {s2[0]:.3f}+-{s2[1]:.3f}")
json.dump(summary, open(f"{OUT_DIR}/summary_lgbm.json", "w"), indent=1)
print(f"\nTermine en {(time.time()-t_all)/60:.1f} min. Resultats : {OUT_DIR}")
