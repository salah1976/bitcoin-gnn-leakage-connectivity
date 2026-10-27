# ============================================================
# construction_tables.py -- tableau unique : pour chaque construction du graphe de test, aretes, lambda, rho et CLASSES (CPU, sans modele)
# Reponses visees : R3.3 (table de K, aretes et classes par point du balayage), R5 / R3.1 (decomposition connecte / isole, fraudes conservees par construction).
# Constructions : natural, Known-Address, Top-K (K = 1, 2, 3, 5, 10, 25, 100, 1000, 100000 ; ordre d'origine), historique conserve (q = 0 a 1).
# Definitions (identiques a structure_sweep.py) :
#   connecte (lambda) = tx de test qui partage une adresse (aretes conservees) avec une AUTRE tx de test ;
#   historique (rho)  = tx de test qui garde au moins une arete vers une adresse deja vue en D1-D4 ;
#   fraudes conservees = positifs de test ayant encore une arete d'historique.
# Sorties : 05_results/construction_tables/{construction_tables.csv, construction_tables.tex}.   Test sans donnees : python construction_tables.py --selftest
# ============================================================
import os, sys, shutil, sqlite3
import numpy as np, pandas as pd

def row(name, e_conn, e_hist, te_arr, y, lam_override=None, conn_set=None):
    """e_conn : aretes (tx_idx, addr_idx) pour la connectivite ; e_hist : aretes vers des adresses connues (conservees)."""
    T = len(te_arr); yt = y[te_arr]; P = int(yt.sum()); Ng = T - P
    if conn_set is None:
        if len(e_conn):
            c = e_conn.groupby("addr_idx")["tx_idx"].nunique(); conn_set = set(e_conn[e_conn["addr_idx"].isin(c[c > 1].index)]["tx_idx"].unique())
        else: conn_set = set()
    hist_set = set(e_hist["tx_idx"].unique()) if len(e_hist) else set()
    ycon = lambda S: (int(y[list(S)].sum()) if S else 0, len(S) - (int(y[list(S)].sum()) if S else 0))
    pc, nc = ycon(conn_set); ph, nh = ycon(hist_set)
    return {"construction": name, "edges": int(len(e_conn)), "lambda_pct": 100 * len(conn_set) / T, "rho_pct": 100 * len(hist_set) / T,
            "pos_connected": pc, "neg_connected": nc, "prev_connected_pct": 100 * pc / max(pc + nc, 1),
            "pos_isolated": P - pc, "neg_isolated": Ng - nc, "prev_isolated_pct": 100 * (P - pc) / max(T - len(conn_set), 1),
            "pos_with_history": ph, "neg_with_history": nh, "fraud_retained_pct": 100 * ph / max(P, 1)}

def hist_preserved(kn, te_arr, y, q, seed=0):
    rng = np.random.default_rng(seed); order = rng.permutation(te_arr); removed = set(order[: int(q * len(order))].tolist())
    send = kn[~kn["tx_idx"].isin(removed)]; cnt = send.groupby("addr_idx")["tx_idx"].nunique()
    m = kn.merge(cnt.rename("n_send"), left_on="addr_idx", right_index=True, how="left").fillna({"n_send": 0})
    m["others"] = m["n_send"] - (~m["tx_idx"].isin(removed)).astype(int)
    conn = set(m[m["others"] > 0]["tx_idx"].unique())
    return row(f"history-preserved q={q:.2f}", send, kn, te_arr, y, conn_set=conn)

def selftest():
    te = np.arange(6); y = np.array([1, 0, 0, 1, 0, 0])
    kn = pd.DataFrame({"tx_idx": [0, 1, 2, 3, 4], "addr_idx": [10, 10, 11, 12, 12]})       # 0,1 partagent 10 ; 3,4 partagent 12 ; 2 seul ; 5 sans arete
    r = row("known", kn, kn, te, y)
    assert abs(r["lambda_pct"] - 100 * 4 / 6) < 1e-9 and r["pos_connected"] == 2 and r["neg_connected"] == 2 and r["pos_with_history"] == 2 and r["pos_isolated"] == 0
    h0 = hist_preserved(kn, te, y, 0.0); h1 = hist_preserved(kn, te, y, 1.0)
    assert abs(h0["lambda_pct"] - r["lambda_pct"]) < 1e-9 and h1["lambda_pct"] == 0 and h1["rho_pct"] == r["rho_pct"]   # q=1 : lambda=0, rho inchange
    print("selftest OK"); sys.exit(0)

if "--selftest" in sys.argv: selftest()

try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception: pass
ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
DB = os.environ.get("BITFRAUD_DB", f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"); LOCAL = os.environ.get("BITFRAUD_LOCALDB", "/content/all_snapshots_v21.db")
if not os.path.exists(LOCAL): shutil.copy2(DB, LOCAL)
OUT = f"{ROOT}/05_results/construction_tables"; os.makedirs(OUT, exist_ok=True)
SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]; K_LIST = [1, 2, 3, 5, 10, 25, 100, 1000, 100000]
lab = pd.read_csv(f"{ROOT}/01_data/processed/dataset_v21_full.csv", usecols=["tx_hash", "snapshot_id", "label_final"]); lab = lab[lab["snapshot_id"].isin(SNAP)].copy()
lab["rk"] = lab["snapshot_id"].map({s: i for i, s in enumerate(SNAP)}); lab = lab.sort_values(["rk", "tx_hash"]).reset_index(drop=True)
y = lab["label_final"].astype(int).values; idx = pd.Series(np.arange(len(lab)), index=lab["tx_hash"])
te_mask = lab["snapshot_id"].isin(["D5", "D6"]).values; tv_mask = lab["snapshot_id"].isin(["D1", "D2", "D3", "D4"]).values
te_arr = np.where(te_mask)[0]; te_ids = set(te_arr); tv_ids = set(np.where(tv_mask)[0])
c = sqlite3.connect(LOCAL)
E = pd.concat([pd.read_sql_query("SELECT tx_hash, input_address AS a FROM tx_inputs", c), pd.read_sql_query("SELECT tx_hash, output_address AS a FROM tx_outputs", c)]).dropna(); c.close()
E = E[E["tx_hash"].isin(idx.index)].copy(); E["tx_idx"] = idx.reindex(E["tx_hash"].values).values.astype(np.int64); E["addr_idx"] = pd.factorize(E["a"].astype(str))[0].astype(np.int64)
E = E[["tx_idx", "addr_idx"]].drop_duplicates().reset_index(drop=True)
known = set(E[E["tx_idx"].isin(tv_ids)]["addr_idx"].values)
nat = E[E["tx_idx"].isin(te_ids)].reset_index(drop=True); kn = nat[nat["addr_idx"].isin(known)].reset_index(drop=True)
cnt = kn.groupby("addr_idx")["tx_idx"].nunique(); shared = kn["addr_idx"].isin(cnt[cnt > 1].index).values; rk = kn.groupby("addr_idx").cumcount().values + 1
print(f"Test : {len(te_arr):,} tx, {int(y[te_arr].sum()):,} positifs | aretes natural {len(nat):,}, known {len(kn):,}")

R = [row("natural", nat, nat[nat["addr_idx"].isin(known)], te_arr, y), row("Known-Address", kn, kn, te_arr, y)]
for K in K_LIST:
    e = kn[(~shared) | (rk <= K)]; R.append(row(f"Top-K, K={K}", e, e, te_arr, y))
for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0): R.append(hist_preserved(kn, te_arr, y, q))
df = pd.DataFrame(R); df.to_csv(f"{OUT}/construction_tables.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
print("\n=== Constructions : aretes, lambda, rho, classes ===")
print(df[["construction", "edges", "lambda_pct", "rho_pct", "pos_with_history", "fraud_retained_pct"]].round(2).to_string(index=False))
print("\n=== Decomposition connecte / isole ===")
print(df[["construction", "pos_connected", "neg_connected", "prev_connected_pct", "pos_isolated", "neg_isolated", "prev_isolated_pct"]].round(3).to_string(index=False))
L = [r"\begin{tabular}{lrrrrr}\toprule", r"Construction & Edges & $\lambda$ (\%) & $\rho$ (\%) & Fraud kept & Fraud kept (\%) \\", r"\midrule"]
for _, r in df.iterrows(): L.append(f"{r['construction']} & {int(r['edges']):,} & {r['lambda_pct']:.2f} & {r['rho_pct']:.2f} & {int(r['pos_with_history']):,} & {r['fraud_retained_pct']:.1f} \\\\".replace(",", "{,}"))
L.append(r"\bottomrule\end{tabular}"); open(f"{OUT}/construction_tables.tex", "w").write("\n".join(L)); print("\nEcrit :", OUT)
