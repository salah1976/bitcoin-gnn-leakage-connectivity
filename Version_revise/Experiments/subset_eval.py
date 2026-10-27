# ============================================================
# subset_eval.py -- AUC-PR / AUC-ROC des GNN dans les sous-ensembles CONNECTE et ISOLE, par graine (CPU, probabilites deja enregistrees)
# Repond a R3.1 (decomposition connecte/isole sous le nouveau protocole, avec graines).
# Connecte = tx de test partageant une adresse conservee avec une autre tx de test, sur le graphe du protocole (natural ou Known-Address).
# Lit <ROOT>/05_results/runs/*_infer/test_probs_<mode>_seed<N>_infer.npz (cles y, p_known, p_natural, tx_hash).
# Sorties : 05_results/subset_eval/{subset_per_run.csv, subset_summary.csv}.   Test sans donnees : python subset_eval.py --selftest
# ============================================================
import os, sys, glob, sqlite3, shutil, warnings
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings("ignore")

def connected_hashes(E, test_hashes, known_addrs=None):
    """E : DataFrame (tx_hash, addr). Retourne l'ensemble des tx_hash de test connectees (adresse partagee, filtree sur known_addrs si fourni)."""
    e = E[E["tx_hash"].isin(test_hashes)]
    if known_addrs is not None: e = e[e["addr"].isin(known_addrs)]
    c = e.groupby("addr")["tx_hash"].nunique()
    return set(e[e["addr"].isin(c[c > 1].index)]["tx_hash"])

def sub_metrics(y, p, mask):
    ys, ps = y[mask], p[mask]
    if ys.sum() == 0 or ys.sum() == len(ys): return dict(n=int(mask.sum()), pos=int(ys.sum()), AUC_PR=np.nan, AUC_ROC=np.nan)
    return dict(n=int(mask.sum()), pos=int(ys.sum()), AUC_PR=average_precision_score(ys, ps), AUC_ROC=roc_auc_score(ys, ps))

def run(files, conn):
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f).replace("test_probs_", "").replace("_infer.npz", ""); mode, seed = tag.rsplit("_seed", 1)
        y = z["y"].astype(int); h = z["tx_hash"].astype(str)
        for proto in ("known", "natural"):
            cm = np.array([t in conn[proto] for t in h])
            for name, m in (("connected", cm), ("isolated", ~cm), ("all", np.ones(len(y), bool))):
                rows.append(dict(mode=mode, seed=int(seed), protocol=proto, subset=name, **sub_metrics(y, z["p_" + proto], m)))
    df = pd.DataFrame(rows); return df

def summarize(df, out):
    df.to_csv(f"{out}/subset_per_run.csv", index=False)
    S = df.groupby(["protocol", "mode", "subset"]).agg(n=("n", "first"), pos=("pos", "first"), AUC_PR_mean=("AUC_PR", "mean"), AUC_PR_sd=("AUC_PR", "std"), AUC_ROC_mean=("AUC_ROC", "mean"), AUC_ROC_sd=("AUC_ROC", "std"), seeds=("seed", "nunique")).round(3)
    S.to_csv(f"{out}/subset_summary.csv"); pd.set_option("display.width", 220); print(S.to_string()); return S

if "--selftest" in sys.argv:
    import tempfile
    d = tempfile.mkdtemp(); rng = np.random.default_rng(0); n = 2000; h = np.array([f"t{i}" for i in range(n)]); y = (rng.random(n) < 0.05).astype(int)
    E = pd.DataFrame({"tx_hash": h[:1000], "addr": rng.integers(0, 300, 1000)}); conn = {"known": connected_hashes(E, set(h)), "natural": connected_hashes(E, set(h))}
    assert 0 < len(conn["known"]) < n
    fs = []
    for s in (1, 2):
        f = f"{d}/test_probs_full_seed{s}_infer.npz"; p = np.clip(y * 0.5 + rng.random(n) * 0.6, 0, 1); np.savez(f, y=y, tx_hash=h, p_known=p, p_natural=p); fs.append(f)
    df = run(fs, conn); S = summarize(df, d); assert (df[df.subset == "all"]["pos"] == y.sum()).all(); print("selftest OK"); sys.exit(0)

try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception: pass
ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
DB = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"; LOCAL = "/content/all_snapshots_v21.db"
if not os.path.exists(LOCAL): shutil.copy2(DB, LOCAL)
OUT = f"{ROOT}/05_results/subset_eval"; os.makedirs(OUT, exist_ok=True)
lab = pd.read_csv(f"{ROOT}/01_data/processed/dataset_v21_full.csv", usecols=["tx_hash", "snapshot_id"]); te = set(lab[lab["snapshot_id"].isin(["D5", "D6"])]["tx_hash"]); tv = set(lab[lab["snapshot_id"].isin(["D1", "D2", "D3", "D4"])]["tx_hash"])
c = sqlite3.connect(LOCAL)
E = pd.concat([pd.read_sql_query("SELECT tx_hash, input_address AS addr FROM tx_inputs", c), pd.read_sql_query("SELECT tx_hash, output_address AS addr FROM tx_outputs", c)]).dropna().drop_duplicates(); c.close()
known = set(E[E["tx_hash"].isin(tv)]["addr"])
conn = {"natural": connected_hashes(E, te), "known": connected_hashes(E, te, known)}
print(f"Connectes : natural {len(conn['natural']):,} | Known-Address {len(conn['known']):,} (attendu 157 636 pour Known-Address si identique a construction_tables)")
files = sorted(glob.glob(f"{ROOT}/05_results/runs/*_infer/test_probs_*_infer.npz"))
if not files: raise SystemExit("STOP : aucun fichier test_probs sous " + ROOT)
summarize(run(files, conn), OUT); print("Ecrit :", OUT)
