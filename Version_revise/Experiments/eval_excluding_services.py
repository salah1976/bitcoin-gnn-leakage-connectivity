# ============================================================
# eval_excluding_services.py (variante de eval_excluding_dominant.py : exclusion par DEGRE de l'adresse signalee, sans jugement manuel) -- evaluation des 12 GNN (+ regle sans modele) SANS les adresses signalees dominantes (CPU)
# Lit runs/<mode>_seed<N>_infer/test_probs_*_infer.npz (cree par la cellule GPU cell_dump_and_run.py).
# Sous-ensembles de test : tous | D5 | D6 | sans tx dependantes du top-1 / top-10 / top-50 des adresses signalees.
#   "dependante du top-N" = tx positive dont TOUTES les adresses signalees valides (signalement <= date de la tx) sont dans le top-N
#   (classement par nombre de tx valides sur D1-D6) ; seules ces tx sont retirees (negatifs conserves). Evaluation seule : modeles et
#   regle ne sont PAS reentraines.
# Graphes : known (protocole principal), natural, k1 (isolation). Metriques : AUC-PR, AUC-ROC, F1 (seuil de validation), lift = AUC-PR/prevalence.
# Sorties : 05_results/dominant_eval/{per_run.csv, summary.csv} ; masques mis en cache dans dominant_masks.pkl
# ============================================================
import os, glob, json, shutil, sqlite3, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
from google.colab import drive
drive.mount("/content/drive")

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
DRIVE_DB = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
LOCAL_DB = "/content/all_snapshots_v21.db"
LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
OUT_DIR = f"{ROOT}/05_results/service_eval"
RUNS = f"{ROOT}/05_results/runs"
OUT_JSON = f"{OUT_DIR}/summary.json"
SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]

# fichier des adresses signalees (cherche scam_addresses_v21_frozen.csv, sinon v20)
cands = sorted(glob.glob(f"{ROOT}/**/scam_addresses_v21_frozen.csv", recursive=True)) or \
        sorted(glob.glob(f"{ROOT}/**/scam_addresses_v2*_frozen.csv", recursive=True))
if not cands:
    raise SystemExit("STOP : scam_addresses_v21_frozen.csv introuvable sous " + ROOT + ". Dis-moi ou il se trouve.")
SCAM_CSV = cands[0]; print("Fichier des adresses signalees :", SCAM_CSV)
if not os.path.exists(LOCAL_DB): shutil.copy2(DRIVE_DB, LOCAL_DB)

# ---------- adresses signalees ----------
sc = pd.read_csv(SCAM_CSV)
sc["address"] = sc["address"].astype(str).str.strip().str.lower()
sc["report_dt"] = pd.to_datetime(sc["created_at"], utc=True, errors="coerce")
sc = sc.dropna(subset=["address", "report_dt"])
print(f"\nLignes : {len(sc):,} | adresses distinctes : {sc['address'].nunique():,} | signalements du {sc['report_dt'].min()} au {sc['report_dt'].max()}")
print("Niveaux de confiance (lignes) :\n", sc["confidence_tier"].value_counts().to_string())
first = sc.sort_values("report_dt").groupby("address").first()          # premier signalement par adresse (comme le script 02)
tier_addr = first["confidence_tier"].value_counts().to_dict()

# ---------- transactions ----------
lab = pd.read_csv(LABELS_CSV, usecols=["tx_hash", "snapshot_id", "datetime", "label_final", "label_verified",
                                       "label_verified_retro", "external_future_info_flag", "external_best_tier"])
lab = lab[lab["snapshot_id"].isin(SNAP)].reset_index(drop=True)
lab["dt"] = pd.to_datetime(lab["datetime"], utc=True)
pos = pd.Series(np.arange(len(lab)), index=lab["tx_hash"])
print("\nlabel_final == label_verified pour %.4f%% des transactions" % (100 * (lab["label_final"] == lab["label_verified"]).mean()))
end = lab.groupby("snapshot_id")["dt"].max()
print("Fin de chaque snapshot :", {k: str(v)[:10] for k, v in end.items()})

conn = sqlite3.connect(LOCAL_DB)
ins = pd.read_sql_query("SELECT tx_hash, input_address AS addr FROM tx_inputs", conn).dropna()
outs = pd.read_sql_query("SELECT tx_hash, output_address AS addr FROM tx_outputs", conn).dropna()
conn.close()
ins["role"] = "in"; outs["role"] = "out"
E = pd.concat([ins, outs], ignore_index=True)
E = E[E["tx_hash"].isin(pos.index)].copy()
E["addr"] = E["addr"].astype(str).str.strip().str.lower()
E["t"] = E["tx_hash"].map(pos).astype(np.int64)
del ins, outs

os.makedirs(OUT_DIR, exist_ok=True)
from sklearn.metrics import f1_score

# ---------- masques de dependance et regle sans modele (calcul une fois) ----------
Es = E[E["addr"].isin(first.index)].drop_duplicates(["t", "addr"]).copy()
Es["report_dt"] = Es["addr"].map(first["report_dt"]).values
Es["tx_dt"] = lab["dt"].values[Es["t"].values]
V = Es[Es["report_dt"] <= Es["tx_dt"]]
rank = V.groupby("addr")["t"].nunique().sort_values(ascending=False)
vsets = V.groupby("t")["addr"].apply(set)
dep = {}
deg = E.groupby("addr")["t"].nunique()                      # degre d'une adresse = nb de tx distinctes du benchmark qui la touchent (D1-D6)
THR = (100, 500, 1000, 5000)
for T in THR:
    hi = set(deg[deg >= T].index) & set(first.index)
    d = np.zeros(len(lab), bool); d[[t for t, s_ in vsets.items() if s_ <= hi]] = True
    dep["deg%d" % T] = d
    print(f"degre >= {T}: {len(hi):,} adresses signalees ; {d[lab['snapshot_id'].isin(['D5','D6']).values].sum():,} tx positives de test dependantes ; reste {int(lab['label_final'].values[lab['snapshot_id'].isin(['D5','D6']).values].sum()) - int(d[lab['snapshot_id'].isin(['D5','D6']).values].sum()):,}")
for N in (1, 10, 50):
    top = set(rank.head(N).index)
    d = np.zeros(len(lab), bool)
    d[[t for t, s_ in vsets.items() if s_ <= top]] = True
    dep[N] = d
    print(f"top-{N}: {d.sum():,} tx positives dependantes ({d[lab['snapshot_id'].isin(['D5','D6']).values].sum():,} en test)")
te = lab["snapshot_id"].isin(["D5", "D6"]).values
tr = lab["snapshot_id"].isin(["D1", "D2", "D3"]).values
y_all = lab["label_final"].values
fr_tx = set(np.where(tr & (y_all == 1))[0])
cnt = E[E["t"].isin(fr_tx)].groupby("addr").size()
Et = E[te[E["t"].values]]
rs = pd.Series(Et["addr"].map(cnt).fillna(0).values, index=Et["t"].values).groupby(level=0).max()
rule_all = np.zeros(len(lab)); rule_all[rs.index] = rs.values

# ---------- metriques ----------
def metr(y, p, th=None):
    n, k = len(y), int(y.sum())
    r = {"n": n, "pos": k, "prev": k / n if n else np.nan, "AUC_PR": np.nan, "AUC_ROC": np.nan, "F1": np.nan, "lift": np.nan}
    if 0 < k < n:
        r["AUC_PR"] = float(average_precision_score(y, p)); r["AUC_ROC"] = float(roc_auc_score(y, p)); r["lift"] = r["AUC_PR"] / r["prev"]
        if th is not None: r["F1"] = float(f1_score(y, (p >= th).astype(int), zero_division=0))
    return r

def subsets(t_idx, snap):
    return {"all": np.ones(len(t_idx), bool), "D5": snap == "D5", "D6": snap == "D6",
            "excl_top1": ~dep[1][t_idx], "excl_top10": ~dep[10][t_idx], "excl_top50": ~dep[50][t_idx],
            **{"excl_deg%d" % T: ~dep["deg%d" % T][t_idx] for T in THR}}

rows = []
files = sorted(glob.glob(f"{RUNS}/*_infer/test_probs_*_infer.npz"))
print(f"\n{len(files)} fichiers de probabilites trouves")
if not files:
    raise SystemExit("STOP : aucun fichier test_probs_*_infer.npz sous " + RUNS + ". Lance d'abord la cellule GPU cell_dump_and_run.py (elle les cree), puis relance celle-ci.")
for f in files:
    z = np.load(f, allow_pickle=True)
    tag = os.path.basename(f).replace("test_probs_", "").replace("_infer.npz", "")
    mode, seed = tag.rsplit("_seed", 1)
    t_idx = pos[pd.Index(z["tx_hash"])].values
    assert (lab["label_final"].values[t_idx] == z["y"]).all(), "etiquettes incoherentes pour " + tag
    snap = z["snap"]; y = z["y"].astype(int); th = float(z["th_f1"])
    for name, mk in subsets(t_idx, snap).items():
        for proto in ("known", "natural", "k1"):
            rows.append(dict(mode=mode, seed=int(seed), protocol=proto, subset=name, **metr(y[mk], z["p_" + proto][mk], th)))
    # regle sans modele (une seule fois, graine 0)
    if not any(r["mode"] == "rule" for r in rows):
        for name, mk in subsets(t_idx, snap).items():
            rows.append(dict(mode="rule", seed=0, protocol="known", subset=name, **metr(y[mk], rule_all[t_idx][mk])))
df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/per_run.csv", index=False)
agg = (df.groupby(["mode", "protocol", "subset"])
         .agg(n_runs=("seed", "nunique"), pos=("pos", "mean"), prev=("prev", "mean"), AUC_PR=("AUC_PR", "mean"), AUC_PR_sd=("AUC_PR", "std"),
              AUC_ROC=("AUC_ROC", "mean"), F1=("F1", "mean"), lift=("lift", "mean")).reset_index())
agg.to_csv(f"{OUT_DIR}/summary.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
order = ["all", "D5", "D6", "excl_top1", "excl_top10", "excl_top50"] + ["excl_deg%d" % T for T in THR]
for proto in ("known", "natural", "k1"):
    print(f"\n=== AUC-PR moyenne (ecart-type) | protocole {proto} ===")
    for mode in ("full", "no_fraud", "rule"):
        a = agg[(agg["mode"] == mode) & (agg["protocol"] == proto)].set_index("subset")
        if a.empty: continue
        print(f"{mode:9s}", " | ".join(f"{s}: {a.loc[s, 'AUC_PR']:.3f}" + (f" ({a.loc[s, 'AUC_PR_sd']:.3f})" if not np.isnan(a.loc[s, 'AUC_PR_sd']) else "") for s in order if s in a.index))
print("\n=== positifs et prevalence par sous-ensemble (protocole known) ===")
a = agg[(agg["mode"] == agg["mode"].iloc[0]) & (agg["protocol"] == "known")].set_index("subset").loc[order, ["pos", "prev"]]
print(a.to_string())
print("\n=== lift (AUC-PR / prevalence), protocole known ===")
for mode in ("full", "no_fraud", "rule"):
    a = agg[(agg["mode"] == mode) & (agg["protocol"] == "known")].set_index("subset")
    if not a.empty: print(f"{mode:9s}", " | ".join(f"{s}: {a.loc[s, 'lift']:.0f}" for s in order if s in a.index))
print("\nEcrit :", OUT_DIR)
