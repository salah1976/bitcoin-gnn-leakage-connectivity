# ============================================================
# label_concentration.py -- concentration des etiquettes positives par adresse signalee (CPU, aucun modele)
# 1) Par niveau de confiance : combien d'adresses signalees sont ACTIVES (>=1 tx valide) et combien de tx positives elles portent.
# 2) Part des tx positives (toutes / test) portee par les N adresses signalees les plus frequentes (N = 1, 10, 50, 100, 500).
# 3) Regle sans modele recalculee APRES exclusion des tx qui ne sont positives que grace aux N adresses les plus frequentes
#    (exclues a la fois de l'entrainement de la regle et de l'evaluation) : robustesse du 0.92 a quelques adresses dominantes.
# Sortie : 05_results/label_concentration.json
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
OUT_JSON = f"{ROOT}/05_results/label_concentration.json"
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

# ---------- tx valides par adresse signalee ----------
Es = E[E["addr"].isin(first.index)].copy()
Es["report_dt"] = Es["addr"].map(first["report_dt"]).values
Es["tx_dt"] = lab["dt"].values[Es["t"].values]
V = Es[Es["report_dt"] <= Es["tx_dt"]].drop_duplicates(["t", "addr"]).copy()
V["tier"] = V["addr"].map(first["confidence_tier"]).values
V["snap"] = lab["snapshot_id"].values[V["t"].values]
y = lab["label_final"].values
te = lab["snapshot_id"].isin(["D5", "D6"]).values
tr = lab["snapshot_id"].isin(["D1", "D2", "D3"]).values
assert y[V["t"].unique()].min() == 1, "une tx valide devrait etre positive"
npos, npos_te = int(y.sum()), int((y[te] == 1).sum())

print("\n=== 1) Adresses signalees actives par niveau ===")
per_addr = V.groupby("addr").agg(n_tx=("t", "nunique"), tier=("tier", "first"))
res = {}
for tier, n_all in first["confidence_tier"].value_counts().items():
    a = per_addr[per_addr["tier"] == tier]
    Vt = V[V["tier"] == tier]
    d = {"adresses_signalees": int(n_all), "adresses_actives(>=1 tx valide)": int(len(a)),
         "tx_positives_touchees(distinctes)": int(Vt["t"].nunique()),
         "mediane_tx_par_adresse_active": float(a["n_tx"].median()) if len(a) else None,
         "max_tx_par_adresse": int(a["n_tx"].max()) if len(a) else None}
    res[tier] = d; print(f"  {tier:28s} {d}")

print("\n=== 2) Part des tx positives portee par les N adresses les plus frequentes ===")
rank = per_addr.sort_values("n_tx", ascending=False)
print("  10 adresses les plus frequentes (tx valides, niveau) :", [(a[:8] + "..", int(r.n_tx), r.tier[:5]) for a, r in rank.head(10).iterrows()])
conc = {}
for N in (1, 10, 50, 100, 500):
    top = set(rank.head(N).index)
    cov_all = V[V["addr"].isin(top)]["t"].nunique()
    cov_te = V[V["addr"].isin(top) & te[V["t"].values]]["t"].nunique()
    conc[N] = {"part_tx_positives": round(cov_all / npos, 4), "part_tx_positives_test": round(cov_te / npos_te, 4)}
    print(f"  N={N:4d} : {conc[N]['part_tx_positives']:.1%} de toutes les positives | {conc[N]['part_tx_positives_test']:.1%} des positives de test")

print("\n=== 3) Regle sans modele apres exclusion des tx dependantes des N adresses les plus frequentes ===")
def rule_eval(excl_tx):
    keep = np.ones(len(lab), bool); keep[list(excl_tx)] = False
    fr = set(np.where(tr & (y == 1) & keep)[0])
    cnt = E[E["t"].isin(fr)].groupby("addr").size()
    Et = E[te[E["t"].values] & keep[E["t"].values]]
    rule = pd.Series(Et["addr"].map(cnt).fillna(0).values, index=Et["t"].values).groupby(level=0).max()
    score = np.zeros(len(lab)); score[rule.index] = rule.values
    m = te & keep
    out = {"n_test": int(m.sum()), "positifs_test": int(y[m].sum()), "AUC_PR": None, "AUC_ROC": None}
    if 0 < y[m].sum() < m.sum():          # metriques definies seulement s'il reste des positifs et des negatifs
        out["AUC_PR"] = round(float(average_precision_score(y[m], score[m])), 4)
        out["AUC_ROC"] = round(float(roc_auc_score(y[m], score[m])), 4)
    return out
rules = {"N=0 (reference)": rule_eval(set())}
print("  N=0 (reference)", rules["N=0 (reference)"])
n_valid_addr = V.groupby("t")["addr"].apply(set)
for N in (1, 10, 50, 100, 500):
    top = set(rank.head(N).index)
    dep = {t for t, s in n_valid_addr.items() if s <= top}      # positives qui ne tiennent QU'A des adresses du top N
    rules[f"N={N}"] = dict(rule_eval(dep), tx_positives_exclues=len(dep))
    print(f"  N={N:4d} ", rules[f"N={N}"])

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump({"par_niveau": res, "concentration": conc, "regle_apres_exclusion": rules}, open(OUT_JSON, "w"), indent=1, default=str)
print("\nEcrit :", OUT_JSON)
