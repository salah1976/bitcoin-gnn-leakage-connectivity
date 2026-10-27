# ============================================================
# label_audit.py -- audit des etiquettes BitFraud v21 (CPU, aucun modele)
# 1) Verifie que label_final == label_verified == "une adresse de la transaction a ete signalee (ChainAbuse) a une date <= la date de la tx"
#    (recalcul independant a partir de la base d'aretes et du fichier des adresses signalees).
# 2) Compte les transactions "signalees apres coup" (adresse signalee, mais signalement posterieur a la tx => etiquette 0).
# 3) Pour les positifs de test : date du premier signalement declencheur (avant D3 / pendant D4 / pendant D5-D6), role de l'adresse
#    (entree / sortie seulement), niveau de confiance ; et performance de la regle sans modele par strate de positifs.
# Sortie : 05_results/label_audit.json
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
OUT_JSON = f"{ROOT}/05_results/label_audit.json"
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

# ---------- 1) recalcul independant de l'etiquette ----------
Es = E[E["addr"].isin(first.index)].copy()
Es["report_dt"] = Es["addr"].map(first["report_dt"]).values
Es["tx_dt"] = lab["dt"].values[Es["t"].values]
Es["valid"] = Es["report_dt"] <= Es["tx_dt"]
lv = np.zeros(len(lab), int); lv[Es[Es["valid"]]["t"].unique()] = 1
retro = np.zeros(len(lab), int); retro[Es["t"].unique()] = 1
print("\n=== 1) Recalcul independant ===")
print(f"  etiquette recalculee == label_final : {(lv == lab['label_final'].values).mean():.4%} (positifs recalcules {lv.sum():,} | dans le fichier {lab['label_final'].sum():,})")
print(f"  appariement retrospectif (sans condition de date) : {retro.sum():,} tx ; dont etiquetees 0 car signalement posterieur : {int(((retro == 1) & (lv == 0)).sum()):,}")
lab["retro"] = retro; lab["lv"] = lv
print(lab.groupby("snapshot_id")[["lv", "retro"]].sum().assign(posterieur=lambda d: d["retro"] - d["lv"]).to_string())

# ---------- 2/3) positifs de test ----------
te = lab["snapshot_id"].isin(["D5", "D6"]).values
tr = lab["snapshot_id"].isin(["D1", "D2", "D3"]).values
va = (lab["snapshot_id"] == "D4").values
V = Es[Es["valid"]].copy()
trig = V.groupby("t")["report_dt"].min()                                   # premier signalement declencheur
role_in = V[V["role"] == "in"].groupby("t").size()
lab["trigger_dt"] = trig.reindex(lab.index)
end_n = {k: v.tz_localize(None) for k, v in end.items()}          # dates UTC sans fuseau (les dates de signalement ci-dessus le sont aussi)
def stratum(dt):
    if pd.isna(dt): return "none"
    if dt <= end_n["D3"]: return "A_signale_avant_fin_D3"
    if dt <= end_n["D4"]: return "B_signale_pendant_D4"
    return "C_signale_pendant_test"
lab["trig_stratum"] = lab["trigger_dt"].map(stratum)
lab["trig_by_input"] = lab.index.isin(role_in.index)

# regle sans modele : max sur les adresses de la tx du nombre de liens avec des tx frauduleuses d'ENTRAINEMENT
fr_tx = set(np.where(tr & (lab["label_final"].values == 1))[0])
cnt = E[E["t"].isin(fr_tx)].groupby("addr").size()
Et = E[te[E["t"].values]]
rule = pd.Series(Et["addr"].map(cnt).fillna(0).values, index=Et["t"].values).groupby(level=0).max()
score = np.zeros(len(lab)); score[rule.index] = rule.values

ytest = lab["label_final"].values[te]; stest = score[te]
print("\n=== 2) Positifs de test selon la date du signalement declencheur ===")
P = lab[te & (lab["label_final"].values == 1)]
rows = {}
for s, g in P.groupby("trig_stratum"):
    sel_pos = (te & (lab["trig_stratum"] == s).values & (lab["label_final"].values == 1))
    mask = ~(te & (lab["label_final"].values == 1) & ~sel_pos)         # positifs de la strate + tous les negatifs de test
    yy = lab["label_final"].values[mask & te]; ss = score[mask & te]
    d = {"n_positifs": int(len(g)), "part_des_positifs": round(len(g) / len(P), 4),
         "part_avec_adresse_deja_signalee_en_train(regle>0)": round(float((score[g.index] > 0).mean()), 4),
         "part_declenchee_par_une_adresse_d_entree": round(float(g["trig_by_input"].mean()), 4),
         "regle_AUC_PR": round(float(average_precision_score(yy, ss)), 4) if yy.sum() > 0 else None,
         "regle_AUC_ROC": round(float(roc_auc_score(yy, ss)), 4) if 0 < yy.sum() < len(yy) else None}
    rows[s] = d
    print(f"  {s:26s} {d}")
print("\n  niveau de confiance des positifs (test) :", P["external_best_tier"].value_counts().to_dict())
print("  niveau de confiance des positifs (train):", lab[tr & (lab['label_final'].values == 1)]["external_best_tier"].value_counts().to_dict())
print("  adresses signalees par niveau (premier signalement) :", tier_addr)

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump({"scam_csv": SCAM_CSV, "recalc_match": float((lv == lab['label_final'].values).mean()),
           "tiers_addresses": tier_addr, "test_positive_strata": rows,
           "test_tiers": P["external_best_tier"].value_counts().to_dict()}, open(OUT_JSON, "w"), indent=1, default=str)
print("\nEcrit :", OUT_JSON)
