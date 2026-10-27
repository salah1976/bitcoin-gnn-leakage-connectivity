# ============================================================
# eval_label_strict.py -- sensibilite aux etiquettes, DANS le jeu v21 de Griffin (CPU, aucun reentrainement)
# Evalue les 12 GNN (entraines sur les etiquettes v21) contre les etiquettes de TEST 'strictes' (v21-bis : tier3 exclu, donc seulement
# les signalements verifies sur ChainAbuse, tiers 1 et 2) pour la meme liste de transactions de test. Peu de positifs de test (puissance faible).
# Lit runs/<mode>_seed<N>_infer/test_probs_*_infer.npz (tx_hash, snap, y, p_known, p_natural, p_k1).
# Sorties : 05_results/label_strict_eval/{per_run.csv, summary.csv, agreement.json}
# ATTENTION : seules les etiquettes de TEST changent ; l'entrainement reste sur v21 (limite a dire).
# ============================================================
import os, glob, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass
ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
RUNS = f"{ROOT}/05_results/runs"; OUT = f"{ROOT}/05_results/label_strict_eval"; os.makedirs(OUT, exist_ok=True)

c = sorted(glob.glob(f"{ROOT}/**/dataset_v21_full1.csv", recursive=True))
if not c: raise SystemExit("STOP : dataset_v21_full1.csv (sortie du script v21-bis, dossier processed_v21_final1) introuvable sous " + ROOT)
V22 = c[0]; print("Etiquettes strictes (v21-bis, tier3 exclu) :", V22)
head = pd.read_csv(V22, nrows=2); print("Colonnes :", list(head.columns))
need = [x for x in ("tx_hash", "label_final") if x not in head.columns]
if need: raise SystemExit(f"STOP : colonnes manquantes {need} dans {V22}. Colle-moi la liste des colonnes ci-dessus.")
lab22 = pd.read_csv(V22, usecols=["tx_hash", "label_final"]).drop_duplicates("tx_hash").set_index("tx_hash")["label_final"].astype(int)

files = sorted(glob.glob(f"{RUNS}/*_infer/test_probs_*_infer.npz"))
print(f"{len(files)} fichiers de probabilites")
if not files: raise SystemExit("STOP : aucun fichier test_probs ; lance la cellule GPU d'abord.")

def metr(y, p):
    n, k = len(y), int(y.sum()); r = dict(n=n, pos=k, prev=k / n if n else np.nan, AUC_PR=np.nan, AUC_ROC=np.nan, lift=np.nan)
    if 0 < k < n:
        r["AUC_PR"] = float(average_precision_score(y, p)); r["AUC_ROC"] = float(roc_auc_score(y, p)); r["lift"] = r["AUC_PR"] / r["prev"]
    return r

rows, agree = [], None
for f in files:
    z = np.load(f, allow_pickle=True)
    tag = os.path.basename(f).replace("test_probs_", "").replace("_infer.npz", ""); mode, seed = tag.rsplit("_seed", 1)
    h = pd.Index(z["tx_hash"]); ok = h.isin(lab22.index)
    y21 = z["y"].astype(int)[ok]; ystrict = lab22.reindex(h[ok]).values.astype(int); snap = z["snap"][ok]
    if agree is None:
        agree = {"n_test_matched": int(ok.sum()), "n_test_total": int(len(h)),
                 "both1": int(((y21 == 1) & (ystrict == 1)).sum()), "only_v21": int(((y21 == 1) & (ystrict == 0)).sum()),
                 "only_strict": int(((y21 == 0) & (ystrict == 1)).sum()), "both0": int(((y21 == 0) & (ystrict == 0)).sum()),
                 "pos_v21": int(y21.sum()), "pos_strict": int(ystrict.sum())}
        print("\nAccord des etiquettes de test (v21 vs strict) :", agree)
    for name, mk in (("all", np.ones(len(snap), bool)), ("D5", snap == "D5"), ("D6", snap == "D6")):
        for proto in ("known", "natural", "k1"):
            for lname, yy in (("v21", y21), ("strict", ystrict)):
                rows.append(dict(mode=mode, seed=int(seed), protocol=proto, subset=name, labels=lname, **metr(yy[mk], z["p_" + proto][ok][mk])))
df = pd.DataFrame(rows); df.to_csv(f"{OUT}/per_run.csv", index=False)
agg = df.groupby(["mode", "protocol", "subset", "labels"]).agg(n_runs=("seed", "nunique"), pos=("pos", "mean"), prev=("prev", "mean"),
        AUC_PR=("AUC_PR", "mean"), AUC_PR_sd=("AUC_PR", "std"), AUC_ROC=("AUC_ROC", "mean"), lift=("lift", "mean")).reset_index()
agg.to_csv(f"{OUT}/summary.csv", index=False)
json.dump(agree, open(f"{OUT}/agreement.json", "w"), indent=1)
pd.set_option("display.width", 220)
for proto in ("known", "natural", "k1"):
    print(f"\n=== AUC-PR moyenne (ecart-type) | protocole {proto} | etiquettes v21 -> strict ===")
    for mode in ("full", "no_fraud"):
        for sub in ("all", "D5", "D6"):
            a = agg[(agg["mode"] == mode) & (agg["protocol"] == proto) & (agg["subset"] == sub)].set_index("labels")
            if len(a) == 2:
                print(f"{mode:9s} {sub:4s} v21: {a.loc['v21','AUC_PR']:.3f} ({a.loc['v21','AUC_PR_sd']:.3f}) pos {a.loc['v21','pos']:.0f} | strict: {a.loc['strict','AUC_PR']:.3f} ({a.loc['strict','AUC_PR_sd']:.3f}) pos {a.loc['strict','pos']:.0f} | ROC {a.loc['v21','AUC_ROC']:.3f} -> {a.loc['strict','AUC_ROC']:.3f}")
print("\nEcrit :", OUT)
