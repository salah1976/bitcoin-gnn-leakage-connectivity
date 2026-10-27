# ============================================================
# elliptic_exposure.py -- exposition historique (rho), connectivite test-test (lambda), regle d'historique et concentration sur Elliptic++ (CPU, aucun modele)
# Fichiers Elliptic++ (git-disl/EllipticPlusPlus) : txs_classes.csv (txId, class : 1 illicite, 2 licite, 3 inconnu), txs_features.csv (txId, Time step, ...),
#   txs_edgelist.csv (txId1, txId2), AddrTx_edgelist.csv (input_address, txId), TxAddr_edgelist.csv (txId, output_address).
# Protocole du manuscrit : pas de temps 1-34 = entrainement, 35-49 = test ; metriques sur les transactions etiquetees.
# Definitions (identiques a celles de BitFraud) :
#   lambda = part des tx de test partageant une adresse avec une AUTRE tx de test ; rho = part des tx de test partageant une adresse avec une tx de la periode d'entrainement.
#   regle d'historique = pour chaque tx de test, plus grand nombre de tx ILLICITES d'entrainement partageant une de ses adresses (etiquettes d'entrainement seulement).
# Sorties : <dossier>/elliptic_exposure.json (+ resume imprime).  Colle-moi toute la sortie imprimee.
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
BASE = os.environ.get("ELL_BASE", "/content/drive/MyDrive")
def find(name):
    c = sorted(glob.glob(f"{BASE}/**/{name}", recursive=True)) or sorted(glob.glob(f"{BASE}/**/{os.path.splitext(name)[0]}.txt", recursive=True))
    if not c: raise SystemExit(f"STOP : {name} introuvable sous {BASE}. Dis-moi ou se trouvent les fichiers Elliptic++ (ou fixe ELL_BASE).")
    print(f"{name:24s} -> {c[0]}"); return c[0]
f_cls, f_feat = find("txs_classes.csv"), find("txs_features.csv")
f_tt = (sorted(glob.glob(f"{BASE}/**/txs_edgelist.csv", recursive=True)) or [None])[0]
f_at, f_ta = find("AddrTx_edgelist.csv"), find("TxAddr_edgelist.csv")
OUT = os.path.join(os.path.dirname(f_cls), "elliptic_exposure.json")
def col(df, *keys):
    for c in df.columns:
        if all(k in c.lower().replace("_", "").replace(" ", "") for k in keys): return c
    raise SystemExit(f"STOP : colonne {keys} absente ; colonnes : {list(df.columns)[:8]}")

cls = pd.read_csv(f_cls); c_id, c_cl = col(cls, "txid"), col(cls, "class")
head = pd.read_csv(f_feat, nrows=2); c_fid, c_ts = col(head, "txid"), col(head, "timestep")
ft = pd.read_csv(f_feat, usecols=[c_fid, c_ts]).rename(columns={c_fid: "tx", c_ts: "ts"})
cls = cls.rename(columns={c_id: "tx", c_cl: "cls"})
tx = ft.merge(cls, on="tx", how="left"); tx["cls"] = tx["cls"].fillna(3).astype(int)
tx["lab"] = tx["cls"].map({1: 1, 2: 0}); tx["is_train"] = tx["ts"] <= 34; tx["is_test"] = tx["ts"] >= 35
print(f"\nTransactions : {len(tx):,} | pas de temps {tx.ts.min()}-{tx.ts.max()} | classes : {tx['cls'].value_counts().to_dict()} (attendu : 4 545 illicites, 42 019 licites)")
print(f"Etiquetees train : {int(tx[tx.is_train].lab.notna().sum()):,} (illicites {int(tx[tx.is_train].lab.eq(1).sum()):,}) | test : {int(tx[tx.is_test].lab.notna().sum()):,} (illicites {int(tx[tx.is_test].lab.eq(1).sum()):,})")

def rd(f):
    d = pd.read_csv(f)
    if d.shape[1] == 1: d = pd.read_csv(f, sep=None, engine="python")   # separateur inattendu (tab, espace)
    return d
at = rd(f_at); ta = rd(f_ta)
print("AddrTx colonnes :", list(at.columns), "| TxAddr colonnes :", list(ta.columns))
a_tx1, a_ad1 = col(at, "txid"), [c for c in at.columns if "address" in c.lower()][0]
a_tx2, a_ad2 = col(ta, "txid"), [c for c in ta.columns if "address" in c.lower()][0]
E = pd.concat([at[[a_tx1, a_ad1]].set_axis(["tx", "addr"], axis=1), ta[[a_tx2, a_ad2]].set_axis(["tx", "addr"], axis=1)], ignore_index=True).drop_duplicates()
E = E[E["tx"].isin(tx["tx"])]
E = E.merge(tx[["tx", "ts", "lab"]], on="tx")
print(f"Paires (tx, adresse) : {len(E):,} | adresses distinctes : {E.addr.nunique():,} | tx avec au moins une adresse : {E.tx.nunique():,} ({E.tx.nunique()/len(tx):.1%})")

# verification : aretes tx-tx qui traversent la coupure
res = {}
if f_tt:
    tt = pd.read_csv(f_tt); i1, i2 = tt.columns[:2]
    m = tt.merge(tx[["tx", "ts"]].rename(columns={"tx": i1, "ts": "ts1"}), on=i1).merge(tx[["tx", "ts"]].rename(columns={"tx": i2, "ts": "ts2"}), on=i2)
    cross = int(((m.ts1 <= 34) != (m.ts2 <= 34)).sum()); res["tx_tx_edges"] = int(len(m)); res["tx_tx_edges_crossing_split"] = cross
    print(f"Aretes tx-tx : {len(m):,} ; traversant la coupure train/test : {cross:,}")

# adresses : periodes d'activite
E["is_train"] = E.ts <= 34; E["is_test"] = E.ts >= 35
ad_train = set(E.loc[E.is_train, "addr"]); n_test_per_addr = E[E.is_test].groupby("addr")["tx"].nunique()
ad_test_multi = set(n_test_per_addr[n_test_per_addr >= 2].index)     # adresse partagee par >= 2 tx de test
Et = E[E.is_test]
rho_tx = Et.groupby("tx")["addr"].apply(lambda s: bool(set(s) & ad_train))
lam_tx = Et.groupby("tx")["addr"].apply(lambda s: bool(set(s) & ad_test_multi))
test_tx = tx[tx.is_test].set_index("tx")
test_tx["rho"] = rho_tx.reindex(test_tx.index).fillna(False); test_tx["lam"] = lam_tx.reindex(test_tx.index).fillna(False)
def share(m): return float(test_tx.loc[m, "rho"].mean()), float(test_tx.loc[m, "lam"].mean()), int(m.sum())
for name, m in (("toutes les tx de test", test_tx.index == test_tx.index), ("tx de test etiquetees", test_tx.lab.notna()),
                ("illicites de test", test_tx.lab == 1), ("licites de test", test_tx.lab == 0)):
    r, l, n = share(m); res[f"rho_lambda_{name}"] = {"rho": r, "lambda": l, "n": n}
    print(f"  {name:24s} n={n:6d} | rho = {r:6.2%} | lambda = {l:6.2%}")

# regle d'historique (etiquettes d'entrainement seulement)
Etr_ill = E[E.is_train & (E.lab == 1)]
cnt = Etr_ill.groupby("addr")["tx"].nunique()
score = Et.assign(c=Et["addr"].map(cnt).fillna(0)).groupby("tx")["c"].max()
test_tx["rule"] = score.reindex(test_tx.index).fillna(0)
L = test_tx[test_tx.lab.notna()]
y = L.lab.astype(int).values; p = L["rule"].values
res["rule"] = {"n": int(len(L)), "pos": int(y.sum()), "prev": float(y.mean()), "AUC_PR": float(average_precision_score(y, p)), "AUC_ROC": float(roc_auc_score(y, p)),
               "share_pos_with_score_gt0": float((p[y == 1] > 0).mean()), "share_neg_with_score_gt0": float((p[y == 0] > 0).mean())}
res["rule"]["lift"] = res["rule"]["AUC_PR"] / res["rule"]["prev"]
print(f"\nRegle d'historique sur Elliptic++ (test, tx etiquetees) : n={len(L):,} positifs={int(y.sum()):,} prevalence={y.mean():.2%} | AUC-PR={res['rule']['AUC_PR']:.3f} AUC-ROC={res['rule']['AUC_ROC']:.3f} lift={res['rule']['lift']:.1f}")
print(f"  score > 0 : {res['rule']['share_pos_with_score_gt0']:.1%} des positifs, {res['rule']['share_neg_with_score_gt0']:.1%} des negatifs")

# concentration : les adresses illicites d'entrainement portent-elles les positifs de test ?
Ill_te = Et[Et.lab == 1]
rank = cnt.sort_values(ascending=False)
conc = {}
for N in (1, 10, 50, 100):
    top = set(rank.head(N).index)
    conc[N] = float(Ill_te[Ill_te.addr.isin(top)]["tx"].nunique() / max(Ill_te["tx"].nunique(), 1))
res["concentration_positifs_test_par_top_adresses_illicites_train"] = conc
print("Part des positifs de test touchant l'une des N adresses les plus 'illicites' de l'entrainement :", {k: f"{v:.1%}" for k, v in conc.items()})

# profil par pas de temps de test
per = test_tx.groupby("ts").agg(n=("rho", "size"), rho=("rho", "mean"), lam=("lam", "mean"))
print("\nrho / lambda par pas de temps de test :\n", per.round(3).T.to_string())
res["par_pas_de_temps"] = per.round(4).reset_index().to_dict("records")
json.dump(res, open(OUT, "w"), indent=1, default=str); print("\nEcrit :", OUT)
