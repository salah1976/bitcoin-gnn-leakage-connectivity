# ============================================================
# elliptic_homophily.py -- homophilie des etiquettes sur le graphe tx-tx d'Elliptic++ (CPU, quelques secondes)
# Memes definitions que structure_stats.py (BitFraud) pour que les deux benchmarks soient comparables :
#   graphe = aretes entre transactions ETIQUETEES (classes 1 et 2), pas de temps 35-49 (test), aretes non orientees ;
#   part de voisins illicites vue par une tx illicite / par une tx licite, enrichissement, homophilie d'aretes et valeur attendue sous melange aleatoire.
# Fichiers : txs_edgelist.csv, txs_classes.csv (variable ELL_BASE).   Test sans donnees : python elliptic_homophily.py --selftest
# ============================================================
import os, sys
import numpy as np, pandas as pd

def homophily(edges, label):
    """edges : DataFrame (a, b) ; label : dict node -> 0/1. Retourne un dict de statistiques."""
    e = edges[edges["a"].isin(label) & edges["b"].isin(label)]
    e = pd.concat([e, e.rename(columns={"a": "b", "b": "a"})]).drop_duplicates()
    e["ya"] = e["a"].map(label); e["yb"] = e["b"].map(label)
    nodes = pd.Series(label); n = len(nodes); p = nodes.mean()
    g = e.groupby("a")["yb"].mean()                   # part de voisins illicites vue par chaque noeud connecte
    ya = nodes.reindex(g.index)
    ill, lic = g[ya == 1].mean(), g[ya == 0].mean()
    same = (e["ya"] == e["yb"]).mean()
    return dict(nodes=n, connected=int(len(g)), connected_pct=100 * len(g) / n, prevalence_pct=100 * p, nb_edges_undirected=len(e) // 2,
                illicit_sees_illicit_pct=100 * ill, licit_sees_illicit_pct=100 * lic, enrichment=(ill / lic if lic > 0 else float("nan")),
                edge_homophily=same, edge_homophily_expected=p ** 2 + (1 - p) ** 2)

if "--selftest" in sys.argv:
    ed = pd.DataFrame({"a": [1, 2, 3, 5], "b": [2, 3, 4, 6]}); lab = {1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 1}
    r = homophily(ed, lab); assert r["connected"] == 6 and abs(r["edge_homophily"] - 0.5) < 1e-9 and r["nb_edges_undirected"] == 4, r
    assert abs(r["illicit_sees_illicit_pct"] - 50) < 1e-9 and abs(r["licit_sees_illicit_pct"] - 50) < 1e-9 and abs(r["enrichment"] - 1) < 1e-9, r; print("selftest OK"); sys.exit(0)

try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception: pass
DIR = os.environ.get("ELL_BASE", "/content/drive/MyDrive/bitcoin_snapshots_project/elliptic_plus_plus")
feat = pd.read_csv(f"{DIR}/txs_features.csv", usecols=["txId", "Time step"]); cls = pd.read_csv(f"{DIR}/txs_classes.csv"); ed = pd.read_csv(f"{DIR}/txs_edgelist.csv").rename(columns={"txId1": "a", "txId2": "b"})
df = feat.merge(cls, on="txId"); df = df[df["class"].isin([1, 2])]
out = {}
for name, rng in (("test (ts 35-49)", range(35, 50)), ("all labelled (ts 1-49)", range(1, 50))):
    sub = df[df["Time step"].isin(rng)]; lab = dict(zip(sub["txId"], (sub["class"] == 1).astype(int))); out[name] = homophily(ed, lab)
R = pd.DataFrame(out).T; pd.set_option("display.width", 220); print(R.round(3).T.to_string())
R.to_csv(f"{DIR}/elliptic_homophily.csv"); print("Ecrit :", f"{DIR}/elliptic_homophily.csv")
