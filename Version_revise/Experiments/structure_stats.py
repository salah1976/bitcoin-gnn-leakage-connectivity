# ============================================================
# structure_stats.py -- composantes connexes, homophilie des etiquettes et statistiques de degre sous chaque protocole (CPU, aucun modele)
# Reponses visees : R1.4 / R3.4 (homophilie, taille des composantes le long du balayage), R5 (degres, nombre de voisins).
# Graphe de test = tx <-> adresse ; deux transactions de test sont voisines si elles partagent une adresse (comme lambda).
# Protocoles : natural (toutes les adresses des tx de test), known (adresses vues en D1-D4), Top-K (K = 1, 10, 100, 1000)
#              sous l'ordre d'origine et sous un ordre aleatoire (graine 0).
# Mesures par protocole :
#   lambda ; nombre de composantes contenant >= 2 tx de test ; part des tx dans la plus grande composante ; part des positifs dans cette composante ;
#   homophilie d'aretes (part des paires de tx voisines de meme etiquette ; attendu sous melange aleatoire = q^2 + (1-q)^2) ;
#   homophilie sans les 10 adresses les plus riches en positifs ;
#   part de voisins positifs vue par un positif / vue par un negatif, et leur rapport (enrichissement) ;
#   degres : adresses par tx, tx de test par adresse (quantiles, maximum, part des liens portes par l'adresse la plus connectee).
# Sorties : 05_results/structure_stats/{structure_stats.json, structure_stats.csv}.  Test sans donnees : python structure_stats.py --selftest
# ============================================================
import os, sys, json, shutil, sqlite3, warnings
import numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
warnings.filterwarnings("ignore")

def pair_stats(e, y):
    """e : DataFrame (tx_idx, addr_idx) sans doublons ; y : vecteur d'etiquettes indexe par tx_idx global."""
    if len(e) == 0: return {}
    g = e.assign(yy=y[e["tx_idx"].values]).groupby("addr_idx")["yy"].agg(m="size", p="sum")
    g = g[g["m"] > 1].astype(float)
    if len(g) == 0: return {"edge_hom": float("nan"), "edge_hom_excl_top10": float("nan"), "pos_nb_seen_by_pos": float("nan"),
                            "pos_nb_seen_by_neg": float("nan"), "enrichment": float("nan")}
    m, p = g["m"].values, g["p"].values; n = m - p
    pairs, same = m * (m - 1) / 2, p * (p - 1) / 2 + n * (n - 1) / 2
    top = np.argsort(-p)[:10]; keep = np.ones(len(g), bool); keep[top] = False
    sp = (p * (p - 1)).sum() / max((p * (m - 1)).sum(), 1e-12)            # voisins positifs vus par un positif
    sn = (n * p).sum() / max((n * (m - 1)).sum(), 1e-12)                  # voisins positifs vus par un negatif
    k = keep; spx = (p[k] * (p[k] - 1)).sum() / max((p[k] * (m[k] - 1)).sum(), 1e-12); snx = (n[k] * p[k]).sum() / max((n[k] * (m[k] - 1)).sum(), 1e-12)
    return {"pos_nb_seen_by_pos_excl_top10": float(spx), "pos_nb_seen_by_neg_excl_top10": float(snx), "enrichment_excl_top10": float(spx / snx) if snx > 0 else float("inf"),
            "edge_hom": float(same.sum() / pairs.sum()),
            "edge_hom_excl_top10": float(same[keep].sum() / max(pairs[keep].sum(), 1e-12)) if keep.any() else float("nan"),
            "pos_nb_seen_by_pos": float(sp), "pos_nb_seen_by_neg": float(sn), "enrichment": float(sp / sn) if sn > 0 else float("inf")}

def components(e, te_arr, y):
    """Composantes du graphe biparti (tx de test, adresses) ; les tx sans arete sont des singletons."""
    T = len(te_arr); pos_in = pd.Series(np.arange(T), index=te_arr)
    if len(e) == 0: return {"n_comp_ge2": 0, "largest_share": 1.0 / T, "pos_in_largest": float("nan"), "median_size_ge2": float("nan"), "largest_size": 1}
    ti = pos_in.reindex(e["tx_idx"].values).values.astype(int)
    ua, ai = np.unique(e["addr_idx"].values, return_inverse=True)
    N = T + len(ua)
    A = coo_matrix((np.ones(len(e)), (ti, T + ai)), shape=(N, N))
    _, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab[:T]); big = int(np.argmax(sizes)); ge2 = sizes[sizes >= 2]
    ytest = y[te_arr]; in_big = lab[:T] == big
    return {"n_comp_ge2": int((sizes >= 2).sum()), "largest_size": int(sizes.max()), "largest_share": float(sizes.max() / T),
            "pos_in_largest": float(ytest[in_big].sum() / max(ytest.sum(), 1)),
            "median_size_ge2": float(np.median(ge2)) if len(ge2) else float("nan")}

def degrees(e):
    if len(e) == 0: return {}
    a_per_tx = e.groupby("tx_idx").size(); t_per_a = e.groupby("addr_idx").size(); sh = t_per_a[t_per_a > 1]
    q = lambda s, v: float(np.quantile(s.values, v)) if len(s) else float("nan")
    return {"addr_per_tx_p50": q(a_per_tx, .5), "addr_per_tx_p99": q(a_per_tx, .99), "addr_per_tx_max": int(a_per_tx.max()),
            "tx_per_shared_addr_p50": q(sh, .5), "tx_per_shared_addr_p99": q(sh, .99), "tx_per_addr_max": int(t_per_a.max()),
            "top1_addr_link_share": float(t_per_a.max() / len(e)), "n_shared_addr": int(len(sh))}

def lam_of(e, n_te):
    if len(e) == 0: return 0.0
    c = e.groupby("addr_idx")["tx_idx"].nunique(); return float(e[e["addr_idx"].isin(c[c > 1].index)]["tx_idx"].nunique() / n_te)

def full_row(name, e, te_arr, y):
    q = float(y[te_arr].mean()); r = {"protocol": name, "lambda": lam_of(e, len(te_arr)), "n_edges": int(len(e)), "expected_hom_random": q * q + (1 - q) ** 2}
    r.update(components(e, te_arr, y)); r.update(pair_stats(e, y)); r.update(degrees(e)); return r

def selftest():
    rng = np.random.default_rng(0); te = np.arange(200)
    y = np.zeros(200, int); y[:20] = 1
    e = pd.DataFrame({"tx_idx": np.r_[np.arange(40), np.arange(40, 200)], "addr_idx": np.r_[np.zeros(40, int), np.arange(1, 161)]})  # un hub de 40 tx (20 positifs), 160 singletons
    r = full_row("hub", e, te, y)
    assert abs(r["lambda"] - 40 / 200) < 1e-9 and r["n_comp_ge2"] == 1 and r["largest_size"] == 40
    assert abs(r["pos_in_largest"] - 1.0) < 1e-9
    exp = (190 + 190) / 780  # 20 pos + 20 neg : C(20,2)*2 / C(40,2)
    assert abs(r["edge_hom"] - exp) < 1e-9, r["edge_hom"]
    assert r["tx_per_addr_max"] == 40 and abs(r["top1_addr_link_share"] - 40 / 200) < 1e-9
    print("selftest OK :", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in ("lambda", "edge_hom", "enrichment", "largest_share")}); sys.exit(0)

if "--selftest" in sys.argv: selftest()

try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception:
    pass
ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
DRIVE_DB = os.environ.get("BITFRAUD_DB", f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db")
LOCAL_DB = os.environ.get("BITFRAUD_LOCALDB", "/content/all_snapshots_v21.db")
LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
OUT_DIR = f"{ROOT}/05_results/structure_stats"; os.makedirs(OUT_DIR, exist_ok=True)
SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]; TRAIN_VAL = ["D1", "D2", "D3", "D4"]; TEST = ["D5", "D6"]
K_LIST = [1, 10, 100, 1000]
if not os.path.exists(LOCAL_DB): shutil.copy2(DRIVE_DB, LOCAL_DB)

lab = pd.read_csv(LABELS_CSV, usecols=["tx_hash", "snapshot_id", "label_final"]); lab = lab[lab["snapshot_id"].isin(SNAP)].copy()
lab["rk"] = lab["snapshot_id"].map({s: i for i, s in enumerate(SNAP)}); lab = lab.sort_values(["rk", "tx_hash"]).reset_index(drop=True)
y = lab["label_final"].astype(int).values; idx = pd.Series(np.arange(len(lab)), index=lab["tx_hash"])
tv_mask = lab["snapshot_id"].isin(TRAIN_VAL).values; te_mask = lab["snapshot_id"].isin(TEST).values
te_arr = np.where(te_mask)[0]; tv_ids = set(np.where(tv_mask)[0]); te_ids = set(te_arr)
print(f"Transactions {len(lab):,} | test {len(te_arr):,} | positifs de test {int(y[te_arr].sum()):,}")

conn = sqlite3.connect(LOCAL_DB)
ins = pd.read_sql_query("SELECT tx_hash, input_address AS addr FROM tx_inputs", conn).dropna()
outs = pd.read_sql_query("SELECT tx_hash, output_address AS addr FROM tx_outputs", conn).dropna(); conn.close()
E = pd.concat([ins, outs], ignore_index=True); del ins, outs
E = E[E["tx_hash"].isin(idx.index)].copy(); E["addr"] = E["addr"].astype(str)
E["tx_idx"] = idx.reindex(E["tx_hash"].values).values.astype(np.int64)
E["addr_idx"] = pd.factorize(E["addr"])[0].astype(np.int64)
E = E[["tx_idx", "addr_idx"]].drop_duplicates().reset_index(drop=True)

known = set(E[E["tx_idx"].isin(tv_ids)]["addr_idx"].values)
nat = E[E["tx_idx"].isin(te_ids)].reset_index(drop=True)                         # toutes les adresses des tx de test
kn = nat[nat["addr_idx"].isin(known)].reset_index(drop=True)                       # protocole Known-Address
cnt = kn.groupby("addr_idx")["tx_idx"].nunique(); shared = kn["addr_idx"].isin(cnt[cnt > 1].index).values
print(f"Aretes de test : natural {len(nat):,} | known {len(kn):,}")

rows = [full_row("natural", nat, te_arr, y), full_row("known", kn, te_arr, y)]
rk_orig = kn.groupby("addr_idx").cumcount().values + 1
perm = np.random.default_rng(0).permutation(len(kn)); rk_rand = np.empty(len(kn), int); rk_rand[perm] = kn.iloc[perm].reset_index(drop=True).groupby("addr_idx").cumcount().values + 1
for K in K_LIST:
    for nm, rk in (("orig", rk_orig), ("random", rk_rand)):
        rows.append(full_row(f"topK={K} ({nm})", kn[(~shared) | (rk <= K)], te_arr, y))

R = pd.DataFrame(rows); R.to_csv(f"{OUT_DIR}/structure_stats.csv", index=False)
json.dump(rows, open(f"{OUT_DIR}/structure_stats.json", "w"), indent=1, default=float)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\nTaux de positifs de test : %.4f | homophilie attendue sous melange aleatoire : %.4f" % (y[te_arr].mean(), R["expected_hom_random"].iloc[0]))
print("\n=== Composantes (graphe de test) ===")
print(R[["protocol", "lambda", "n_comp_ge2", "largest_size", "largest_share", "pos_in_largest", "median_size_ge2"]].round(4).to_string(index=False))
print("\n=== Homophilie des etiquettes ===")
print(R[["protocol", "edge_hom", "edge_hom_excl_top10", "pos_nb_seen_by_pos", "pos_nb_seen_by_neg", "enrichment", "pos_nb_seen_by_pos_excl_top10", "pos_nb_seen_by_neg_excl_top10", "enrichment_excl_top10"]].round(4).to_string(index=False))
print("\n=== Degres ===")
print(R[["protocol", "n_edges", "addr_per_tx_p50", "addr_per_tx_p99", "addr_per_tx_max", "n_shared_addr", "tx_per_shared_addr_p50", "tx_per_shared_addr_p99", "tx_per_addr_max", "top1_addr_link_share"]].round(4).to_string(index=False))
print("\nEcrit :", OUT_DIR)
