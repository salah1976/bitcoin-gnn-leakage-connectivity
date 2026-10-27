# ============================================================
# elliptic_gnn_multiseed.py -- Elliptic++ : 6 seeds, MEME modele evalue sous plusieurs graphes de test (GPU conseille, ~10-25 min)
# Fidele a la cellule 6 du notebook (memes donnees, split 1-29 / 30-34 / 35-49, memes MLP et GraphSAGE+gate, memes hyperparametres).
# Par seed :
#   A) MLP sans graphe
#   B) GNN entraine sur le graphe standard puis EVALUE (sans reentrainement) sous : standard / sans test-test / sans historique-test / isole
#      -> l'ecart entre protocoles ne vient alors que du graphe d'inference (meme poids)
#   C) (RETRAIN_SEPARATE=1) GNN reentraine separement pour chaque protocole, comme dans le notebook, pour mesurer le bruit entre runs
# Sorties : <OUT>/elliptic_multiseed_per_run.csv, elliptic_multiseed_summary.csv (moyenne +- ecart-type, 6 seeds) et ecarts apparies vs standard.
# ============================================================
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score
try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception:
    pass
DIR = os.environ.get("ELL_BASE", "/content/drive/MyDrive/bitcoin_snapshots_project/elliptic_plus_plus")
OUT = os.environ.get("ELL_OUT", "/content/drive/MyDrive/bitcoin_snapshots_project/results/elliptic_multiseed")
SEEDS = [42, 123, 2024, 7, 101, 202]
RETRAIN_SEPARATE = int(os.environ.get("RETRAIN_SEPARATE", "1"))
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("Device:", dev)

feat = pd.read_csv(f"{DIR}/txs_features.csv"); cls = pd.read_csv(f"{DIR}/txs_classes.csv"); edges = pd.read_csv(f"{DIR}/txs_edgelist.csv")
df = feat.merge(cls, on="txId", how="left"); df = df[df["class"].isin([1, 2])].copy(); df["label"] = (df["class"] == 1).astype(int); df = df.reset_index(drop=True)
fc = [c for c in df.columns if c.startswith("Local_feature_")] + [c for c in df.columns if ("in_BTC" in c or "out_BTC" in c)]
tr, va, te = (df[df["Time step"].isin(r)] for r in (range(1, 30), range(30, 35), range(35, 50)))
print(f"Features {len(fc)} | train {len(tr):,} val {len(va):,} test {len(te):,} (illicites test {int(te.label.sum()):,})")
idx = {t: i for i, t in enumerate(df["txId"].values)}
tri, vai, tei = (torch.tensor([idx[t] for t in d["txId"]]) for d in (tr, va, te))
te_ids, hist_ids = set(te["txId"]), set(tr["txId"]) | set(va["txId"])
Xr = df[fc].replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)
sc = StandardScaler().fit(Xr[tri.numpy()]); X = torch.tensor(sc.transform(Xr).astype(np.float32)).to(dev)
Y = torch.tensor(df["label"].values.astype(np.int64)).to(dev)

def ei(sub):
    s, d = sub["txId1"].map(idx), sub["txId2"].map(idx); m = s.notna() & d.notna(); s, d = s[m].astype(int).values, d[m].astype(int).values
    return torch.tensor(np.vstack([np.r_[s, d], np.r_[d, s]]), dtype=torch.long)
a, b = edges["txId1"], edges["txId2"]
G = {"standard": edges,
     "no_test_test": edges[~(a.isin(te_ids) & b.isin(te_ids))],
     "no_history_test": edges[~((a.isin(hist_ids) & b.isin(te_ids)) | (a.isin(te_ids) & b.isin(hist_ids)))],
     "isolated": edges[~(a.isin(te_ids) | b.isin(te_ids))]}
EI = {k: ei(v).to(dev) for k, v in G.items()}
for k, v in EI.items(): print(f"{k:16s} aretes dirigees {v.shape[1]:,}")

class MLP(nn.Module):
    def __init__(s, i, h=128, p=0.3):
        super().__init__(); s.n = nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 2))
    def forward(s, x, e=None): return s.n(x)
class SAGE(nn.Module):
    def __init__(s, i, h=128, p=0.3):
        super().__init__(); s.c1, s.c2 = SAGEConv(i, h), SAGEConv(h, h); s.n1, s.n2 = nn.LayerNorm(h), nn.LayerNorm(h)
        s.tab = nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())
        s.gate = nn.Sequential(nn.Linear(2 * h, 2), nn.Softmax(dim=-1)); s.clf = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(p), nn.Linear(h // 2, 2)); s.p = p
    def forward(s, x, e):
        h = F.gelu(s.n1(s.c1(x, e))); h = F.dropout(h, p=s.p, training=s.training); hg = F.gelu(s.n2(s.c2(h, e)))
        g = s.gate(torch.cat([hg, s.tab(x)], 1)); return s.clf(g[:, :1] * hg + g[:, 1:] * s.tab(x))

ytr = Y[tri.to(dev)].cpu().numpy(); W = torch.tensor([1.0, (ytr == 0).sum() / max((ytr == 1).sum(), 1)], dtype=torch.float32).to(dev)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def probs(m, e, ix):
    m.eval()
    with torch.no_grad(): return F.softmax(m(X, e)[ix.to(dev)], 1)[:, 1].cpu().numpy()
def train(cls_, e, seed, epochs=200, lr=1e-3):
    seed_all(seed); m = cls_(len(fc)).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4); best, st, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        m.train(); opt.zero_grad(); loss = F.cross_entropy(m(X, e)[tri.to(dev)], Y[tri.to(dev)], weight=W); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if ep % 5 == 0:
            ap = average_precision_score(Y[vai.to(dev)].cpu().numpy(), probs(m, e, vai))
            if ap > best: best, st, bad = ap, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, 0
            else: bad += 1
            if bad >= 20: break
    m.load_state_dict(st); m.to(dev); return m, best
def metr(m, e_inf, e_val):
    yv, yt = Y[vai.to(dev)].cpu().numpy(), Y[tei.to(dev)].cpu().numpy()
    pr, rc, th = precision_recall_curve(yv, probs(m, e_val, vai)); f = 2 * pr * rc / (pr + rc + 1e-12); t = float(th[np.argmax(f[:-1])])
    p = probs(m, e_inf, tei); return {"AUC_PR": average_precision_score(yt, p), "AUC_ROC": roc_auc_score(yt, p), "F1": f1_score(yt, (p >= t).astype(int))}

rows = []
for sd in SEEDS:
    print(f"\n===== seed {sd} =====")
    m, _ = train(MLP, None, sd); rows.append({"seed": sd, "mode": "MLP", "protocol": "no_graph", **metr(m, None, None)})
    g, _ = train(SAGE, EI["standard"], sd)
    for k in EI: rows.append({"seed": sd, "mode": "same_model", "protocol": k, **metr(g, EI[k], EI["standard"])})
    if RETRAIN_SEPARATE:
        for k in ["no_test_test", "no_history_test", "isolated"]:
            g2, _ = train(SAGE, EI[k], sd); rows.append({"seed": sd, "mode": "retrained", "protocol": k, **metr(g2, EI[k], EI[k])})
    print(pd.DataFrame([r for r in rows if r["seed"] == sd]).round(4).to_string(index=False))
    pd.DataFrame(rows).to_csv(f"{OUT}/elliptic_multiseed_per_run.csv", index=False)

R = pd.DataFrame(rows); S = R.groupby(["mode", "protocol"])[["AUC_PR", "AUC_ROC", "F1"]].agg(["mean", "std"]).round(4)
S.to_csv(f"{OUT}/elliptic_multiseed_summary.csv"); print("\n=== SYNTHESE (6 seeds, moyenne/ecart-type) ===\n", S.to_string())
sm = R[R["mode"] == "same_model"].pivot(index="seed", columns="protocol", values="AUC_PR")
print("\n=== ECARTS APPARIES (meme modele, AUC-PR) vs standard : moyenne +- ecart-type, min..max ===")
for k in ["no_test_test", "no_history_test", "isolated"]:
    d = sm[k] - sm["standard"]; print(f"{k:16s} {d.mean():+.4f} +- {d.std():.4f}  [{d.min():+.4f} .. {d.max():+.4f}]  seeds>0 : {(d > 0).sum()}/{len(d)}")
mlp = R[R["mode"] == "MLP"].set_index("seed")["AUC_PR"]; print(f"\nMLP AUC-PR {mlp.mean():.4f} +- {mlp.std():.4f} | GNN standard {sm['standard'].mean():.4f} +- {sm['standard'].std():.4f}")
print("Ecrit :", OUT)
