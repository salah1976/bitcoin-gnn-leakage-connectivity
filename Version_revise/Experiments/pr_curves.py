# ============================================================
# pr_curves.py -- courbes precision-rappel des GNN a partir des fichiers test_probs_*_infer.npz (CPU, sans entrainement)
# Lit  <ROOT>/05_results/runs/*_infer/test_probs_<mode>_seed<N>_infer.npz  (cles : y, p_known, p_natural, p_k1, th_f1, tx_hash, snap)
# Figure : 3 panneaux (Known-Address, natural, K=1) ; courbes moyennes sur les graines (bande = min-max) pour full et no_fraud,
#          ligne horizontale = prevalence. Les points de fonctionnement a seuil de validation sont marques.
# Sorties : 05_results/pr_curves/{pr_curves.pdf, pr_curves.png, pr_points.csv}.  Test sans donnees : python pr_curves.py --selftest
# ============================================================
import os, sys, glob, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
warnings.filterwarnings("ignore")
GRID = np.linspace(0, 1, 201)

def curve_on_grid(y, p):
    pr, rc, _ = precision_recall_curve(y, p)           # rc decroissant
    order = np.argsort(rc); rc, pr = rc[order], pr[order]
    # precision interpolee "enveloppe" : max de la precision pour un rappel >= r
    env = np.maximum.accumulate(pr[::-1])[::-1]
    return np.interp(GRID, rc, env, right=env[-1])

def op_point(y, p, th):
    pred = p >= th; tp = int((pred & (y == 1)).sum())
    return tp / max(int(pred.sum()), 1), tp / max(int((y == 1).sum()), 1)

def make(files, out_dir):
    os.makedirs(out_dir, exist_ok=True); C, pts, prev = {}, [], []
    for f in files:
        z = np.load(f, allow_pickle=True); tag = os.path.basename(f).replace("test_probs_", "").replace("_infer.npz", ""); mode, seed = tag.rsplit("_seed", 1)
        y = z["y"].astype(int); th = float(z["th_f1"]); prev.append(y.mean())
        for proto in ("known", "natural", "k1"):
            p = z["p_" + proto]; C.setdefault((mode, proto), []).append(curve_on_grid(y, p))
            pr, rc = op_point(y, p, th); pts.append(dict(mode=mode, seed=int(seed), protocol=proto, AUC_PR=average_precision_score(y, p), precision=pr, recall=rc))
    prev = float(np.mean(prev)); fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    col = {"full": "#1f4e79", "no_fraud": "#c55a11"}; ttl = {"known": "Known-Address", "natural": "Natural graph", "k1": "Top-K, K=1 (no shared address)"}
    for a, proto in zip(ax, ("known", "natural", "k1")):
        for mode in ("full", "no_fraud"):
            if (mode, proto) not in C: continue
            M = np.array(C[(mode, proto)]); a.plot(GRID, M.mean(0), color=col[mode], lw=1.8, label=f"Griffin, {mode.replace('_', ' ')}")
            a.fill_between(GRID, M.min(0), M.max(0), color=col[mode], alpha=0.15, lw=0)
        a.axhline(prev, color="grey", ls="--", lw=1, label=f"prevalence {prev:.2%}"); a.set_title(ttl[proto], fontsize=10); a.set_xlabel("Recall"); a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.grid(alpha=0.25)
    ax[0].set_ylabel("Precision"); ax[0].legend(fontsize=8, loc="lower left"); fig.tight_layout()
    fig.savefig(f"{out_dir}/pr_curves.pdf"); fig.savefig(f"{out_dir}/pr_curves.png", dpi=200)
    df = pd.DataFrame(pts); df.to_csv(f"{out_dir}/pr_points.csv", index=False)
    print(df.groupby(["mode", "protocol"])[["AUC_PR", "precision", "recall"]].agg(["mean", "std"]).round(3).to_string()); print("prevalence", round(prev, 4), "| ecrit :", out_dir)

if "--selftest" in sys.argv:
    import tempfile
    d = tempfile.mkdtemp(); rng = np.random.default_rng(0); files = []
    for mode in ("full", "no_fraud"):
        for s in (1, 2):
            y = (rng.random(3000) < 0.03).astype(int); f = f"{d}/test_probs_{mode}_seed{s}_infer.npz"
            np.savez(f, y=y, th_f1=0.5, p_known=np.clip(y * 0.6 + rng.random(3000) * 0.5, 0, 1), p_natural=np.clip(y * 0.4 + rng.random(3000) * 0.6, 0, 1), p_k1=rng.random(3000)); files.append(f)
    make(files, d + "/out"); assert os.path.exists(d + "/out/pr_curves.pdf"); print("selftest OK"); sys.exit(0)

try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception:
    pass
ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
files = sorted(glob.glob(f"{ROOT}/05_results/runs/*_infer/test_probs_*_infer.npz"))
if not files: raise SystemExit("STOP : aucun fichier test_probs_*_infer.npz sous " + ROOT + "/05_results/runs")
print(len(files), "fichiers"); make(files, f"{ROOT}/05_results/pr_curves")
