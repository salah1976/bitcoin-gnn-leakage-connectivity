"""
aggregate_sweep.py -- agrege les balayages sans reentrainement (sweep_<mode>_seed<N>_infer.json) :
moyenne +- ecart-type sur les graines, pour : Top-K ordre d'origine, Top-K ordre aleatoire, "historique conserve" (q).
Ajoute les points de reference (Known-Address complet = lambda 0.2427, rho 0.2605) a partir des anchors_*_infer.json.
Sorties (dossier 05_results/sweep_summary/) : sweep_summary.csv, sweep_tables.tex, sweep_figure.png/.pdf.
Utilisation :  %run aggregate_sweep.py      (ou colle le code dans une cellule ; marche avec des resultats PARTIELS)
"""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
RUNS = os.environ.get("SWEEP_RUNS", f"{ROOT}/05_results/runs")
OUT = os.environ.get("SWEEP_OUT", f"{ROOT}/05_results/sweep_summary")
os.makedirs(OUT, exist_ok=True)
LABEL = {"topk_orig": "Top-K (original order)", "topk_random": "Top-K (random order)", "hist_keep": "History-preserving removal"}

rows = []
for f in sorted(glob.glob(f"{RUNS}/*_infer/sweep_*_infer.json")):
    d = json.load(open(f))
    for p in d["points"]:
        rows.append({"mode": d["feature_mode"], "seed": d["seed"], "kind": p["kind"], "param": p["param"],
                     "lambda": p["lambda"], "rho": p["rho"], "AUC-PR": p["AUC-PR"], "AUC-ROC": p["AUC-ROC"], "F1": p["F1"]})
    # reference : graphe Known-Address complet (lambda 0.2427 / rho 0.2605) pris dans les ancres de la meme run
    af = os.path.join(os.path.dirname(f), os.path.basename(f).replace("sweep_", "anchors_", 1))
    if os.path.exists(af):
        a = json.load(open(af))["known_address"]
        for kind in ("topk_orig", "topk_random", "hist_keep"):
            rows.append({"mode": d["feature_mode"], "seed": d["seed"], "kind": kind,
                         "param": ("inf" if kind != "hist_keep" else 0.0), "lambda": 0.2427, "rho": 0.2605,
                         "AUC-PR": a["AUC-PR"], "AUC-ROC": a["AUC-ROC"], "F1": a["F1"]})
if not rows:
    print("Aucun fichier sweep_*_infer.json trouve dans", RUNS); sys.exit(0)
df = pd.DataFrame(rows); df["param_s"] = df["param"].astype(str)
print(f"{df['seed'].nunique()} graines, modes {sorted(df['mode'].unique())}, {len(df)} points")

def num(x):
    try: return float(x)
    except ValueError: return float("inf")

g = (df.groupby(["mode", "kind", "param_s"])
       .agg(n=("seed", "nunique"), lam=("lambda", "mean"), rho=("rho", "mean"),
            pr=("AUC-PR", "mean"), pr_sd=("AUC-PR", "std"), roc=("AUC-ROC", "mean"), f1=("F1", "mean")).reset_index())
g["order"] = g["param_s"].map(num)
g = g.sort_values(["mode", "kind", "order"]).drop(columns="order")
g.to_csv(f"{OUT}/sweep_summary.csv", index=False)
pd.set_option("display.width", 180)
for mode in sorted(g["mode"].unique()):
    print(f"\n=== {mode} : AUC-PR (moyenne +- ecart-type sur les graines) ===")
    for kind in ("topk_orig", "topk_random", "hist_keep"):
        s = g[(g["mode"] == mode) & (g["kind"] == kind)]
        print(f"  {LABEL[kind]}")
        for _, r in s.iterrows():
            print(f"    param={r['param_s']:>6s} | n={int(r['n'])} | lambda={r['lam']:.4f} rho={r['rho']:.4f} | AUC-PR={r['pr']:.3f} +- {0 if np.isnan(r['pr_sd']) else r['pr_sd']:.3f}")

# difference appariee ordre d'origine - ordre aleatoire (par graine)
print("\n=== Top-K : ordre d'origine moins ordre aleatoire (AUC-PR, apparie par graine) ===")
pv = df[df["kind"].isin(["topk_orig", "topk_random"])].pivot_table(index=["mode", "seed", "param_s"], columns="kind", values="AUC-PR").reset_index()
pv["diff"] = pv["topk_orig"] - pv["topk_random"]
for (mode, p), s in sorted(pv.groupby(["mode", "param_s"]), key=lambda kv: (kv[0][0], num(kv[0][1]))):
    if p == "inf": continue
    print(f"  {mode:9s} K={p:>5s} : {s['diff'].mean():+.4f} +- {s['diff'].std() if len(s) > 1 else 0:.4f} (n={len(s)})")

# tableau LaTeX (par mode) : K ou q, lambda, rho, AUC-PR
with open(f"{OUT}/sweep_tables.tex", "w") as fh:
    for mode in sorted(g["mode"].unique()):
        fh.write(f"% mode {mode}\n\\begin{{tabular}}{{llrrr}}\n\\toprule\nConstruction & Parameter & $\\lambda$ & $\\rho$ & AUC-PR \\\\\n\\midrule\n")
        for kind in ("topk_orig", "topk_random", "hist_keep"):
            for _, r in g[(g["mode"] == mode) & (g["kind"] == kind)].iterrows():
                par = ("$K=" + r["param_s"] + "$") if kind != "hist_keep" and r["param_s"] != "inf" else ("full" if r["param_s"] in ("inf", "0.0") else "$q=" + r["param_s"] + "$")
                sd = 0 if np.isnan(r["pr_sd"]) else r["pr_sd"]
                fh.write(f"{LABEL[kind]} & {par} & {r['lam']:.3f} & {r['rho']:.3f} & ${r['pr']:.3f}\\pm{sd:.3f}$ \\\\\n")
            fh.write("\\midrule\n")
        fh.write("\\bottomrule\n\\end{tabular}\n\n")

# figure : AUC-PR en fonction de lambda, puis de rho
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    modes = sorted(g["mode"].unique())
    fig, ax = plt.subplots(2, len(modes), figsize=(5.2 * len(modes), 7.6), squeeze=False, sharey=True)
    sty = {"topk_orig": ("o-", "#1f4e79"), "topk_random": ("s--", "#7a7a7a"), "hist_keep": ("^-", "#c0504d")}
    for j, mode in enumerate(modes):
        for i, (xcol, xl) in enumerate((("lam", r"$\lambda$ (test-test connectivity)"), ("rho", r"$\rho$ (historical exposure)"))):
            a = ax[i][j]
            for kind in ("topk_orig", "topk_random", "hist_keep"):
                s = g[(g["mode"] == mode) & (g["kind"] == kind)].sort_values(xcol)
                m, c = sty[kind]
                a.errorbar(s[xcol], s["pr"], yerr=s["pr_sd"].fillna(0), fmt=m, color=c, capsize=2, ms=4, lw=1.2, label=LABEL[kind])
            a.set_xlabel(xl); a.grid(alpha=.3)
            if j == 0: a.set_ylabel("AUC-PR (test, mean $\\pm$ sd over seeds)")
            if i == 0: a.set_title({"full": "With label-derived address attributes", "no_fraud": "Without label-derived address attributes"}.get(mode, mode), fontsize=10)
    ax[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/sweep_figure.png", dpi=200); fig.savefig(f"{OUT}/sweep_figure.pdf")
    print("\nFigure :", f"{OUT}/sweep_figure.png")
except Exception as e:
    print("Figure non produite :", e)
print("\nResultats ecrits dans", OUT)
