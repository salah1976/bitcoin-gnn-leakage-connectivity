"""
plot_sweep_figure.py -- figure du balayage pour le manuscrit (qualite publication), a partir de sweep_summary.csv.
Utilisation Colab :  %run /content/drive/MyDrive/bitcoin_snapshots_project/scripts/plot_sweep_figure.py
Sorties (dossier sweep_summary/) : sweep_figure_pub.pdf (vectoriel, polices embarquees), sweep_figure_pub.png (600 dpi).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
DIR = os.environ.get("SWEEP_OUT", f"{ROOT}/05_results/sweep_summary")
CSV = os.path.join(DIR, "sweep_summary.csv")
if not os.path.exists(CSV):
    sys.exit(f"STOP : {CSV} introuvable (lance d'abord aggregate_sweep.py)")
g = pd.read_csv(CSV)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
C = {"topk_orig": "#1b4f8a", "topk_random": "#6f6f6f", "hist_keep": "#c43c35"}
M = {"topk_orig": "o", "topk_random": "s", "hist_keep": "^"}
LS = {"topk_orig": "-", "topk_random": (0, (4, 2)), "hist_keep": "-"}
NAME = {"topk_orig": "Top-$K$, original order", "topk_random": "Top-$K$, random order",
        "hist_keep": "History-preserving removal"}
TITLE = {"full": "With label-derived address attributes", "no_fraud": "Without label-derived address attributes"}
modes = [m for m in ("full", "no_fraud") if m in set(g["mode"])]
n_seeds = int(g["n"].max())

fig, ax = plt.subplots(2, len(modes), figsize=(7.2, 5.6), squeeze=False, sharey=True)
for j, mode in enumerate(modes):
    for i, (xcol, xl) in enumerate((("lam", r"$\lambda$ (test-test connectivity)"), ("rho", r"$\rho$ (historical exposure)"))):
        a = ax[i][j]
        for kind in ("hist_keep", "topk_random", "topk_orig"):
            s = g[(g["mode"] == mode) & (g["kind"] == kind)].copy()
            s["o"] = pd.to_numeric(s["param_s"].replace("inf", np.inf), errors="coerce").fillna(np.inf)
            s = s.sort_values("o")
            x, y, e = s[xcol].values, s["pr"].values, s["pr_sd"].fillna(0).values
            if kind != "hist_keep":                      # dernier segment (K=1000 -> graphe complet) en pointille : pas d'interpolation lisse
                a.plot(x[:-1], y[:-1], ls=LS[kind], color=C[kind], lw=1.3, zorder=3)
                a.plot(x[-2:], y[-2:], ls=":", color=C[kind], lw=1.0, zorder=3)
            else:
                a.plot(x, y, ls=LS[kind], color=C[kind], lw=1.3, zorder=3)
            a.fill_between(x, y - e, y + e, color=C[kind], alpha=0.10, lw=0, zorder=1)
            a.plot(x, y, ls="none", marker=M[kind], ms=3.8, color=C[kind], mec="white", mew=0.5, zorder=4)
        a.set_xlabel(xl); a.grid(alpha=0.25, lw=0.5); a.set_ylim(-0.02, 0.9)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.text(0.0, 1.02, "(" + "abcd"[i * len(modes) + j] + ")", transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")
        if i == 0: a.set_title(TITLE.get(mode, mode), pad=14)
        if j == 0: a.set_ylabel("AUC-PR (test)")
        if i == 1:
            a.axvline(0.2605, color="#c43c35", lw=0.6, ls=":", alpha=0.7)
            a.text(0.2575, 0.87, "all history kept ($\\rho=0.26$)", fontsize=6.8, color="#c43c35", ha="right", va="top")
h = [Line2D([0], [0], color=C[k], ls=LS[k] if k != "hist_keep" else "-", marker=M[k], ms=4, lw=1.3, mec="white", label=NAME[k])
     for k in ("topk_orig", "topk_random", "hist_keep")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.text(0.5, 0.045, f"Mean over {n_seeds} seeds, band = $\\pm$1 sd. Dotted segment: last Top-$K$ step to the full Known-Address graph.",
         ha="center", fontsize=6.8, color="#444444")
fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.6)
fig.savefig(f"{DIR}/sweep_figure_pub.pdf"); fig.savefig(f"{DIR}/sweep_figure_pub.png", dpi=600)
print("Figures :", f"{DIR}/sweep_figure_pub.pdf", "et .png")
