# ============================================================
# lgbm_strata_table.py -- LightGBM par strates S1 / S2 / S3 (agregation des JSON deja produits par lightgbm_revision.py ; CPU, sans reentrainement)
# S1 = au moins une adresse deja signalee ; S2 = connectee a l'historique sans adresse signalee ; S3 = isolee.
# Pour chaque variante et protocole (natural, known) : AUC-PR et AUC-ROC par strate, moyenne +- ecart-type sur les graines, avec n et nb de positifs.
# Sorties : 05_results/runs_lgbm/lgbm_strata_summary.csv + lgbm_strata_table.tex.   Test : python lgbm_strata_table.py --selftest
# ============================================================
import os, sys, glob, json, tempfile
import numpy as np, pandas as pd
STR = ["S1_reported", "S2_connected_not_reported", "S3_isolated"]

def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/*_seed*.json")):
            r = json.load(open(f))
            if "variant" not in r: continue
            for proto in ("natural", "known"):
                st = r.get(proto, {}).get("strata", {})
                for sname in STR:
                    b = st.get(sname)
                    if b: rows.append(dict(variant=r["variant"], seed=r["seed"], proto=proto, stratum=sname, n=b["n"], pos=b["pos"], AUC_PR=b.get("AUC-PR", np.nan), AUC_ROC=b.get("AUC-ROC", np.nan), src=os.path.basename(d)))
    return pd.DataFrame(rows)

def summarize(df):
    df = df.sort_values("src").drop_duplicates(["variant", "seed", "proto", "stratum"], keep="last")   # le dossier le plus recent l'emporte
    g = df.groupby(["variant", "proto", "stratum"]).agg(n=("n", "first"), pos=("pos", "first"), runs=("seed", "nunique"),
        PR=("AUC_PR", "mean"), PR_sd=("AUC_PR", "std"), ROC=("AUC_ROC", "mean"), ROC_sd=("AUC_ROC", "std")).reset_index()
    return g

def latex(g, variants=("A_tabular", "B_graph_nolabel", "B_no_prev"), proto="known"):
    L = [r"\begin{tabular}{lccc}\toprule", r"Model (LightGBM) & S1: AUC-PR & S2: AUC-ROC & S3: AUC-ROC \\", r"\midrule"]
    h = g[g["proto"] == proto]
    for v in variants:
        x = h[h["variant"] == v].set_index("stratum")
        if len(x) < 3: continue
        f = lambda s, m: "%.3f$\\pm$%.3f" % (x.loc[s, m], x.loc[s, m + "_sd"])
        L.append(f"{v.replace('_', ' ')} & {f('S1_reported', 'PR')} & {f('S2_connected_not_reported', 'ROC')} & {f('S3_isolated', 'ROC')} \\\\")
    L += [r"\bottomrule\end{tabular}"]; return "\n".join(L)

def run(dirs, out):
    df = load(dirs)
    if df.empty: raise SystemExit("STOP : aucun JSON avec 'strata' sous " + str(dirs))
    g = summarize(df); os.makedirs(out, exist_ok=True); g.to_csv(f"{out}/lgbm_strata_summary.csv", index=False)
    open(f"{out}/lgbm_strata_table.tex", "w").write(latex(g)); pd.set_option("display.width", 220)
    for proto in ("known", "natural"):
        print(f"\n=== LightGBM par strates | protocole {proto} (moyenne +- ecart-type sur les graines) ===")
        h = g[g["proto"] == proto]
        for v in sorted(h["variant"].unique()):
            x = h[h["variant"] == v].set_index("stratum")
            print(f"{v:18s} " + " | ".join(f"{s.split('_')[0]} n={int(x.loc[s, 'n']):,} pos={int(x.loc[s, 'pos']):,} PR={x.loc[s, 'PR']:.3f}+-{x.loc[s, 'PR_sd']:.3f} ROC={x.loc[s, 'ROC']:.3f}+-{x.loc[s, 'ROC_sd']:.3f}" for s in STR if s in x.index))
    print("\nEcrit :", out)

if "--selftest" in sys.argv:
    d = tempfile.mkdtemp(); rng = np.random.default_rng(0)
    for v in ("A_tabular", "B_graph_nolabel"):
        for s in (1, 2, 3):
            st = {k: {"n": 100 * (i + 1), "pos": 10 - 3 * i, "AUC-PR": float(rng.random()), "AUC-ROC": float(rng.random())} for i, k in enumerate(STR)}
            json.dump({"variant": v, "seed": s, "natural": {"strata": st}, "known": {"strata": st}}, open(f"{d}/{v}_seed{s}.json", "w"))
    run([d], d + "/out"); print("selftest OK"); sys.exit(0)

ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception: pass
dirs = sorted(glob.glob(f"{ROOT}/05_results/runs_lgbm*"))
print("Dossiers :", dirs); run(dirs, f"{ROOT}/05_results/runs_lgbm")
