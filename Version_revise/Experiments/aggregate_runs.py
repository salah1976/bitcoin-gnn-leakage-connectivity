"""
aggregate_runs.py -- compiles every run folder (anchors_*.json + run_metrics_*.json) into
  * a per-seed table, mean +- std (sample std, ddof=1) per feature mode,
  * relative drop Standard -> Isolated per seed,
  * sign of (Known-Address minus Standard) per seed (is the 'intermediate optimum' robust ?),
  * a LaTeX table (booktabs) and a CSV.

Usage (Colab or local):
    python aggregate_runs.py  [RUNS_DIR]   (default: /content/drive/MyDrive/bitcoin_snapshots_project/05_results/runs)
Folders are named <mode>_seed<N>[_infer]; if the same (mode, seed) exists twice the trained run (no _infer) wins.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

DEFAULT_RUNS = "/content/drive/MyDrive/bitcoin_snapshots_project/05_results/runs"
# Colab/Jupyter pass "-f <kernel.json>" in sys.argv: only accept a real folder given on the command line
_args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json") and os.path.isdir(a)]
RUNS = _args[0] if _args else DEFAULT_RUNS


def _get(d, *keys):
    for k in keys:
        if d is None or k not in d: return np.nan
        d = d[k]
    return d if d is not None else np.nan


def load_rows(runs_dir):
    rows = {}; extras = {}
    for folder in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        a = glob.glob(os.path.join(folder, "anchors_*.json")); m = glob.glob(os.path.join(folder, "run_metrics_*.json"))
        if not a: continue
        an = json.load(open(a[0])); rm = json.load(open(m[0])) if m else {}
        mode, seed = an.get("feature_mode", "full"), an.get("seed")
        infer = bool(an.get("load_pt"))
        st = rm.get("strata", {})
        row = {"mode": mode, "seed": seed, "infer": infer, "folder": os.path.basename(folder),
               "known_PR": _get(an, "known_address", "AUC-PR"), "known_ROC": _get(an, "known_address", "AUC-ROC"),
               "std_PR": _get(an, "standard", "AUC-PR"), "std_ROC": _get(an, "standard", "AUC-ROC"),
               "iso_PR": _get(an, "isolated_K1", "AUC-PR"), "iso_ROC": _get(an, "isolated_K1", "AUC-ROC"),
               "conn_PR": _get(rm, "connected", "AUC-PR"), "S1_model_PR": _get(st, "S1_reported_address", "model_AUC_PR"),
               "S2_ROC": _get(st, "S2_connected_not_reported", "model_AUC_ROC"), "S2_PR": _get(st, "S2_connected_not_reported", "model_AUC_PR"), "S3_ROC": _get(st, "S3_isolated", "model_AUC_ROC"),
               "rule_PR": _get(st, "all", "rule_AUC_PR"), "S1_rule_PR": _get(st, "S1_reported_address", "rule_AUC_PR")}
        key = (mode, seed)
        for nm in ("hist_preserved_known", "hist_preserved_standard"):
            if nm in an:
                extras.setdefault(key, {})[nm + "_PR"] = _get(an, nm, "AUC-PR")
                extras.setdefault(key, {})[nm + "_ROC"] = _get(an, nm, "AUC-ROC")
        if key not in rows or (rows[key]["infer"] and not infer):   # trained run wins over inference-only
            rows[key] = row
    for key, ex in extras.items():
        if key in rows: rows[key].update(ex)
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    for c in ("hist_preserved_known_PR", "hist_preserved_known_ROC", "hist_preserved_standard_PR", "hist_preserved_standard_ROC"):
        if c not in df.columns: df[c] = np.nan
    if len(df):
        df["drop_std_to_iso_%"] = 100 * (df["std_PR"] - df["iso_PR"]) / df["std_PR"]
        df["known_minus_std"] = df["known_PR"] - df["std_PR"]
    return df.sort_values(["mode", "seed"]).reset_index(drop=True)


def summarize(df):
    cols = ["known_PR", "std_PR", "iso_PR", "drop_std_to_iso_%", "known_minus_std", "conn_PR", "S1_model_PR", "S2_ROC", "S2_PR", "S3_ROC"]
    out = []
    for mode, g in df.groupby("mode"):
        r = {"mode": mode, "n_seeds": len(g)}
        for c in cols:
            v = g[c].astype(float)
            r[c] = f"{v.mean():.4f} +- {v.std(ddof=1):.4f}" if len(v.dropna()) > 1 else (f"{v.mean():.4f}" if len(v.dropna()) else "n/a")
        out.append(r)
    return pd.DataFrame(out)


def latex_table(df, mode="full"):
    title = {"full": "with historical fraud address attributes", "no_fraud": "without historical fraud address attributes"}.get(mode, mode)
    L = ["\\begin{table}[!htbp]", "\\centering", "\\small",
         f"\\caption{{GriffinGNN across training seeds, {title}: AUC-PR under three test-graph constructions (Known-Address, natural graph, fully isolated $K{{=}}1$).}}",
         f"\\label{{tab:seeds_{mode}}}", "\\begin{tabular}{lcccc}", "\\hline", "Seed & Known-Address & Natural graph & Isolated ($K{=}1$) & Relative drop (\\%) \\\\", "\\hline"]
    g = df[df["mode"] == mode]
    for _, r in g.iterrows():
        L.append(f"{int(r['seed'])} & {r['known_PR']:.4f} & {r['std_PR']:.4f} & {r['iso_PR']:.4f} & {r['drop_std_to_iso_%']:.1f} \\\\")
    if len(g) > 1:
        f = lambda c, p=4: f"{g[c].mean():.{p}f} $\\pm$ {g[c].std(ddof=1):.{p}f}"
        L += ["\\hline", f"Mean $\\pm$ std & {f('known_PR')} & {f('std_PR')} & {f('iso_PR')} & {f('drop_std_to_iso_%', 1)} \\\\"]
    L += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)


def paired_report(df):
    """no_fraud minus full, seed by seed (only seeds present in both modes)."""
    a = df[df["mode"] == "full"].set_index("seed"); b = df[df["mode"] == "no_fraud"].set_index("seed")
    seeds = sorted(set(a.index) & set(b.index))
    if not seeds: return None
    cols = ["known_PR", "std_PR", "iso_PR", "drop_std_to_iso_%", "S1_model_PR", "S2_ROC", "S2_PR", "S3_ROC"]
    d = (b.loc[seeds, cols] - a.loc[seeds, cols])
    summ = pd.DataFrame({"mean_diff": d.mean(), "std_diff": d.std(ddof=1), "n_positive": (d > 0).sum(), "n_seeds": len(seeds)})
    return d, summ


def ci95(v):
    v = pd.Series(v).astype(float).dropna()
    n = len(v)
    if n < 2: return (np.nan, np.nan)
    try:
        from scipy.stats import t
        tc = t.ppf(0.975, n - 1)
    except Exception:
        tc = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}.get(n, 2.0)
    h = tc * v.std(ddof=1) / np.sqrt(n)
    return (v.mean() - h, v.mean() + h)


def hypothesis_report(df, lightgbm=0.7657, prevalence=0.000972):
    out = []
    a = df[df["mode"] == "full"].set_index("seed"); b = df[df["mode"] == "no_fraud"].set_index("seed")
    def need(n): return int(np.ceil(5 / 6 * n))
    for mode, g in df.groupby("mode"):
        n = len(g); k = int((g["drop_std_to_iso_%"] > 50).sum())
        out.append((f"H1 [{mode}] drop > 50% in {k}/{n} seeds (mean {g['drop_std_to_iso_%'].mean():.1f}%)", k >= need(n) and g["drop_std_to_iso_%"].mean() > 50))
    if len(b):
        n = len(b); k = int(((b["iso_PR"] < b["known_PR"]) & (b["known_PR"] < b["std_PR"])).sum())
        out.append((f"H2 [no_fraud] isolated < Known < natural in {k}/{n} seeds", k >= need(n)))
    if len(a):
        n = len(a); k = int((a["known_PR"] > a["std_PR"]).sum())
        out.append((f"H3 [full] Known > natural in {k}/{n} seeds (criterion: fewer than {need(n)})", k < need(n)))
    common = sorted(set(a.index) & set(b.index))
    if common:
        k = int((b.loc[common, "S2_ROC"] > a.loc[common, "S2_ROC"]).sum()); lift = b.loc[common, "S2_PR"].mean() / prevalence
        out.append((f"H4 S2 AUC-ROC higher without fraud attributes in {k}/{len(common)} seeds; mean S2 lift {lift:.1f}", k >= need(len(common)) and lift > 3))
    if len(b):
        n = len(b); k = int((b["std_PR"] > lightgbm).sum())
        out.append((f"H5 [no_fraud] natural-graph AUC-PR > LightGBM ({lightgbm}) in {k}/{n} seeds", k >= need(n)))
    return out


def monotone_report(df):
    out = []
    for mode, g in df.groupby("mode"):
        ok = int(((g["iso_PR"] < g["known_PR"]) & (g["known_PR"] < g["std_PR"])).sum())
        out.append(f"{mode}: isolated < Known-Address < natural graph in {ok}/{len(g)} seeds")
    return out


if __name__ == "__main__":
    # Colab: mount Drive if the runs folder is not visible (typical cause of "nothing found")
    if not os.path.isdir(RUNS):
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except Exception as e:
            print("Drive non montable ici:", e)
    print("RUNS =", RUNS, "| exists:", os.path.isdir(RUNS))
    if os.path.isdir(RUNS):
        for f in sorted(os.listdir(RUNS)):
            fp = os.path.join(RUNS, f)
            if os.path.isdir(fp):
                print("  ", f, "->", [x for x in os.listdir(fp) if x.endswith(".json")])
    df = load_rows(RUNS)
    if df.empty:
        print("\nAucun anchors_*.json trouve sous", RUNS, "(verifiez le chemin RUNS ou montez Drive).")
        sys.exit(1)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    show = ["mode", "seed", "infer", "known_PR", "std_PR", "iso_PR", "drop_std_to_iso_%", "known_minus_std", "conn_PR", "S1_model_PR", "S2_ROC", "S3_ROC"]
    print(df[show].round(4).to_string(index=False))
    print("\n=== mean +- std per mode ===")
    print(summarize(df).to_string(index=False))
    for line in monotone_report(df): print(line)
    print("\n=== 95% confidence intervals (t distribution) ===")
    for mode, g in df.groupby("mode"):
        for c in ("known_PR", "std_PR", "iso_PR", "drop_std_to_iso_%", "S2_ROC"):
            lo, hi = ci95(g[c]); print(f"{mode:9s} {c:18s} mean {g[c].mean():.4f}  CI95 [{lo:.4f}, {hi:.4f}]  (n={len(g)})")
    print("\n=== pre-specified hypotheses ===")
    for text, ok in hypothesis_report(df): print(("CONFIRMED  " if ok else "NOT MET    ") + text)
    if df["hist_preserved_known_PR"].notna().any():
        print("\n=== history-preserving isolation (test-test paths removed, train-test edges kept) ===")
        print(df[["mode", "seed", "known_PR", "hist_preserved_known_PR", "std_PR", "hist_preserved_standard_PR", "iso_PR"]].round(4).to_string(index=False))
        d2 = df.assign(d_known=df["hist_preserved_known_PR"] - df["known_PR"],
                       d_std=df["hist_preserved_standard_PR"] - df["std_PR"],
                       d_iso_vs_hp_std=df["iso_PR"] - df["hist_preserved_standard_PR"])
        print("\n--- effect of removing ONLY test-test paths (history kept) and of removing history as well (K=1) ---")
        for mode, g in d2.groupby("mode"):
            print(f"{mode:9s} AUC-PR  Known {g['known_PR'].mean():.4f} -> history-preserved {g['hist_preserved_known_PR'].mean():.4f} (mean diff {g['d_known'].mean():+.4f}, sd {g['d_known'].std(ddof=1):.4f});"
                  f"  natural {g['std_PR'].mean():.4f} -> history-preserved {g['hist_preserved_standard_PR'].mean():.4f} (mean diff {g['d_std'].mean():+.4f}, sd {g['d_std'].std(ddof=1):.4f});"
                  f"  K=1 isolation {g['iso_PR'].mean():.4f}")
    pr = paired_report(df)
    if pr is not None:
        d, summ = pr
        print("\n=== paired difference (no_fraud - full), per seed ===")
        print(d.round(4).to_string())
        print("\n=== paired summary ===")
        print(summ.round(4).to_string())
    g = df[df["mode"] == "full"]
    if len(g):
        pos = int((g["known_minus_std"] > 0).sum())
        print(f"\nKnown-Address > natural graph in {pos}/{len(g)} seeds (mean difference {g['known_minus_std'].mean():+.4f}) "
              f"-> the 'intermediate optimum' is {'robust' if pos in (0, len(g)) else 'NOT robust (sign changes across seeds)'}")
        print(f"rule baseline AUC-PR = {g['rule_PR'].dropna().iloc[0] if g['rule_PR'].notna().any() else float('nan'):.4f}; "
              f"GriffinGNN Known-Address max over seeds = {g['known_PR'].max():.4f}")
    out_dir = os.path.join(RUNS, "_aggregate"); os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "seeds_table.csv"), index=False)
    tex = "\n\n".join(latex_table(df, m) for m in sorted(df["mode"].unique()))
    open(os.path.join(out_dir, "seeds_table.tex"), "w").write(tex)
    print("\nsaved:", out_dir)
    print("\n" + tex)
