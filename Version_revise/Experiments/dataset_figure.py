# ============================================================
# dataset_figure.py -- figure "BitFraud en un coup d'oeil" + table de controle (CPU, aucun modele)
# Calcule tout depuis dataset_v21_full.csv, la base d'aretes et le fichier des adresses signalees :
#   (a) chronologie des 6 snapshots (dates reelles, duree, nb tx, taux de positifs, role train/val/test)
#   (b) positifs par snapshot : adresse signalee la plus active vs autres adresses
#   (c) concentration : part des positifs de test portee par les N adresses signalees les plus actives
#   (d) decalage de signalement : part des tx liees a une adresse signalee mais etiquetees 0 (signalement posterieur)
# Sorties (05_results/dataset_figure/) : dataset_figure.pdf (vectoriel), dataset_figure.png (600 dpi),
#   dataset_table.csv, dataset_figure.json.   Le tableau est AUSSI imprime : colle-le moi pour controle (somme des tx).
# Utilisation Colab :  %run /content/drive/MyDrive/bitcoin_snapshots_project/scripts/dataset_figure.py
# ============================================================
import os, glob, json, shutil, sqlite3, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates, matplotlib.ticker
warnings.filterwarnings("ignore")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
DRIVE_DB = os.environ.get("BITFRAUD_DB", f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db")
LOCAL_DB = os.environ.get("BITFRAUD_LOCALDB", "/content/all_snapshots_v21.db")
LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
OUT_DIR = f"{ROOT}/05_results/dataset_figure"
SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]
SPLIT = {"D1": "train", "D2": "train", "D3": "train", "D4": "val", "D5": "test", "D6": "test"}
START_BLOCK = {"D1": 875000, "D2": 882200, "D3": 888000, "D4": 896500, "D5": 905500, "D6": 914500}   # table du manuscrit
os.makedirs(OUT_DIR, exist_ok=True)

cands = sorted(glob.glob(f"{ROOT}/**/scam_addresses_v21_frozen.csv", recursive=True)) or \
        sorted(glob.glob(f"{ROOT}/**/scam_addresses_v2*_frozen.csv", recursive=True))
if not cands:
    raise SystemExit("STOP : scam_addresses_v21_frozen.csv introuvable sous " + ROOT)
SCAM_CSV = cands[0]; print("Adresses signalees :", SCAM_CSV)
if not os.path.exists(LOCAL_DB): shutil.copy2(DRIVE_DB, LOCAL_DB)

# ---------- adresses signalees (premier signalement par adresse, comme les autres scripts) ----------
sc = pd.read_csv(SCAM_CSV)
sc["address"] = sc["address"].astype(str).str.strip().str.lower()
sc["report_dt"] = pd.to_datetime(sc["created_at"], utc=True, errors="coerce")
sc = sc.dropna(subset=["address", "report_dt"])
first = sc.sort_values("report_dt").groupby("address").first()

# ---------- transactions ----------
lab = pd.read_csv(LABELS_CSV, usecols=["tx_hash", "snapshot_id", "datetime", "label_final"])
lab = lab[lab["snapshot_id"].isin(SNAP)].reset_index(drop=True)
lab["dt"] = pd.to_datetime(lab["datetime"], utc=True)
pos = pd.Series(np.arange(len(lab)), index=lab["tx_hash"])
y = lab["label_final"].values.astype(int)
snap_arr = lab["snapshot_id"].values
te = np.isin(snap_arr, ["D5", "D6"])

conn = sqlite3.connect(LOCAL_DB)
ins = pd.read_sql_query("SELECT tx_hash, input_address AS addr FROM tx_inputs", conn).dropna()
outs = pd.read_sql_query("SELECT tx_hash, output_address AS addr FROM tx_outputs", conn).dropna()
conn.close()
E = pd.concat([ins, outs], ignore_index=True); del ins, outs
E = E[E["tx_hash"].isin(pos.index)].copy()
E["addr"] = E["addr"].astype(str).str.strip().str.lower()
E["t"] = E["tx_hash"].map(pos).astype(np.int64)

Es = E[E["addr"].isin(first.index)].drop_duplicates(["t", "addr"]).copy()
del E
Es["valid"] = (Es["addr"].map(first["report_dt"]).values <= lab["dt"].values[Es["t"].values])
V = Es[Es["valid"]]
assert y[V["t"].unique()].min() == 1, "une tx valide devrait etre positive"

# ---------- table par snapshot ----------
g = lab.groupby("snapshot_id")["dt"].agg(["min", "max"]).reindex(SNAP)
tab = pd.DataFrame({"snapshot": SNAP, "start_block": [START_BLOCK[s] for s in SNAP],
                    "start_utc": g["min"].dt.strftime("%Y-%m-%d %H:%M").values,
                    "end_utc": g["max"].dt.strftime("%Y-%m-%d %H:%M").values,
                    "duration_h": ((g["max"] - g["min"]).dt.total_seconds() / 3600).round(1).values,
                    "n_tx": [int((snap_arr == s).sum()) for s in SNAP],
                    "n_pos": [int(y[snap_arr == s].sum()) for s in SNAP], "split": [SPLIT[s] for s in SNAP]})
tab["prev_pct"] = (100 * tab["n_pos"] / tab["n_tx"]).round(2)
gap = (g["min"].shift(-1) - g["max"]).dt.total_seconds() / 86400
tab["gap_to_next_days"] = gap.round(0).values
pd.set_option("display.width", 220)
print("\n=== Table par snapshot ===\n", tab.to_string(index=False))
print(f"\nTotal tx : {tab['n_tx'].sum():,} | total positifs : {tab['n_pos'].sum():,} ({100*tab['n_pos'].sum()/tab['n_tx'].sum():.2f} %)")
print("Test (D5+D6) : tx", int(tab[tab.split == 'test']['n_tx'].sum()), "| positifs", int(tab[tab.split == 'test']['n_pos'].sum()))
print("Ecart entre snapshots (jours, fin -> debut suivant) :", [None if pd.isna(v) else int(round(v)) for v in gap.values])
print("ATTENTION : le manuscrit annonce 1 687 072 transactions ; verifie l'addition ci-dessus.")
tab.to_csv(f"{OUT_DIR}/dataset_table.csv", index=False)

# ---------- (b) adresse la plus active (rang sur tx valides, comme top_address_by_snapshot) ----------
rank = V.groupby("addr")["t"].nunique().sort_values(ascending=False)
top1 = rank.index[0]
top1_t = V.loc[V["addr"] == top1, "t"].unique()
dom = np.array([int((snap_arr[top1_t] == s).sum()) for s in SNAP])
n_pos = tab["n_pos"].values
share_dom = np.where(n_pos > 0, dom / np.maximum(n_pos, 1), 0.0)
print("\nAdresse la plus active :", top1[:10] + "..", "| tx positives par snapshot :", dict(zip(SNAP, dom.tolist())),
      "| part des positifs :", dict(zip(SNAP, np.round(share_dom, 3).tolist())))

# ---------- (c) concentration des positifs de test ----------
rk = pd.Series(np.arange(1, len(rank) + 1), index=rank.index)
Vt = V[te[V["t"].values]].copy(); Vt["rk"] = Vt["addr"].map(rk).values
minrk = Vt.groupby("t")["rk"].min().values                      # rang minimal parmi les adresses signalees valides de la tx
n_te = int(y[te].sum())
Ns = np.unique(np.round(np.logspace(0, np.log10(max(len(rank), 2)), 120)).astype(int))
cov = np.array([(minrk <= n).sum() / n_te for n in Ns])
conc = {int(n): float((minrk <= n).sum() / n_te) for n in (1, 10, 50, 100)}
print("Concentration (positifs de test portes par les N adresses les plus actives) :", {k: f"{v:.1%}" for k, v in conc.items()})
print(f"Adresses signalees actives (>=1 tx valide) : {len(rank):,}")

# ---------- (d) decalage de signalement ----------
retro = Es.groupby("t").size().index.values                       # tx touchant une adresse signalee, sans condition de date
valid_t = V["t"].unique()
r_s = np.array([int((snap_arr[retro] == s).sum()) for s in SNAP]); v_s = np.array([int((snap_arr[valid_t] == s).sum()) for s in SNAP])
late = np.where(r_s > 0, 1 - v_s / np.maximum(r_s, 1), 0.0)
print("Tx liees a une adresse signalee (toute date) :", dict(zip(SNAP, r_s.tolist())), "| etiquetees 1 :", dict(zip(SNAP, v_s.tolist())))
print("Part etiquetee 0 car signalement posterieur :", dict(zip(SNAP, np.round(late, 3).tolist())))

json.dump({"table": tab.to_dict("records"), "top_address_share_by_snapshot": dict(zip(SNAP, share_dom.tolist())),
           "concentration_test": conc, "late_report_share": dict(zip(SNAP, late.tolist())),
           "n_active_reported_addresses": int(len(rank))}, open(f"{OUT_DIR}/dataset_figure.json", "w"), indent=1, default=str)

# ============================ figure ============================
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42})
CS = {"train": "#1b4f8a", "val": "#d98e04", "test": "#c43c35"}
LAB = {"train": "Train", "val": "Validation", "test": "Test"}
fig = plt.figure(figsize=(7.2, 5.3))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.5, wspace=0.55, left=0.075, right=0.985, top=0.93, bottom=0.095)

# (a) chronologie
a = fig.add_subplot(gs[0, :])
mid = (g["min"] + (g["max"] - g["min"]) / 2).dt.tz_localize(None)
xs = mdates.date2num(mid.values)
a.axhline(0, color="#999999", lw=0.8, zorder=1)
for i, s in enumerate(SNAP):
    a.plot(xs[i], 0, marker="D", ms=7, color=CS[SPLIT[s]], mec="white", mew=0.8, zorder=3)
    a.text(xs[i], -0.34, f"{s}\n{tab.n_tx[i]:,} tx\n{tab.n_pos[i]:,} pos.\n({tab.prev_pct[i]:.2f}%)\n{tab.duration_h[i]:.1f} h",
           ha="center", va="top", fontsize=6.8, linespacing=1.25)
    if i < len(SNAP) - 1 and not pd.isna(gap.values[i]):
        a.text((xs[i] + xs[i + 1]) / 2, 0.07, f"{int(round(gap.values[i]))} d", ha="center", va="bottom", fontsize=6.6, color="#555555")
groups = {}
for i, s in enumerate(SNAP): groups.setdefault(SPLIT[s], []).append(xs[i])
pad = 9
for k, v in groups.items():
    a.plot([min(v) - pad, max(v) + pad], [0.62, 0.62], color=CS[k], lw=3, solid_capstyle="round")
    a.text((min(v) + max(v)) / 2, 0.72, LAB[k], ha="center", va="bottom", fontsize=7.6, color=CS[k], fontweight="bold")
a.set_ylim(-1.45, 1.05); a.set_xlim(xs[0] - 28, xs[-1] + 28)
a.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
a.get_yaxis().set_visible(False)
for sp in ("top", "right", "left"): a.spines[sp].set_visible(False)
a.spines["bottom"].set_position(("outward", 4))
a.text(0.0, 1.06, "(a)", transform=a.transAxes, fontsize=9, fontweight="bold", va="bottom")
a.set_title("Six snapshots of 100 consecutive blocks; gaps in days between snapshots", fontsize=8.2, loc="left", x=0.04, pad=6)

# (b) positifs par snapshot
b = fig.add_subplot(gs[1, 0]); xi = np.arange(len(SNAP))
rest = n_pos - dom
b.bar(xi, dom, color="#444444", width=0.68, label="Most active reported address")
b.bar(xi, rest, bottom=dom, color="#9db7d5", width=0.68, label="All other")
for i in xi:
    if n_pos[i] > 0: b.text(i, n_pos[i], f"{100*share_dom[i]:.0f}%", ha="center", va="bottom", fontsize=6.6)
b.set_xticks(xi); b.set_xticklabels(SNAP); b.set_ylabel("Positive transactions")
b.set_ylim(0, max(n_pos) * 1.15); b.legend(frameon=False, loc="upper left", fontsize=6.4, handlelength=1.0, borderaxespad=0.1)
b.text(0.0, 1.06, "(b)", transform=b.transAxes, fontsize=9, fontweight="bold", va="bottom")

# (c) concentration
c = fig.add_subplot(gs[1, 1])
c.step(Ns, 100 * cov, where="post", color=CS["test"], lw=1.5)
for n in (1, 10, 50):
    c.plot(n, 100 * conc[n], "o", ms=3.8, color=CS["test"], mec="white", mew=0.5)
    c.annotate(f"{n}: {100*conc[n]:.0f}%", (n, 100 * conc[n]), xytext=((5, -12) if n == 1 else (-34, 5) if n == 50 else (6, -10)), textcoords="offset points", fontsize=6.6)
c.set_xscale("log"); c.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(round(v)):,}")); c.set_ylim(0, 103); c.set_xlabel("Number of reported addresses (ranked)"); c.set_ylabel("Share of test positives (%)")
c.grid(alpha=0.25, lw=0.5)
c.text(0.0, 1.06, "(c)", transform=c.transAxes, fontsize=9, fontweight="bold", va="bottom")

# (d) decalage de signalement
d = fig.add_subplot(gs[1, 2])
d.bar(xi, 100 * late, color=[CS[SPLIT[s]] for s in SNAP], width=0.68)
for i in xi: d.text(i, 100 * late[i], f"{100*late[i]:.0f}%", ha="center", va="bottom", fontsize=6.6)
d.set_xticks(xi); d.set_xticklabels(SNAP); d.set_ylim(0, max(100 * late.max() * 1.18, 5))
d.set_ylabel("Matched, labelled 0 (%)")
d.text(0.0, 1.06, "(d)", transform=d.transAxes, fontsize=9, fontweight="bold", va="bottom")
for ax_ in (b, c, d):
    for sp in ("top", "right"): ax_.spines[sp].set_visible(False)
fig.savefig(f"{OUT_DIR}/dataset_figure.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/dataset_figure.png", dpi=600, bbox_inches="tight")
print("\nEcrit :", OUT_DIR)
