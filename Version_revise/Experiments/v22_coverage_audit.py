# ============================================================
# v22_coverage_audit.py -- audit de COUVERTURE des labels v21 avec l'historique ChainAbuse complet (chainabuse_raw_v22_full_history.csv) (CPU)
# Le fichier v22 sert ici de MESURE seulement (comme un annuaire de signalements plus complet) ; il ne modifie aucun label de Griffin.
# 1) combien d'adresses du fichier v22 / v21 apparaissent dans les 6 snapshots (jointure), par snapshot ;
# 2) pour chaque transaction : positive v21 (label_final) ; "signalee v22" = au moins une adresse dont le PREMIER signalement v22 est <= date de la tx
#    (variante 'tous statuts' et variante 'trusted' seulement) ; tableau croise par jeu (train / val / test) ;
# 3) part des discordances (v21 = 0 mais signalee v22) due a la fenetre de collecte v21 (signalement v22 anterieur au 2024-01-01) ;
# 4) adresses les plus frequentes parmi les discordances.
# Sorties : 05_results/v22_coverage/{coverage_summary.csv, discordant_top_addresses.csv}.  Test : python v22_coverage_audit.py --selftest
# ============================================================
import os, sys, sqlite3, shutil
import numpy as np, pandas as pd

def crosstab(tx, E, first_any, first_trusted, v21_start="2024-01-01"):
    """tx : DataFrame (t, dt, y, split) ; E : DataFrame (t, addr) ; first_* : Series adresse -> premier signalement (UTC)."""
    e = E.merge(tx[["t", "dt"]], on="t")
    out = {}
    for nm, fs in (("any", first_any), ("trusted", first_trusted)):
        f = e["addr"].map(fs); ok = f.notna() & (f <= e["dt"])
        flag = pd.Series(False, index=tx["t"].values); flag.loc[e.loc[ok, "t"].unique()] = True
        pre = e.loc[ok & (f < pd.Timestamp(v21_start, tz="UTC")), "t"].unique()
        flag_pre = pd.Series(False, index=tx["t"].values); flag_pre.loc[pre] = True
        out[nm] = (flag, flag_pre)
    rows = []
    for sp, g in tx.groupby("split"):
        for nm, (flag, flag_pre) in out.items():
            f = flag.loc[g["t"].values].values; fp = flag_pre.loc[g["t"].values].values; y = g["y"].values == 1
            rows.append(dict(split=sp, variant=nm, n=len(g), v21_pos=int(y.sum()), v22_flagged=int(f.sum()), both=int((y & f).sum()), v21_only=int((y & ~f).sum()),
                             v22_only=int((~y & f).sum()), v22_only_pre2024=int((~y & f & fp).sum())))
    return pd.DataFrame(rows)

if "--selftest" in sys.argv:
    tx = pd.DataFrame({"t": [0, 1, 2, 3], "dt": pd.to_datetime(["2025-03-01"] * 4, utc=True), "y": [1, 0, 0, 0], "split": ["test"] * 4})
    E = pd.DataFrame({"t": [0, 1, 2, 3], "addr": ["a", "b", "c", "d"]})
    fa = pd.Series({"a": pd.Timestamp("2024-06-01", tz="UTC"), "b": pd.Timestamp("2022-05-01", tz="UTC"), "c": pd.Timestamp("2025-09-01", tz="UTC")})
    ft = pd.Series({"a": pd.Timestamp("2024-06-01", tz="UTC")})
    r = crosstab(tx, E, fa, ft); a = r[r.variant == "any"].iloc[0]
    assert (a.both, a.v21_only, a.v22_only, a.v22_only_pre2024) == (1, 0, 1, 1), a   # b signalee en 2022 (avant la fenetre v21) ; c signalee apres la tx -> non comptee
    t = r[r.variant == "trusted"].iloc[0]; assert (t.both, t.v22_only) == (1, 0)
    print("selftest OK"); sys.exit(0)

try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception: pass
ROOT = os.environ.get("BITFRAUD_ROOT", "/content/drive/MyDrive/bitcoin_snapshots_project")
DB = os.environ.get("BITFRAUD_DB", f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"); LOCAL = os.environ.get("BITFRAUD_LOCALDB", "/content/all_snapshots_v21.db")
if not os.path.exists(LOCAL): shutil.copy2(DB, LOCAL)
OUT = f"{ROOT}/05_results/v22_coverage"; os.makedirs(OUT, exist_ok=True)
SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]; SPLIT = {"D1": "train", "D2": "train", "D3": "train", "D4": "val", "D5": "test", "D6": "test"}

r22 = pd.read_csv(f"{ROOT}/01_data/raw/chainabuse/chainabuse_raw_v22_full_history.csv", low_memory=False)
r21 = pd.read_csv(f"{ROOT}/01_data/raw/chainabuse/chainabuse_raw_v21.csv", low_memory=False)
for d in (r22, r21): d["address"] = d["address"].astype(str).str.strip(); d["dt"] = pd.to_datetime(d["created_at"], utc=True, errors="coerce")
print(f"v22 : {len(r22):,} lignes, {r22.address.nunique():,} adresses distinctes, du {r22.dt.min()} au {r22.dt.max()} ; statuts : {r22['status'].astype(str).value_counts().head(4).to_dict()}")
print(f"v21 : {len(r21):,} lignes, {r21.address.nunique():,} adresses distinctes, du {r21.dt.min()} au {r21.dt.max()}")
first_any = r22.dropna(subset=["dt"]).groupby("address")["dt"].min()
first_tr = r22[r22["status"].astype(str).str.lower().str.contains("trusted") & ~r22["status"].astype(str).str.lower().str.contains("not_verified")].dropna(subset=["dt"]).groupby("address")["dt"].min()

lab = pd.read_csv(f"{ROOT}/01_data/processed/dataset_v21_full.csv", usecols=["tx_hash", "snapshot_id", "datetime", "label_final"]); lab = lab[lab["snapshot_id"].isin(SNAP)].reset_index(drop=True)
lab["dt"] = pd.to_datetime(lab["datetime"], utc=True); lab["t"] = np.arange(len(lab)); lab["y"] = lab["label_final"].astype(int); lab["split"] = lab["snapshot_id"].map(SPLIT)
c = sqlite3.connect(LOCAL)
E = pd.concat([pd.read_sql_query("SELECT tx_hash, input_address AS addr FROM tx_inputs", c), pd.read_sql_query("SELECT tx_hash, output_address AS addr FROM tx_outputs", c)]).dropna(); c.close()
E = E[E["tx_hash"].isin(set(lab["tx_hash"]))].copy(); E["addr"] = E["addr"].astype(str).str.strip()
E["t"] = pd.Series(lab["t"].values, index=lab["tx_hash"].values).reindex(E["tx_hash"].values).values; E = E[["t", "addr"]].drop_duplicates()

# ---- 1) jointure : adresses signalees presentes dans le benchmark ----
snap_of = lab.set_index("t")["snapshot_id"]; E["snap"] = E["t"].map(snap_of)
rows = []
for nm, s in (("v22 (toutes les adresses)", set(r22.address)), ("v21 brut", set(r21.address)), ("v22 trusted", set(first_tr.index))):
    m = E[E["addr"].isin(s)]
    row = dict(source=nm, adresses_fichier=len(s), adresses_dans_benchmark=m["addr"].nunique(), tx_touchees=m["t"].nunique(), part_adresses_pct=round(100 * m["addr"].nunique() / max(len(s), 1), 2))
    for sn in SNAP: row["adr_" + sn] = m[m["snap"] == sn]["addr"].nunique()
    rows.append(row)
cov = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== Jointure : adresses signalees presentes dans les 6 snapshots ==="); print(cov.to_string(index=False)); cov.to_csv(f"{OUT}/coverage_join.csv", index=False)

# ---- 2) tableau croise ----
ct = crosstab(lab[["t", "dt", "y", "split"]], E[["t", "addr"]], first_any, first_tr)
print("\n=== v21 (label_final) contre 'signalee dans v22 avant la tx' ===\n(both : les deux ; v21_only : positive v21 non retrouvee ; v22_only : signalee v22 mais label v21 = 0 ; v22_only_pre2024 : dont le 1er signalement v22 precede la fenetre v21)")
print(ct.to_string(index=False)); ct.to_csv(f"{OUT}/coverage_summary.csv", index=False)

# ---- 4) adresses les plus frequentes parmi les discordances (variante tous statuts, test) ----
e = E.merge(lab[["t", "dt", "y", "split"]], on="t"); f = e["addr"].map(first_any); d = e[(e["y"] == 0) & f.notna() & (f <= e["dt"])]
top = d.groupby("addr")["t"].nunique().sort_values(ascending=False).head(15).to_frame("tx_discordantes"); top["premier_signalement_v22"] = first_any.reindex(top.index).values
top["nb_rapports_v22"] = r22.groupby("address").size().reindex(top.index).values; top.index = [a[:10] + "..." for a in top.index]
print("\n=== Adresses les plus frequentes parmi les transactions 'signalees v22 mais label v21 = 0' ==="); print(top.to_string()); top.to_csv(f"{OUT}/discordant_top_addresses.csv")
print("\nEcrit :", OUT)
