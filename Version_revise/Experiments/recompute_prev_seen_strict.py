# ============================================================
# Recalcule prev_addr_seen_ratio / prev_addr_seen_count avec un historique STRICT D1-D4 pour toutes les transactions de test.
# Version d'origine (build_temporal_addr_sets) : historique d'un snapshot S = adresses de TOUS les snapshots strictement anterieurs.
#   -> D5 : D1-D4 (identique a la version stricte) ; D6 : D1-D5, donc inclut une partie de la periode de test.
# Ici : train/val inchanges ; test (D5, D6) : historique = adresses vues en D1-D4 uniquement.
# Etape 1 : reproduit les valeurs d'origine a partir de la base d'aretes (validation contre dataset_v21_full.csv).
# Etape 2 : ecrit dataset_v21_prevstrict.csv (meme colonnes, deux colonnes remplacees pour D5-D6).
# CPU, quelques minutes. Aucune etiquette utilisee.
# ============================================================
import os, shutil, sqlite3, numpy as np, pandas as pd

SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]
TEST_RANKS = (4, 5)
STRICT_MAX_RANK = 3          # D4 = rang 3


def compute_counts(labels_tx, tx_rank, inputs, outputs, mode):
    """labels_tx : tx_hash (ordre du CSV). tx_rank : rang du snapshot de chaque tx (meme ordre).
    inputs/outputs : DataFrames (tx_hash, addr). mode : 'roles' | 'union' | 'raw' (3 conventions possibles pour n)."""
    n_tx = len(labels_tx)
    pos = pd.Series(np.arange(n_tx), index=labels_tx)
    rank_arr = np.asarray(tx_rank)
    a_in = inputs.assign(role=0); a_out = outputs.assign(role=1)
    E = pd.concat([a_in, a_out], ignore_index=True)
    E["t"] = E["tx_hash"].map(pos)
    E = E.dropna(subset=["t"]).copy()
    E["t"] = E["t"].astype(np.int64)
    E["a"] = pd.factorize(E["addr"])[0]
    E["rk"] = rank_arr[E["t"].values]
    first = pd.Series(E["rk"].values).groupby(E["a"].values).min()         # premier snapshot ou l'adresse apparait
    if mode == "roles":
        E = E.drop_duplicates(["t", "a", "role"])
    elif mode == "union":
        E = E.drop_duplicates(["t", "a"])
    f = first.reindex(E["a"].values).values
    out = {}
    for name, seen in (("orig", f < E["rk"].values), ("strict", f <= STRICT_MAX_RANK)):
        out[name] = np.bincount(E["t"].values, weights=seen.astype(float), minlength=n_tx)
    out["n"] = np.bincount(E["t"].values, minlength=n_tx).astype(float)
    return out


def ratio(seen, n):
    return np.where(n > 0, seen / np.maximum(n, 1), 0.0)


if __name__ == "__main__":
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
    DRIVE_DB = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
    LOCAL_DB = "/content/all_snapshots_v21.db"
    CSV_IN = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
    CSV_OUT = f"{ROOT}/01_data/processed/dataset_v21_prevstrict.csv"
    if not os.path.exists(LOCAL_DB): shutil.copy2(DRIVE_DB, LOCAL_DB)

    df = pd.read_csv(CSV_IN)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df[df["snapshot_id"].isin(SNAP)].copy()
    df["snapshot_rank"] = df["snapshot_id"].map({s: i for i, s in enumerate(SNAP)})
    df = df.sort_values(["snapshot_rank", "tx_hash"]).reset_index(drop=True)   # meme ordre que griffin_v28_revision.py
    print("Transactions:", len(df))
    conn = sqlite3.connect(LOCAL_DB)
    ins = pd.read_sql_query("SELECT tx_hash, input_address AS addr FROM tx_inputs", conn).dropna()
    outs = pd.read_sql_query("SELECT tx_hash, output_address AS addr FROM tx_outputs", conn).dropna()
    conn.close()
    keep = set(df["tx_hash"]); ins = ins[ins["tx_hash"].isin(keep)]; outs = outs[outs["tx_hash"].isin(keep)]
    ins["addr"] = ins["addr"].astype(str); outs["addr"] = outs["addr"].astype(str)

    best = None
    print("\n=== Validation : reproduction de la version d'origine ===")
    for mode in ("roles", "union", "raw"):
        r = compute_counts(df["tx_hash"].values, df["snapshot_rank"].values, ins, outs, mode)
        m_cnt = float((r["orig"] == df["prev_addr_seen_count"].values).mean())
        m_rat = float(np.mean(np.abs(ratio(r["orig"], r["n"]) - df["prev_addr_seen_ratio"].values) < 1e-6))
        print(f"  convention n='{mode}': count identique {m_cnt:.4%} | ratio identique {m_rat:.4%}")
        if best is None or (m_cnt + m_rat) > best[0]: best = (m_cnt + m_rat, mode, r, m_cnt, m_rat)
    _, mode, r, m_cnt, m_rat = best
    print(f"-> convention retenue : {mode}")
    if min(m_cnt, m_rat) < 0.99:
        print("\nSTOP : la reproduction de la version d'origine est inferieure a 99 % ; je n'ecris PAS de fichier. Colle-moi ces lignes."); raise SystemExit(0)

    test = df["snapshot_rank"].isin(TEST_RANKS).values
    new_cnt = df["prev_addr_seen_count"].values.astype(float).copy()
    new_rat = df["prev_addr_seen_ratio"].values.astype(float).copy()
    new_cnt[test] = r["strict"][test]
    new_rat[test] = ratio(r["strict"], r["n"])[test]
    for s in ("D5", "D6"):
        sel = (df["snapshot_id"] == s).values
        ch = np.abs(new_cnt[sel] - df["prev_addr_seen_count"].values[sel]) > 1e-9
        print(f"{s}: transactions dont le compte change : {ch.mean():.2%} | moyenne ratio avant {df['prev_addr_seen_ratio'].values[sel].mean():.4f} -> apres {new_rat[sel].mean():.4f}")
    df2 = df.copy()
    df2["prev_addr_seen_count"] = new_cnt.astype(df["prev_addr_seen_count"].dtype)
    df2["prev_addr_seen_ratio"] = new_rat
    df2.drop(columns=["snapshot_rank"]).to_csv(CSV_OUT, index=False)
    print("\nEcrit :", CSV_OUT)
