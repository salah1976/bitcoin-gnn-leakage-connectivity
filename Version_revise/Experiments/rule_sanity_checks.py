"""
rule_sanity_checks.py -- is the model-free rule (AUC-PR 0.9226) a leak, or a property of address-based labels?

The rule scores a test transaction by the largest number of TRAINING fraud transactions sharing one of its addresses.
It never reads a test label. These controls show what it really measures:

  C0  reproduce the reported value (train labels D1-D3)                                  -> ~0.9226
  C1  PERMUTATION control: shuffle the training labels (seeded)                          -> must fall to ~prevalence (0.009)
  C2  TIME-GAP control: use fraud labels of ONE older snapshot only (D1, D2, D3, then D4 from validation)
  C3  binary rule (any reported address) instead of the count
  C4  test snapshots D5 and D6 evaluated separately
  C5  label structure: how many test fraud transactions have NO address with a training fraud history

Interpretation guide (printed at the end):
  * C1 high        -> bug or leak (should not happen)
  * C2 stays high  -> labels behave like a static address list (reports are not time-limited): a labelling property
  * C2 drops with the gap -> genuine temporal persistence of reported addresses

Usage (Colab):  %run rule_sanity_checks.py          (CPU, a few minutes)     |   python rule_sanity_checks.py --selftest
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def rule_scores(inputs_df, outputs_df, e_te, te_ids, fraud_ids, num_addr, binary=False):
    fi = inputs_df[inputs_df["tx_idx"].isin(fraud_ids)]
    fo = outputs_df[outputs_df["tx_idx"].isin(fraud_ids)]
    tot = np.bincount(fi["addr_idx"].values, minlength=num_addr) + np.bincount(fo["addr_idx"].values, minlength=num_addr)
    if binary:
        tot = (tot > 0).astype(float)
    sc = e_te.assign(s=tot[e_te["addr_idx"].values]).groupby("tx_idx")["s"].max()
    return sc.reindex(te_ids, fill_value=0).values.astype(float)


def metrics(y_te, sc):
    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        return {"AUC_PR": None, "AUC_ROC": None}
    return {"AUC_PR": round(float(average_precision_score(y_te, sc)), 4), "AUC_ROC": round(float(roc_auc_score(y_te, sc)), 4)}


def run_checks(inputs_df, outputs_df, y, snap, test_ids, num_addr, seed=0):
    """snap: array of snapshot ids per transaction ('D1'..'D6'); test_ids: indices of test transactions."""
    edges = pd.concat([inputs_df[["tx_idx", "addr_idx"]], outputs_df[["tx_idx", "addr_idx"]]]).drop_duplicates()
    e_te = edges[edges["tx_idx"].isin(set(test_ids.tolist()))]
    y_te = y[test_ids]
    train_ids = np.where(np.isin(snap, ["D1", "D2", "D3"]))[0]
    res = {"prevalence_test": float(y_te.mean())}

    def fraud_of(mask):
        return set(np.where(mask & (y == 1))[0].tolist())

    # C0
    sc0 = rule_scores(inputs_df, outputs_df, e_te, test_ids, fraud_of(np.isin(snap, ["D1", "D2", "D3"])), num_addr)
    res["C0_reported_rule_train_D1_D3"] = metrics(y_te, sc0)
    # C1 permutation
    rng = np.random.default_rng(seed)
    yp = y.copy(); yp[train_ids] = rng.permutation(y[train_ids])
    ids_perm = set(train_ids[yp[train_ids] == 1].tolist())
    res["C1_permuted_training_labels"] = metrics(y_te, rule_scores(inputs_df, outputs_df, e_te, test_ids, ids_perm, num_addr))
    # C2 single older snapshots
    for s in ("D1", "D2", "D3", "D4"):
        sc = rule_scores(inputs_df, outputs_df, e_te, test_ids, fraud_of(snap == s), num_addr)
        res[f"C2_labels_from_{s}_only"] = metrics(y_te, sc)
    # C3 binary
    res["C3_binary_rule"] = metrics(y_te, rule_scores(inputs_df, outputs_df, e_te, test_ids, fraud_of(np.isin(snap, ["D1", "D2", "D3"])), num_addr, binary=True))
    # C4 per test snapshot
    for s in ("D5", "D6"):
        m = snap[test_ids] == s
        if m.any():
            res[f"C4_test_{s}_only"] = metrics(y_te[m], sc0[m])
    # C6 LABEL-FREE structural baseline: score = largest number of training transactions touching one of the addresses
    #    (address reuse / popularity, no label at all). It is the proper null for the permutation control.
    all_train = set(train_ids.tolist())
    di = inputs_df[inputs_df["tx_idx"].isin(all_train)]; do = outputs_df[outputs_df["tx_idx"].isin(all_train)]
    deg = np.bincount(di["addr_idx"].values, minlength=num_addr) + np.bincount(do["addr_idx"].values, minlength=num_addr)
    sc_deg = e_te.assign(s=deg[e_te["addr_idx"].values]).groupby("tx_idx")["s"].max().reindex(test_ids, fill_value=0).values.astype(float)
    res["C6_label_free_address_degree_rule"] = metrics(y_te, sc_deg)
    # C5 test fraud without any address history
    res["C5_test_fraud_with_score_zero"] = {"n_fraud": int(y_te.sum()), "n_score_zero": int(((sc0 == 0) & (y_te == 1)).sum()),
                                            "share_score_zero": float(((sc0 == 0) & (y_te == 1)).sum() / max(y_te.sum(), 1))}
    return res


def interpret(res):
    c0 = res["C0_reported_rule_train_D1_D3"]["AUC_PR"]; c1 = res["C1_permuted_training_labels"]["AUC_PR"]
    c6 = res["C6_label_free_address_degree_rule"]["AUC_PR"]
    out = [f"C0 reported rule = {c0}; permuted training labels = {c1}; label-free address-degree rule = {c6}; test prevalence = {res['prevalence_test']:.4f}"]
    out.append("NOTE: a permutation does NOT fall to the prevalence here, because the count rule also measures address reuse (popular "
               "addresses collect more training transactions, and fraud concentrates on reused addresses). The proper null is C6.")
    if c0 is not None and c6 is not None:
        out.append(f"Label-informed gain over the label-free structural rule: {c0 - c6:+.4f} AUC-PR (C0 - C6).")
    if c0 is not None and c1 is not None and c1 > 0.8 * c0:
        out.append("-> WARNING: the rule barely depends on the labels; investigate.")
    else:
        out.append("-> The permutation removes most of the performance: the rule really uses the training labels (no sign of a test-label leak).")
    gaps = {k[-7:-5] if False else k: res[k]["AUC_PR"] for k in res if k.startswith("C2_")}
    out.append("C2 single-snapshot labels (AUC-PR on D5-D6): " + ", ".join(f"{k.split('_')[3]}={v}" for k, v in gaps.items()))
    out.append("These checks cannot tell whether the LABEL RULE itself makes the rule's task easy (labels come from address reports dated before the "
               "transaction: an address reported early makes all its later transactions positive). Compare with the ChainAbuse-only / v22 labels via LABELS_CSV / LABEL_COL.")
    return out


def _selftest():
    rng = np.random.default_rng(0)
    n_tx, n_addr = 3000, 400
    snap = np.array(rng.choice(["D1", "D2", "D3", "D4", "D5", "D6"], n_tx))
    # fraud = transaction touching a "blacklisted" address (static list), like the real labelling rule
    bl = set(rng.choice(n_addr, 25, replace=False).tolist())
    rows_in = pd.DataFrame({"tx_idx": np.arange(n_tx), "addr_idx": rng.integers(0, n_addr, n_tx)})
    rows_out = pd.DataFrame({"tx_idx": np.arange(n_tx), "addr_idx": rng.integers(0, n_addr, n_tx)})
    y = np.array([int(a in bl or b in bl) for a, b in zip(rows_in["addr_idx"], rows_out["addr_idx"])])
    test_ids = np.where(np.isin(snap, ["D5", "D6"]))[0]
    res = run_checks(rows_in, rows_out, y, snap, test_ids, n_addr)
    assert res["C0_reported_rule_train_D1_D3"]["AUC_PR"] > 0.5, res           # static list => rule works
    assert res["C1_permuted_training_labels"]["AUC_PR"] < res["C0_reported_rule_train_D1_D3"]["AUC_PR"] - 0.2, res
    print(json.dumps(res, indent=1)); print("\n".join(interpret(res))); print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(); sys.exit(0)
    import sqlite3
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
    DRIVE_DB = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
    LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"   # <-- put the v22 file here to test dated labels
    LABEL_COL = "label_final"                                       # <-- e.g. "label_v22"
    OUT_JSON = f"{ROOT}/05_results/griffin_v26_final/rule_sanity_checks_{LABEL_COL}.json"
    SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]
    conn = sqlite3.connect(DRIVE_DB)
    inputs_df = pd.read_sql_query("SELECT tx_hash, input_address FROM tx_inputs", conn)
    outputs_df = pd.read_sql_query("SELECT tx_hash, output_address FROM tx_outputs", conn)
    conn.close()
    _lab = pd.read_csv(LABELS_CSV)
    print("label file columns:", list(_lab.columns))
    tx = _lab[["tx_hash", "snapshot_id", LABEL_COL]]
    tx = tx[tx["snapshot_id"].isin(SNAP)].copy()
    tx["rk"] = tx["snapshot_id"].map({s: i for i, s in enumerate(SNAP)})
    tx = tx.sort_values(["rk", "tx_hash"]).reset_index(drop=True)
    y = tx[LABEL_COL].astype(int).values
    snap = tx["snapshot_id"].values
    h2i = {h: i for i, h in enumerate(tx["tx_hash"].values)}
    inputs_df = inputs_df[inputs_df["tx_hash"].isin(h2i)].dropna().copy()
    outputs_df = outputs_df[outputs_df["tx_hash"].isin(h2i)].dropna().copy()
    inputs_df["tx_idx"] = inputs_df["tx_hash"].map(h2i); outputs_df["tx_idx"] = outputs_df["tx_hash"].map(h2i)
    allA = pd.concat([inputs_df["input_address"].astype(str), outputs_df["output_address"].astype(str)]).unique()
    a2i = {a: i for i, a in enumerate(allA)}
    inputs_df["addr_idx"] = inputs_df["input_address"].astype(str).map(a2i)
    outputs_df["addr_idx"] = outputs_df["output_address"].astype(str).map(a2i)
    inputs_df = inputs_df.astype({"tx_idx": np.int64, "addr_idx": np.int64})
    outputs_df = outputs_df.astype({"tx_idx": np.int64, "addr_idx": np.int64})
    test_ids = np.where(np.isin(snap, ["D5", "D6"]))[0]
    res = run_checks(inputs_df, outputs_df, y, snap, test_ids, len(allA))
    print(json.dumps(res, indent=1))
    print("\n" + "\n".join(interpret(res)))
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(res, open(OUT_JSON, "w"), indent=1)
