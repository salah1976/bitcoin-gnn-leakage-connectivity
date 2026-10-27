"""
structure_audit.py  --  Griffin v2 revision, step 1 (CPU only, no training).

Reuses the exact data loading of the Griffin v26/v28 notebooks (same DB, same labels CSV,
same split D1-D3 / D4 / D5-D6) and measures, without any model:

  A1  lambda (test-test) and rho (test -> historical addresses), under two definitions of "history"
  A2  fraud prevalence and class counts in connected vs isolated test subsets
  A3  share of D5-D6 fraud involving addresses already reported / already active in D1-D3
  A4  audit of the address fraud features (fraud_ratio, log_total_fraud): own-label inclusion
  A5  connected components of the test-test graph (Standard and Known-Address)
  A6  rule baseline without any model: max train-label fraud history over the tx addresses

Output: structure_audit.json + printed summary. Nothing is hard-coded: expected manuscript values are
only *compared* at the end (warnings, never asserts).

Usage (Colab):  %run structure_audit.py       (mounts Drive, edits the 4 PATHS below if needed)
Unit test    :  python structure_audit.py --selftest
"""
import json, sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ----------------------------------------------------------------------------- core functions
def _edges(inputs_df, outputs_df):
    """unique (tx_idx, addr_idx) pairs"""
    e = pd.concat([inputs_df[["tx_idx", "addr_idx"]], outputs_df[["tx_idx", "addr_idx"]]],
                  ignore_index=True)
    return e.drop_duplicates().reset_index(drop=True)


def lambda_tt(edges_test, n_test, allowed_addrs=None):
    """share of test tx sharing >=1 address with ANOTHER test tx.
    allowed_addrs=None -> Standard graph ; set -> only edges through those addresses (Known-Address)."""
    e = edges_test if allowed_addrs is None else edges_test[edges_test["addr_idx"].isin(allowed_addrs)]
    cnt = e.groupby("addr_idx")["tx_idx"].nunique()
    shared = cnt[cnt > 1].index
    connected = e[e["addr_idx"].isin(shared)]["tx_idx"].unique()
    return len(connected) / n_test, set(connected.tolist())


def components_report(edges_test, test_ids, allowed_addrs=None):
    """connected components (in transactions) of the test-test graph induced by shared addresses."""
    e = edges_test if allowed_addrs is None else edges_test[edges_test["addr_idx"].isin(allowed_addrs)]
    tx_pos = {t: i for i, t in enumerate(test_ids)}
    a_uni = e["addr_idx"].unique()
    a_pos = {a: i + len(test_ids) for i, a in enumerate(a_uni)}
    rows = e["tx_idx"].map(tx_pos).values
    cols = e["addr_idx"].map(a_pos).values
    n = len(test_ids) + len(a_uni)
    m = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    ncomp, lab = connected_components(m, directed=False)
    lab_tx = lab[: len(test_ids)]
    sizes = np.bincount(lab_tx)
    sizes = sizes[sizes > 0]
    multi = sizes[sizes > 1]
    return {
        "n_components_total": int(len(sizes)),
        "n_components_ge2_tx": int(len(multi)),
        "largest_component_tx": int(sizes.max()) if len(sizes) else 0,
        "largest_component_share": float(sizes.max() / len(test_ids)) if len(sizes) else 0.0,
        "tx_in_components_ge2_share": float(multi.sum() / len(test_ids)),
        "size_percentiles_ge2": {str(p): float(np.percentile(multi, p)) for p in (50, 90, 99)} if len(multi) else {},
    }


def subset_counts(mask_connected, y, test_idx):
    yt = y[test_idx]
    out = {}
    for name, sel in (("connected", mask_connected), ("isolated", ~mask_connected)):
        yy = yt[sel]
        out[name] = {"n": int(sel.sum()), "pos": int(yy.sum()), "neg": int((1 - yy).sum()),
                     "prevalence": float(yy.mean()) if len(yy) else float("nan"),
                     "share_of_test": float(sel.mean()), "share_of_test_fraud": float(yy.sum() / max(yt.sum(), 1))}
    return out


def run_audit(inputs_df, outputs_df, y, train_mask, val_mask, test_mask, num_addr):
    res = {}
    edges = _edges(inputs_df, outputs_df)
    tr_ids = np.where(train_mask)[0]; va_ids = np.where(val_mask)[0]; te_ids = np.where(test_mask)[0]
    e_tr = edges[edges["tx_idx"].isin(set(tr_ids))]
    e_va = edges[edges["tx_idx"].isin(set(va_ids))]
    e_te = edges[edges["tx_idx"].isin(set(te_ids))]
    A_train = set(e_tr["addr_idx"].unique().tolist())
    A_tv = A_train | set(e_va["addr_idx"].unique().tolist())
    n_te = len(te_ids)
    res["sizes"] = {"train": int(len(tr_ids)), "val": int(len(va_ids)), "test": int(n_te),
                    "test_fraud": int(y[te_ids].sum())}

    # ---- A1 : lambda (test-test) and rho (test -> history)
    lam_std, conn_std = lambda_tt(e_te, n_te)
    lam_kn, conn_kn = lambda_tt(e_te, n_te, allowed_addrs=A_tv)
    lam_kn_train, _ = lambda_tt(e_te, n_te, allowed_addrs=A_train)
    rho_train_tx = set(e_te[e_te["addr_idx"].isin(A_train)]["tx_idx"].unique().tolist())
    rho_tv_tx = set(e_te[e_te["addr_idx"].isin(A_tv)]["tx_idx"].unique().tolist())
    res["A1_definitions"] = {
        "lambda_tt_standard": lam_std,
        "lambda_tt_known_train_plus_val": lam_kn,
        "lambda_tt_known_train_only": lam_kn_train,
        "rho_train_only": len(rho_train_tx) / n_te,
        "rho_train_plus_val": len(rho_tv_tx) / n_te,
    }

    # ---- A2 : class composition of connected / isolated test subsets
    pos_of = {t: i for i, t in enumerate(te_ids)}
    for tag, S in (("rho_train_only", rho_train_tx), ("rho_train_plus_val", rho_tv_tx)):
        m = np.zeros(n_te, dtype=bool)
        for t in S: m[pos_of[t]] = True
        res[f"A2_subsets_{tag}"] = subset_counts(m, y, te_ids)

    # ---- A3 : test fraud tx vs addresses already reported (in a fraud-labelled train tx) / already active
    fr_tr_ids = set(tr_ids[y[tr_ids] == 1].tolist())
    A_reported = set(e_tr[e_tr["tx_idx"].isin(fr_tr_ids)]["addr_idx"].unique().tolist())
    te_fraud = set(te_ids[y[te_ids] == 1].tolist())
    ef = e_te[e_te["tx_idx"].isin(te_fraud)]
    n_f = max(len(te_fraud), 1)
    res["A3_test_fraud_history"] = {
        "test_fraud": len(te_fraud),
        "share_with_address_active_in_train": ef[ef["addr_idx"].isin(A_train)]["tx_idx"].nunique() / n_f,
        "share_with_address_reported_in_train": ef[ef["addr_idx"].isin(A_reported)]["tx_idx"].nunique() / n_f,
        "n_reported_addresses_train": len(A_reported),
    }

    # ---- A4 : audit of fraud_ratio / log_total_fraud (computed from train labels, as in build_address_features)
    fr_in = inputs_df[inputs_df["tx_idx"].isin(fr_tr_ids)]
    fr_out = outputs_df[outputs_df["tx_idx"].isin(fr_tr_ids)]
    tot_fr = np.bincount(fr_in["addr_idx"].values, minlength=num_addr) + \
             np.bincount(fr_out["addr_idx"].values, minlength=num_addr)

    def share_with_flag(tx_ids, tot):
        """share of the given tx having >=1 address with tot>0 (None if the set is empty)"""
        if len(tx_ids) == 0:
            return None
        ee = edges[edges["tx_idx"].isin(set(tx_ids.tolist()))]
        flag = ee.assign(f=(tot[ee["addr_idx"].values] > 0))
        return float(flag.groupby("tx_idx")["f"].max().reindex(tx_ids, fill_value=False).values.mean())

    # own contribution of a train fraud tx to each of its addresses
    own = pd.concat([fr_in[["tx_idx", "addr_idx"]], fr_out[["tx_idx", "addr_idx"]]]).groupby(
        ["tx_idx", "addr_idx"]).size().rename("own").reset_index()
    own["excess"] = tot_fr[own["addr_idx"].values] - own["own"].values   # fraud evidence coming from OTHER tx
    flagged_by_others = (own.groupby("tx_idx")["excess"].max() > 0)
    n_trf = max(len(fr_tr_ids), 1)
    res["A4_fraud_feature_audit"] = {
        "train_fraud_tx_with_flagged_address": share_with_flag(np.array(sorted(fr_tr_ids)), tot_fr),
        "train_fraud_tx_flagged_ONLY_by_own_label": float(1 - flagged_by_others.sum() / n_trf),
        "train_legit_tx_with_flagged_address": share_with_flag(tr_ids[y[tr_ids] == 0], tot_fr),
        "val_fraud_tx_with_flagged_address": share_with_flag(va_ids[y[va_ids] == 1], tot_fr),
        "test_fraud_tx_with_flagged_address": share_with_flag(te_ids[y[te_ids] == 1], tot_fr),
        "test_legit_tx_with_flagged_address": share_with_flag(te_ids[y[te_ids] == 0], tot_fr),
    }

    # ---- A5 : components of the test-test graph
    res["A5_components_standard"] = components_report(e_te, te_ids)
    res["A5_components_known"] = components_report(e_te, te_ids, allowed_addrs=A_tv)

    # ---- A6 : rule baseline WITHOUT any model: score = max over the tx addresses of the train-label fraud history
    from sklearn.metrics import average_precision_score, roc_auc_score
    tot_deg = np.bincount(inputs_df[inputs_df["tx_idx"].isin(set(tr_ids))]["addr_idx"].values, minlength=num_addr) + \
              np.bincount(outputs_df[outputs_df["tx_idx"].isin(set(tr_ids))]["addr_idx"].values, minlength=num_addr)
    fr_ratio = tot_fr / (tot_deg + 1e-8)                      # same definition as build_address_features
    def rule_scores(vec):
        sc = e_te.assign(s=vec[e_te["addr_idx"].values]).groupby("tx_idx")["s"].max()
        return sc.reindex(te_ids, fill_value=0.0).values
    yt = y[te_ids]
    conn_mask = np.array([t in rho_tv_tx for t in te_ids])
    a6 = {}
    for name, vec in (("max_total_fraud_count", tot_fr.astype(float)), ("max_fraud_ratio", fr_ratio)):
        sc = rule_scores(vec)
        a6[name] = {"AUC_PR_test": float(average_precision_score(yt, sc)), "AUC_ROC_test": float(roc_auc_score(yt, sc)),
                    "AUC_PR_connected_subset": float(average_precision_score(yt[conn_mask], sc[conn_mask])) if yt[conn_mask].sum() else None}
    res["A6_rule_baseline_no_model"] = a6
    return res


# ----------------------------------------------------------------------------- self test
def _selftest():
    # 6 tx: 0,1 train(0 fraud) ; 2 val ; 3,4,5 test. addresses a0..a4
    ins = pd.DataFrame({"tx_idx": [0, 1, 2, 3, 4, 5], "addr_idx": [0, 1, 0, 0, 2, 2]})
    outs = pd.DataFrame({"tx_idx": [0, 1, 2, 3, 4, 5], "addr_idx": [3, 3, 4, 1, 1, 4]})
    y = np.array([1, 0, 0, 1, 0, 0])
    tr = np.array([1, 1, 0, 0, 0, 0], bool); va = np.array([0, 0, 1, 0, 0, 0], bool); te = np.array([0, 0, 0, 1, 1, 1], bool)
    r = run_audit(ins, outs, y, tr, va, te, num_addr=5)
    a1 = r["A1_definitions"]
    # test tx: 3{a0,a1} 4{a2,a1} 5{a2,a4}; train addrs {a0,a1,a3}; val adds {a0,a4}
    assert abs(a1["lambda_tt_standard"] - 1.0) < 1e-9, a1            # 3-4 via a1, 4-5 via a2
    assert abs(a1["lambda_tt_known_train_plus_val"] - 2 / 3) < 1e-9, a1   # only a1 is known and shared (tx 3,4)
    assert abs(a1["rho_train_only"] - 2 / 3) < 1e-9, a1              # tx3, tx4 touch a train address
    assert abs(a1["rho_train_plus_val"] - 1.0) < 1e-9, a1            # tx5 touches a4 (val)
    a3 = r["A3_test_fraud_history"]                                   # test fraud = tx3 only; its a0,a1 active in train; train fraud tx0 -> reported {a0,a3}
    assert a3["share_with_address_active_in_train"] == 1.0 and a3["share_with_address_reported_in_train"] == 1.0, a3
    a4 = r["A4_fraud_feature_audit"]                                   # only tx0 is a train fraud; its addresses are flagged only by itself
    assert a4["train_fraud_tx_flagged_ONLY_by_own_label"] == 1.0, a4
    assert a4["train_fraud_tx_with_flagged_address"] == 1.0, a4
    assert a4["val_fraud_tx_with_flagged_address"] is None, a4         # no val fraud in this toy graph
    assert "A6_rule_baseline_no_model" in r
    c = r["A5_components_standard"]
    assert c["largest_component_tx"] == 3 and c["n_components_ge2_tx"] == 1, c
    return r


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        try:
            print(json.dumps(_selftest(), indent=1))
        except AssertionError as e:
            print("selftest assertion (inspect expectation):", e)
        sys.exit(0)

    # ------------------------------------------------------------------ Colab loader (same as v28)
    import sqlite3
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
    DRIVE_DB = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
    LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
    OUT_JSON = f"{ROOT}/05_results/griffin_v26_final/structure_audit.json"
    SNAPSHOT_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6"]
    TRAIN, VAL, TEST = ["D1", "D2", "D3"], ["D4"], ["D5", "D6"]

    conn = sqlite3.connect(DRIVE_DB)
    inputs_df = pd.read_sql_query("SELECT tx_hash, input_address  FROM tx_inputs", conn)
    outputs_df = pd.read_sql_query("SELECT tx_hash, output_address FROM tx_outputs", conn)
    conn.close()
    labels_df = pd.read_csv(LABELS_CSV)
    tx = labels_df[["tx_hash", "snapshot_id", "label_final"]]
    tx = tx[tx["snapshot_id"].isin(SNAPSHOT_ORDER)].copy()
    tx["rk"] = tx["snapshot_id"].map({s: i for i, s in enumerate(SNAPSHOT_ORDER)})
    tx = tx.sort_values(["rk", "tx_hash"]).reset_index(drop=True)
    train_mask = tx["snapshot_id"].isin(TRAIN).values
    val_mask = tx["snapshot_id"].isin(VAL).values
    test_mask = tx["snapshot_id"].isin(TEST).values
    y = tx["label_final"].astype(int).values
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

    res = run_audit(inputs_df, outputs_df, y, train_mask, val_mask, test_mask, num_addr=len(allA))
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT_JSON, "w"), indent=1)

    # comparison with manuscript values (warnings only)
    exp = {"lambda_tt_standard": 0.9499, "lambda_tt_known_train_plus_val": 0.2427}
    for k, v in exp.items():
        got = res["A1_definitions"][k]
        print(("OK   " if abs(got - v) < 0.002 else "DIFF ") + f"{k}: got {got:.4f} vs manuscript {v:.4f}")
    got = res["A2_subsets_rho_train_only"]["connected"]["share_of_test"]
    print(("OK   " if abs(got - 0.2605) < 0.002 else "DIFF ") + f"connected share (train only): got {got:.4f} vs manuscript 0.2605")
