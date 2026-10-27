"""
v28_revision_patch.py  --  three SMALL insertions into the Griffin v28 cell (nothing else is modified).

WHY
 * The v28 sweep reloads  OUT_DIR/lambda_sweep_v28_checkpoint.json  if it exists, so any new seed or feature mode
   run in the same folder silently REUSES the old sweep. Patch 1 gives every run its own folder.
 * R3-2 / R4-D5 : retrain without the two label-derived address features (fraud_ratio, log1p(total_fraud)).
 * The audit showed a model-free rule (max train-label fraud count over the tx addresses) gets AUC-PR 0.9226 on D5-D6.
   Patch 3 reports model vs rule by stratum, which isolates where a model can add anything beyond address history.

HOW  (the three blocks below are copy-paste, with their anchor in v28)

----------------------------------------------------------------------------------------------------------------
PATCH 1  -- replace the lines  SEED = 999 ... OUT_DIR = f"{ROOT}/results/griffin_v26_final"  by:

    SEED = 42                       # change per run: 42, 101, 202, 303, 404
    FEATURE_MODE = "full"           # "full" | "no_fraud"
    RUN_TAG = f"{FEATURE_MODE}_seed{SEED}"
    OUT_DIR = f"{ROOT}/05_results/runs/{RUN_TAG}"
    os.makedirs(OUT_DIR, exist_ok=True)
    # and set  RESUME_FROM_CHECKPOINT = False  (before section 9) so that each run really trains

  (keep DRIVE_DB / LABELS_CSV on the corrected 01_data paths)

----------------------------------------------------------------------------------------------------------------
PATCH 2  -- right after the line   addr_x_raw = build_address_features(train_mask)   insert:

    if FEATURE_MODE == "no_fraud":
        addr_x_raw[:, [8, 9]] = 0.0   # columns 8 = fraud_ratio, 9 = log1p(total_fraud) (see np.vstack order)

----------------------------------------------------------------------------------------------------------------
PATCH 3  -- right after the block "=== STRUCTURAL DECOMPOSITION ===" (tp, tt, conn_mask already exist), insert:

    te_arr = np.where(test_mask)[0]                     # same order as tp / tt (shuffle=False)
    hist_f = set(np.where(train_mask & (y == 1))[0].tolist())
    _fi = inputs_df[inputs_df["tx_idx"].isin(hist_f)]; _fo = outputs_df[outputs_df["tx_idx"].isin(hist_f)]
    tot_fr_rule = np.bincount(_fi["addr_idx"].values, minlength=num_addr) + np.bincount(_fo["addr_idx"].values, minlength=num_addr)
    _e = pd.concat([inp_te[["tx_idx", "addr_idx"]], out_te[["tx_idx", "addr_idx"]]]).drop_duplicates()
    rule = _e.assign(s=tot_fr_rule[_e["addr_idx"].values]).groupby("tx_idx")["s"].max().reindex(te_arr, fill_value=0).values.astype(float)
    strata = strata_report(tp, tt, rule, conn_mask[te_arr])
    print(json.dumps(strata, indent=1))
    json.dump(strata, open(f"{OUT_DIR}/strata_{RUN_TAG}.json", "w"), indent=1)

  and paste the function strata_report (below) once, anywhere above that line (e.g. next to mets()).
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def strata_report(model_p, y_true, rule_score, connected):
    """Model vs model-free rule, overall and by stratum.
    S1 : tx with >=1 address already reported in a train fraud tx (rule > 0)
    S2 : connected to history (address seen in train/val) but no reported address (rule == 0)
    S3 : isolated (no address seen in train/val)
    Strata S2 and S3 are where a model must rely on something other than address history."""
    model_p = np.asarray(model_p, float); y_true = np.asarray(y_true).astype(int)
    rule_score = np.asarray(rule_score, float); connected = np.asarray(connected).astype(bool)

    def block(sel):
        yy = y_true[sel]
        out = {"n": int(sel.sum()), "pos": int(yy.sum()), "neg": int((1 - yy).sum()),
               "prevalence": float(yy.mean()) if sel.sum() else None}
        if yy.sum() > 0 and yy.sum() < len(yy):
            out["model_AUC_PR"] = round(float(average_precision_score(yy, model_p[sel])), 4)
            out["model_AUC_ROC"] = round(float(roc_auc_score(yy, model_p[sel])), 4)
            out["rule_AUC_PR"] = round(float(average_precision_score(yy, rule_score[sel])), 4)
            out["rule_AUC_ROC"] = round(float(roc_auc_score(yy, rule_score[sel])), 4)
        else:
            out["note"] = "no positives or no negatives: metrics undefined"
        return out

    s1 = rule_score > 0
    s2 = (~s1) & connected
    s3 = (~s1) & (~connected)
    assert (s1 | s2 | s3).all() and not (s1 & s2).any() and not (s2 & s3).any() and not (s1 & s3).any()
    return {"all": block(np.ones(len(y_true), bool)),
            "S1_reported_address": block(s1),
            "S2_connected_not_reported": block(s2),
            "S3_isolated": block(s3)}


if __name__ == "__main__":      # unit test on synthetic data
    rng = np.random.default_rng(0)
    n = 6000
    connected = rng.random(n) < 0.3
    rule = np.where(connected & (rng.random(n) < 0.6), rng.integers(1, 20, n), 0)
    y = ((rule > 0) & (rng.random(n) < 0.15)) | (rng.random(n) < 0.002)
    p = np.clip(0.2 * (rule > 0) + 0.05 * rng.random(n) + 0.5 * y * rng.random(n), 0, 1)
    r = strata_report(p, y.astype(int), rule, connected)
    tot = sum(r[k]["n"] for k in ("S1_reported_address", "S2_connected_not_reported", "S3_isolated"))
    assert tot == n and r["all"]["pos"] == int(y.sum())
    assert r["S3_isolated"]["rule_AUC_PR"] <= 0.01 + r["S3_isolated"]["prevalence"] or "note" in r["S3_isolated"]
    import json; print(json.dumps(r, indent=1)); print("unit test OK")
