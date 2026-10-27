"""
strata_from_checkpoint.py -- NO training, NO GPU.

The Griffin v26/v28 scripts save, inside the .pt file, the test predictions of the model
(keys: test_probs, test_true, val_probs, conn_probs, isol_probs, connectivity_mask, theta_f1, lambda_sweep ...).
So the article's own model (griffin_v26_gpu.pt, seed 42) can be analysed directly:

  1. recompute AUC-PR / AUC-ROC / F1 from the stored predictions and compare with the stored values
     (does this checkpoint really give 0.8229 / 0.9631 / 0.5226 ?)
  2. model vs model-free rule (max train-label fraud count over the tx addresses), overall and by stratum:
        S1 reported address | S2 connected, no reported address | S3 isolated

Usage (Colab):   %run strata_from_checkpoint.py
Unit test    :   python strata_from_checkpoint.py --selftest
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score


def strata_report(model_p, y_true, rule_score, connected):
    model_p = np.asarray(model_p, float); y_true = np.asarray(y_true).astype(int)
    rule_score = np.asarray(rule_score, float); connected = np.asarray(connected).astype(bool)

    def block(sel):
        yy = y_true[sel]
        out = {"n": int(sel.sum()), "pos": int(yy.sum()), "neg": int((1 - yy).sum()),
               "prevalence": float(yy.mean()) if sel.sum() else None}
        if 0 < yy.sum() < len(yy):
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
    assert (s1 | s2 | s3).all()
    return {"all": block(np.ones(len(y_true), bool)), "S1_reported_address": block(s1),
            "S2_connected_not_reported": block(s2), "S3_isolated": block(s3)}


def evaluate_checkpoint(ck, y_test, rule, connected):
    """ck: dict loaded from a Griffin .pt ; y_test / rule / connected are aligned with np.where(test_mask)[0]."""
    p = np.asarray(ck["test_probs"], float); t = np.asarray(ck["test_true"]).astype(int)
    out = {"keys_present": sorted(k for k in ck.keys() if k != "model_state_dict")}
    out["alignment_ok"] = bool(len(t) == len(y_test) and (t == y_test).all())
    if not out["alignment_ok"]:
        out["ERROR"] = "stored test_true differs from y[test_mask]: ordering or labels differ, do not interpret"
        return out
    th = float(ck.get("theta_f1", 0.5))
    pred = (p >= th).astype(int)
    out["recomputed"] = {"AUC_PR": round(float(average_precision_score(t, p)), 4),
                         "AUC_ROC": round(float(roc_auc_score(t, p)), 4),
                         "F1_at_theta_f1": round(float(f1_score(t, pred, zero_division=0)), 4),
                         "precision": round(float(precision_score(t, pred, zero_division=0)), 4),
                         "recall": round(float(recall_score(t, pred, zero_division=0)), 4), "theta_f1": th}
    out["stored"] = {k: (round(float(ck[k]), 4) if k in ck else None) for k in ("test_ap", "test_auc", "test_f1")}
    out["strata"] = strata_report(p, t, rule, connected)
    sw = ck.get("lambda_sweep")
    if sw:
        import hashlib
        pts = sorted((round(float(r["lambda"]), 4), round(float(r["AUC_PR"]), 4)) for r in sw)
        out["stored_sweep"] = {"n_points": len(pts),
                               "fingerprint": hashlib.md5(str(pts).encode()).hexdigest()[:8],
                               "AUC_PR_at_24pct": next((a for l, a in pts if abs(l - 0.2427) < 0.002), None),
                               "AUC_PR_max": max(a for l, a in pts), "lambda_at_max": max(pts, key=lambda x: x[1])[0]}
    return out


def _selftest():
    rng = np.random.default_rng(1)
    n = 4000
    conn = rng.random(n) < 0.3
    rule = np.where(conn & (rng.random(n) < 0.6), rng.integers(1, 9, n), 0).astype(float)
    y = ((rule > 0) & (rng.random(n) < 0.2)) | (rng.random(n) < 0.003)
    p = np.clip(0.3 * (rule > 0) + 0.05 * rng.random(n) + 0.4 * y * rng.random(n), 0, 1)
    ck = {"test_probs": p, "test_true": y.astype(int), "theta_f1": 0.3, "test_ap": 0.5,
          "lambda_sweep": [{"lambda": 0.2427, "AUC_PR": 0.8}, {"lambda": 0.63, "AUC_PR": 0.82}]}
    r = evaluate_checkpoint(ck, y.astype(int), rule, conn)
    assert r["alignment_ok"] and "strata" in r and r["strata"]["all"]["n"] == n
    assert r["stored_sweep"]["n_points"] == 2 and r["stored_sweep"]["AUC_PR_at_24pct"] == 0.8
    bad = evaluate_checkpoint({**ck, "test_true": 1 - y.astype(int)}, y.astype(int), rule, conn)
    assert not bad["alignment_ok"] and "ERROR" in bad
    print(json.dumps(r["recomputed"], indent=1)); print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest(); sys.exit(0)

    import sqlite3, torch
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
    DRIVE_DB = f"{ROOT}/01_data/databases/all_snapshots_extended_D1_D4_D6_D7_D8_D9_v21.db"
    LABELS_CSV = f"{ROOT}/01_data/processed/dataset_v21_full.csv"
    OUT_JSON = f"{ROOT}/05_results/griffin_v26_final/strata_from_checkpoints.json"
    NAMES = ["griffin_v26_gpu.pt", "griffin_v26_fullgraph_gpu.pt", "griffin_v27_WITH_WEIGHTS.pt", "griffin_v26_seed999_gpu.pt"]
    SNAP = ["D1", "D2", "D3", "D4", "D5", "D6"]

    # ---- data (same loading as v28 / structure_audit.py)
    conn_db = sqlite3.connect(DRIVE_DB)
    inputs_df = pd.read_sql_query("SELECT tx_hash, input_address FROM tx_inputs", conn_db)
    outputs_df = pd.read_sql_query("SELECT tx_hash, output_address FROM tx_outputs", conn_db)
    conn_db.close()
    tx = pd.read_csv(LABELS_CSV)[["tx_hash", "snapshot_id", "label_final"]]
    tx = tx[tx["snapshot_id"].isin(SNAP)].copy()
    tx["rk"] = tx["snapshot_id"].map({s: i for i, s in enumerate(SNAP)})
    tx = tx.sort_values(["rk", "tx_hash"]).reset_index(drop=True)
    train_mask = tx["snapshot_id"].isin(["D1", "D2", "D3"]).values
    val_mask = tx["snapshot_id"].isin(["D4"]).values
    test_mask = tx["snapshot_id"].isin(["D5", "D6"]).values
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
    num_addr = len(allA)

    te_arr = np.where(test_mask)[0]
    tr_set = set(np.where(train_mask)[0].tolist()); tv_set = tr_set | set(np.where(val_mask)[0].tolist())
    te_set = set(te_arr.tolist())
    edges = pd.concat([inputs_df[["tx_idx", "addr_idx"]], outputs_df[["tx_idx", "addr_idx"]]]).drop_duplicates()
    e_te = edges[edges["tx_idx"].isin(te_set)]
    A_tv = set(edges[edges["tx_idx"].isin(tv_set)]["addr_idx"].unique().tolist())
    connected_set = set(e_te[e_te["addr_idx"].isin(A_tv)]["tx_idx"].unique().tolist())
    connected = np.array([t in connected_set for t in te_arr])

    fr_tr = set(np.where(train_mask & (y == 1))[0].tolist())
    tot_fr = (np.bincount(inputs_df[inputs_df["tx_idx"].isin(fr_tr)]["addr_idx"].values, minlength=num_addr) +
              np.bincount(outputs_df[outputs_df["tx_idx"].isin(fr_tr)]["addr_idx"].values, minlength=num_addr))
    rule = (e_te.assign(s=tot_fr[e_te["addr_idx"].values]).groupby("tx_idx")["s"].max()
            .reindex(te_arr, fill_value=0).values.astype(float))
    y_test = y[te_arr]
    print(f"test={len(te_arr):,} | connected share={connected.mean():.4f} | rule AUC-PR={average_precision_score(y_test, rule):.4f}")

    # ---- checkpoints
    MANUAL_PATHS = {}   # si vous retrouvez un fichier a la main: {"griffin_v26_gpu.pt": "/content/drive/MyDrive/.../griffin_v26_gpu.pt"}

    # inventaire de TOUS les .pt de Drive (le glob sous ROOT n'a rien trouve)
    inventory = []
    for base in ["/content/drive/MyDrive", "/content/drive/Shareddrives"]:
        for dp, dn, fn in os.walk(base):
            for f in fn:
                if f.endswith((".pt", ".pth", ".ckpt")):
                    fp = os.path.join(dp, f); inventory.append((f, fp, os.path.getsize(fp) / 1e6))
    print(f"\n{len(inventory)} fichiers .pt/.pth/.ckpt sur Drive :")
    for f, fp, mb in sorted(inventory, key=lambda x: x[1]):
        print(f"  {mb:8.1f} MB  {fp}")

    NAMES = sorted({f for f, fp, mb in inventory if f.startswith("griffin_") and f.endswith(".pt") and mb > 10} | set(NAMES))
    print("\ncheckpoints analysés (ceux > 10 MB contiennent normalement les prédictions) :", NAMES)
    results = {}
    for name in NAMES:
        found = [MANUAL_PATHS[name]] if name in MANUAL_PATHS else [fp for f, fp, mb in inventory if f == name]
        if not found:
            print(f"[absent] {name}"); continue
        path = found[0]
        print(f"\n=== {name}  ({path})")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "test_probs" not in ck:
            print("   no stored test_probs in this checkpoint; keys:", list(ck.keys())[:15]); continue
        r = evaluate_checkpoint(ck, y_test, rule, connected)
        results[name] = r
        print(json.dumps({k: v for k, v in r.items() if k != "keys_present"}, indent=1))

    # ---- résumé compact entre checkpoints
    rows = []
    for name, r in results.items():
        if "strata" not in r: continue
        st = r["strata"]; sw = r.get("stored_sweep", {})
        rows.append((name, r["recomputed"]["AUC_PR"], r["recomputed"]["AUC_ROC"],
                     st["S1_reported_address"].get("model_AUC_PR"), st["S1_reported_address"].get("rule_AUC_PR"),
                     st["S2_connected_not_reported"].get("model_AUC_PR"), st["S2_connected_not_reported"].get("model_AUC_ROC"),
                     st["S3_isolated"].get("model_AUC_PR"), st["S3_isolated"].get("model_AUC_ROC"),
                     sw.get("fingerprint"), sw.get("AUC_PR_at_24pct")))
    print("\n=== RESUME (AUC-PR global | S1 modele / regle | S2 PR / ROC | S3 PR / ROC | empreinte du balayage stocke | AUC-PR a 24 %) ===")
    for row in rows: print(row)
    if len(rows) > 1:
        arr = np.array([row[1] for row in rows], float)
        print(f"AUC-PR global sur {len(rows)} checkpoints : moyenne={arr.mean():.4f} ecart-type={arr.std(ddof=1):.4f} min={arr.min():.4f} max={arr.max():.4f}")
    fps = [row[9] for row in rows if row[9]]
    if len(fps) > 1 and len(set(fps)) < len(fps):
        print("ATTENTION : plusieurs checkpoints ont EXACTEMENT le meme balayage stocke (empreinte identique) : balayage recharge d'un ancien JSON, pas recalcule.")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(results, open(OUT_JSON, "w"), indent=1)
    print("\nSaved ->", OUT_JSON)
