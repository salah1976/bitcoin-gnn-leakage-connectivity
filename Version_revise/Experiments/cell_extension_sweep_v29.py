# ============================================================
# 11ter) EXTENSION DU SWEEP λ — 24% -> ~95%
# Nouvelle cellule, à exécuter APRÈS la section 11bis (sweep v28)
# Réutilise : inputs_df, outputs_df, train_mask, val_mask, test_mask,
#             make_hetero, make_loader, predict, mets, th_f1, model,
#             device, GPU_MODE, OUT_DIR, test_idx_t
# ============================================================

import json, gc
import numpy as np
import pandas as pd

print("\n================ λ SWEEP v29 — EXTENSION 24% -> 95% ================\n")

# -------------------------------------------------
# 1) Ensembles de base (identiques à la section known-address)
# -------------------------------------------------
tv_ids = set(np.where(train_mask | val_mask)[0].tolist())
te_ids = set(np.where(test_mask)[0].tolist())

inp_tv = inputs_df[inputs_df["tx_idx"].isin(tv_ids)]
out_tv = outputs_df[outputs_df["tx_idx"].isin(tv_ids)]
known  = set(inp_tv["addr_idx"].values) | set(out_tv["addr_idx"].values)

inp_te_all = inputs_df[inputs_df["tx_idx"].isin(te_ids)]
out_te_all = outputs_df[outputs_df["tx_idx"].isin(te_ids)]

# -------------------------------------------------
# 2) Adresses purement test-internes partagées (hors "known")
#    -> ce sont elles qui font monter λ au-delà de 24%
# -------------------------------------------------
te_edges_all = pd.concat([
    inp_te_all[["tx_idx", "addr_idx"]],
    out_te_all[["tx_idx", "addr_idx"]],
]).drop_duplicates()

te_only = te_edges_all[~te_edges_all["addr_idx"].isin(known)]
addr_tx_count = te_only.groupby("addr_idx")["tx_idx"].nunique()
shared_test_only_addrs = addr_tx_count[addr_tx_count > 1].index.values

# Révéler d'abord les adresses les plus "connectrices"
addr_order = addr_tx_count.loc[shared_test_only_addrs].sort_values(ascending=False).index.values
n_shared = len(addr_order)
print(f"Adresses test-internes partagées (hors known) : {n_shared:,}")

# -------------------------------------------------
# 3) Construction du graphe — cache limité à 1 entrée (évite l'OOM)
# -------------------------------------------------
_extended_cache_single = {"frac": None, "lam": None, "data": None}

def build_test_extended(frac):
    """frac=0.0 -> known-address seul (λ≈24.27%)
       frac=1.0 -> graphe test complet"""
    fk = round(frac, 4)

    # Cache hit
    if _extended_cache_single["frac"] == fk:
        return _extended_cache_single["lam"], _extended_cache_single["data"]

    # Libère l'ancien graphe AVANT d'en construire un nouveau
    if _extended_cache_single["data"] is not None:
        del _extended_cache_single["data"]
        _extended_cache_single["data"] = None
        gc.collect()
        if GPU_MODE: torch.cuda.empty_cache()

    n_reveal = int(round(frac * n_shared))
    revealed = set(addr_order[:n_reveal])
    allowed  = known | revealed

    ik = inp_te_all[inp_te_all["addr_idx"].isin(allowed)]
    ok = out_te_all[out_te_all["addr_idx"].isin(allowed)]

    te_edges = pd.concat([ik[["tx_idx","addr_idx"]], ok[["tx_idx","addr_idx"]]]).drop_duplicates()
    fc = te_edges.groupby("addr_idx")["tx_idx"].nunique()
    connected_tx = set(te_edges[te_edges["addr_idx"].isin(fc[fc > 1].index)]["tx_idx"].values)
    lam = len(connected_tx) / len(te_ids)

    data_ext = make_hetero(
        pd.concat([inp_tv, ik], ignore_index=True),
        pd.concat([out_tv, ok], ignore_index=True),
    )

    _extended_cache_single["frac"] = fk
    _extended_cache_single["lam"]  = lam
    _extended_cache_single["data"] = data_ext
    return lam, data_ext

def eval_frac(frac):
    lam, data_ext = build_test_extended(frac)
    loader = make_loader(data_ext, test_idx_t)
    probs, true = predict(loader)
    m = mets(probs, true, th_f1)
    del loader, data_ext
    gc.collect()
    if GPU_MODE: torch.cuda.empty_cache()
    return lam, m

lambda_cache_frac = {}

def eval_frac_cached(frac):
    fk = round(frac, 4)
    if fk not in lambda_cache_frac:
        lambda_cache_frac[fk] = eval_frac(frac)
    return lambda_cache_frac[fk]

def find_frac_for_lambda(target, tol=0.0015, max_iter=14):
    lo, hi = 0.0, 1.0
    lam_lo, _ = eval_frac_cached(lo)
    lam_hi, _ = eval_frac_cached(hi)
    if target <= lam_lo: return lo
    if target >= lam_hi: return hi
    best_frac = hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        lam_mid, _ = eval_frac_cached(mid)
        if abs(lam_mid - target) <= tol:
            return mid
        if lam_mid < target: lo = mid
        else: hi = mid; best_frac = mid
    return best_frac

# -------------------------------------------------
# 4) Plafond réel à frac=1.0
# -------------------------------------------------
lam_full, m_full = eval_frac_cached(1.0)
print(f"λ max atteignable (frac=1.0, graphe test complet) : {lam_full*100:.2f}%")
print(f"  AUC-PR={m_full['AUC-PR']:.4f} | F1={m_full['F1']:.4f}\n")

# -------------------------------------------------
# 5) Grille de cibles : 25% -> λ_max, dense en fin de plage
# -------------------------------------------------
targets_ext = list(np.round(np.arange(0.25, min(lam_full, 0.90), 0.02), 4))
if lam_full >= 0.85:
    targets_ext += list(np.round(np.arange(0.85, lam_full + 0.001, 0.01), 4))
targets_ext = sorted(set(t for t in targets_ext if t <= lam_full))

checkpoint_file_ext = f"{OUT_DIR}/lambda_sweep_v29_extension_checkpoint.json"
if os.path.exists(checkpoint_file_ext):
    with open(checkpoint_file_ext) as f:
        sweep_results_ext = json.load(f)
    done_ext = {round(x["target_lambda"], 4) for x in sweep_results_ext}
    print(f"Checkpoint extension chargé ({len(done_ext)} points).")
else:
    sweep_results_ext, done_ext = [], set()

print(f"\n{'cible λ':>9} | {'frac':>7} | {'λ réel':>8} | {'AUC-PR':>8} | {'F1':>8} | {'Prec':>8} | {'Rec':>8}")
print("-"*72)

for target in targets_ext:
    if target in done_ext:
        continue
    frac_found = find_frac_for_lambda(target)
    lam, m = eval_frac_cached(frac_found)
    print(f"{target*100:8.2f}% | {frac_found:7.4f} | {lam*100:7.2f}% | "
          f"{m['AUC-PR']:8.4f} | {m['F1']:8.4f} | {m['P']:8.4f} | {m['R']:8.4f}")
    sweep_results_ext.append({
        "target_lambda": float(target), "frac_revealed": float(frac_found),
        "lambda": float(lam), "AUC_PR": m["AUC-PR"], "AUC_ROC": m["AUC-ROC"],
        "F1": m["F1"], "Precision": m["P"], "Recall": m["R"],
    })
    with open(checkpoint_file_ext, "w") as f:
        json.dump(sweep_results_ext, f, indent=2)

# -------------------------------------------------
# 6) Fusion avec le sweep v28 pour la courbe complète 0% -> λ_max
# -------------------------------------------------
v28_file = f"{OUT_DIR}/lambda_sweep_v28_final.json"
sweep_v28 = json.load(open(v28_file)) if os.path.exists(v28_file) else []
combined = sorted(sweep_v28 + sweep_results_ext, key=lambda x: x["lambda"])

combined_file = f"{OUT_DIR}/lambda_sweep_full_0_to_max.json"
with open(combined_file, "w") as f:
    json.dump(combined, f, indent=2)

print("\n========================================")
print("λ Sweep v29 (extension) terminé.")
print(f"Courbe complète fusionnée (0% -> {lam_full*100:.1f}%) : {combined_file}")
print(f"Nombre total de points : {len(combined)}")
print("========================================")