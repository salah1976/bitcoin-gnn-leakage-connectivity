ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
SRC = f"{ROOT}/scripts/griffin_v28_revision.py"
DST = f"{ROOT}/scripts/griffin_sweep_v28.py"
s = open(SRC).read()

s = s.replace('RUN_TAG = f"{FEATURE_MODE}_seed{SEED}" + ("_infer" if LOAD_PT else "")',
  'RUN_TAG = f"{FEATURE_MODE}_seed{SEED}" + ("_infer" if LOAD_PT else "")\nif LOAD_PT is None:\n    raise SystemExit("STOP : griffin_sweep_v28.py n\'entraine jamais. Lance-le uniquement avec run_queue_sweep.py (qui fournit LOAD_PT = checkpoint). Aucun fichier n\'a ete modifie.")', 1)
mk = "# ============================================================\n# 11) TEST EVALUATION"
assert s.count(mk) == 1
ins = '''# ============================================================
# REVISION : BALAYAGES SANS REENTRAINEMENT (checkpoint charge via LOAD_PT)
#  (a) Top-K, ordre d'origine (arbitraire : score d'exposition nul pour les tx de test)
#  (b) Top-K, ordre ALEATOIRE (controle du biais de selection)
#  (c) "historique conserve" : on supprime les aretes tx->adresse d'une fraction q de transactions de test (rho constant)
# Pour chaque point : lambda, rho, AUC-PR, AUC-ROC, F1 (seuil theta_F1 de validation). Sortie : sweep_<RUN_TAG>.json
# ============================================================
import json as _json
SWEEP_K = [1, 2, 5, 25, 100, 1000]
SWEEP_Q = [0.5, 0.9, 0.98, 0.995, 1.0]
_te_arr = np.where(test_mask)[0]
_te_ids = set(_te_arr.tolist())
_te_e0 = pd.concat([inp_te[inp_te["addr_idx"].isin(known)][["tx_idx", "addr_idx"]],
                    out_te[out_te["addr_idx"].isin(known)][["tx_idx", "addr_idx"]]]).drop_duplicates().reset_index(drop=True)
_ik0 = inp_te[inp_te["addr_idx"].isin(known)]
_ok0 = out_te[out_te["addr_idx"].isin(known)]
_cnt0 = _te_e0.groupby("addr_idx")["tx_idx"].nunique()
_shared0 = set(_cnt0[_cnt0 > 1].index)

def _lam_rho(e):
    fc = e.groupby("addr_idx")["tx_idx"].nunique()
    lam = e[e["addr_idx"].isin(fc[fc > 1].index)]["tx_idx"].nunique() / len(_te_ids)
    return float(lam), float(e["tx_idx"].nunique() / len(_te_ids))

def _graph_topk(K, shuffle_seed=None):
    e = _te_e0
    if shuffle_seed is not None:
        e = _te_e0.iloc[np.random.default_rng(shuffle_seed).permutation(len(_te_e0))].reset_index(drop=True)
    rank = e.groupby("addr_idx").cumcount().values + 1
    e_k = e[(~e["addr_idx"].isin(_shared0).values) | (rank <= K)]
    lam, rho = _lam_rho(e_k)
    ik_f = _ik0.merge(e_k, on=["tx_idx", "addr_idx"], how="inner")
    ok_f = _ok0.merge(e_k, on=["tx_idx", "addr_idx"], how="inner")
    return make_hetero(pd.concat([inp_tv, ik_f], ignore_index=True), pd.concat([out_tv, ok_f], ignore_index=True)), lam, rho

def _graph_hp(q, seed=0):
    d = build_test_known()
    removed = set(np.random.default_rng(seed).permutation(_te_arr)[: int(q * len(_te_arr))].tolist())
    flag = torch.zeros(num_tx, dtype=torch.bool)
    if removed: flag[torch.tensor(sorted(removed), dtype=torch.long)] = True
    for et in (("transaction", "tx_to_input", "address"), ("transaction", "tx_to_output", "address")):
        ei = d[et].edge_index
        d[et].edge_index = ei[:, ~flag[ei[0]]]
    snd = _te_e0[~_te_e0["tx_idx"].isin(removed).values]
    cnt = snd.groupby("addr_idx")["tx_idx"].nunique()
    m = _te_e0.merge(cnt.rename("n_send"), left_on="addr_idx", right_index=True, how="left").fillna({"n_send": 0})
    m["others"] = m["n_send"] - (~m["tx_idx"].isin(removed)).astype(int)
    lam = float(m[m["others"] > 0]["tx_idx"].nunique() / len(_te_ids))
    return d, lam, float(_te_e0["tx_idx"].nunique() / len(_te_ids))

_sweep = []
def _run(kind, param, builder):
    t0 = time.time()
    g, lam, rho = builder()
    p_, t_ = predict(make_loader(g, test_idx_t))
    m = mets(p_, t_, th_f1)
    rec = {"kind": kind, "param": param, "lambda": round(lam, 4), "rho": round(rho, 4), **m}
    _sweep.append(rec)
    print(f"  {kind:12s} {str(param):>6s} | lambda={lam:.4f} rho={rho:.4f} | AUC-PR={m['AUC-PR']:.4f} AUC-ROC={m['AUC-ROC']:.4f} F1={m['F1']:.4f} ({time.time()-t0:.0f}s)")
    del g, p_, t_; gc.collect()
    if GPU_MODE: torch.cuda.empty_cache()

print("\\n=== BALAYAGES (sans reentrainement) ===")
for K in SWEEP_K:
    _run("topk_orig", K, lambda K=K: _graph_topk(K))
    _run("topk_random", K, lambda K=K: _graph_topk(K, shuffle_seed=0))
for q in SWEEP_Q:
    _run("hist_keep", q, lambda q=q: _graph_hp(q))
_json.dump({"seed": SEED, "feature_mode": FEATURE_MODE, "th_f1": th_f1, "points": _sweep},
           open(f"{OUT_DIR}/sweep_{RUN_TAG}.json", "w"), indent=1)
print(f"Ecrit : {OUT_DIR}/sweep_{RUN_TAG}.json")
import sys; sys.exit(0)

'''
open(DST, 'w').write(s.replace(mk, ins + mk))
print('Ecrit :', DST, '|', len((s.replace(mk, ins + mk)).splitlines()), 'lignes')
