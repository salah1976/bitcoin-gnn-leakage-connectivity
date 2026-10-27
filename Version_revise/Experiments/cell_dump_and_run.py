# --- verification du GPU, puis installation de PyG si le runtime est neuf ---
import subprocess, sys
import torch
if not torch.cuda.is_available():
    raise SystemExit("STOP : ce notebook n'a PAS de GPU. Menu Execution > Modifier le type d'execution > T4 GPU > Enregistrer (la session redemarre), puis relance cette cellule.")
try:
    import torch_geometric
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)
    _v = torch.__version__.split("+")[0]; _cu = "cu" + torch.version.cuda.replace(".", "")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyg_lib", "-f", f"https://data.pyg.org/whl/torch-{_v}+{_cu}.html"])
    import torch_geometric
print("GPU :", torch.cuda.get_device_name(0), "| torch_geometric", torch_geometric.__version__)

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
SRC = f"{ROOT}/scripts/griffin_v28_revision.py"
DST = f"{ROOT}/scripts/griffin_dump_v28.py"
s = open(SRC).read()
s = s.replace('RUN_TAG = f"{FEATURE_MODE}_seed{SEED}" + ("_infer" if LOAD_PT else "")',
  'RUN_TAG = f"{FEATURE_MODE}_seed{SEED}" + ("_infer" if LOAD_PT else "")\nif LOAD_PT is None:\n    raise SystemExit("STOP : griffin_dump_v28.py n\'entraine jamais. Lance-le uniquement avec run_queue_dump.py (qui fournit LOAD_PT = checkpoint). Aucun fichier n\'a ete modifie.")', 1)
mk = "# ============================================================\n# 11) TEST EVALUATION"
assert s.count(mk) == 1
ins = '''# ============================================================
# REVISION : SAUVEGARDE DES PROBABILITES DE TEST (sans reentrainement ; checkpoint charge via LOAD_PT)
# Trois graphes de test : Known-Address (protocole principal), graphe naturel complet, isolation K=1.
# Sortie : test_probs_<RUN_TAG>.npz (hash des tx, snapshot, etiquette, probabilites, seuil theta_F1)
# ============================================================
_te_arr = np.where(test_mask)[0]
_p_known, _y_known = predict(make_loader(data_test, test_idx_t))
_p_nat, _ = predict(make_loader(make_hetero(inputs_df, outputs_df), test_idx_t))
_g1, _lam1, _ = build_test_topk(1)
_p_k1, _ = predict(make_loader(_g1, test_idx_t))
assert len(_p_known) == len(_te_arr) == len(_p_nat) == len(_p_k1)
assert (_y_known == y[_te_arr]).all()
np.savez_compressed(f"{OUT_DIR}/test_probs_{RUN_TAG}.npz",
                    tx_hash=np.asarray(transactions["tx_hash"].values[_te_arr], dtype="U80"),
                    snap=np.asarray(transactions["snapshot_id"].values[_te_arr], dtype="U8"),
                    y=y[_te_arr].astype(np.int8), th_f1=np.float64(th_f1), lam_k1=np.float64(_lam1),
                    p_known=_p_known.astype(np.float32), p_natural=_p_nat.astype(np.float32), p_k1=_p_k1.astype(np.float32))
print(f"Ecrit : {OUT_DIR}/test_probs_{RUN_TAG}.npz | n_test={len(_te_arr):,} | lambda(K=1)={_lam1:.4f}")
import sys; sys.exit(0)

'''
out = s.replace(mk, ins + mk)
open(DST, 'w').write(out)
print('Ecrit :', DST, '|', len(out.splitlines()), 'lignes')

# ---------------- file d'attente (inference seule, reprenable) ----------------
import gc, glob, os, re, sys, time, traceback

SCRIPT = DST
RUNS = f"{ROOT}/05_results/runs"
SEEDS = (42, 123, 2024, 7, 101, 202)
WORK = [(m, s) for s in SEEDS for m in ("full", "no_fraud")]


def patch_config(src, mode, seed, load_pt):
    rules = [(r'^SEED = \d+.*$', f'SEED = {seed}'),
             (r'^FEATURE_MODE = "[a-z_]+".*$', f'FEATURE_MODE = "{mode}"'),
             (r'^RUN_SWEEP = (True|False).*$', 'RUN_SWEEP = False'),
             (r'^LOAD_PT = .*$', f'LOAD_PT = {repr(load_pt)}')]
    for pat, rep in rules:
        src, n = re.subn(pat, rep, src, flags=re.M)
        if n != 1:
            raise RuntimeError(f"ligne de configuration non trouvee une seule fois : {pat!r} ({n})")
    return src


def find_script():
    if os.path.exists(SCRIPT):
        return SCRIPT
    hits = sorted(glob.glob(f"{ROOT}/**/griffin_dump_v28*.py", recursive=True))
    return hits[0] if hits else None


def main():
    script = find_script()
    if script is None:
        print(f"STOP : griffin_dump_v28.py est introuvable. Depose-le dans {os.path.dirname(SCRIPT)} puis relance.")
        return
    base = open(script).read()
    log = []; t_all = time.time()
    for mode, seed in WORK:
        tag = f"{mode}_seed{seed}"
        ckpt = f"{RUNS}/{tag}/griffin_v26_seed{seed}_pretrain_checkpoint.pt"
        out = f"{RUNS}/{tag}_infer/test_probs_{tag}_infer.npz"
        if not os.path.exists(ckpt):
            print(f"[skip] {tag} : pas de checkpoint"); continue
        if os.path.exists(out):
            print(f"[skip] {tag} : probabilites deja sauvegardees"); continue
        print(f"\n{'=' * 70}\n>>> START {tag} ({time.strftime('%H:%M:%S')})\n{'=' * 70}")
        t0 = time.time(); ns = {"__name__": "__main__"}; status = "ok"
        try:
            exec(compile(patch_config(base, mode, seed, ckpt), script, "exec"), ns)
        except SystemExit:
            pass
        except Exception:
            status = "ERREUR"; traceback.print_exc()
        finally:
            ns.clear(); gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        ok = os.path.exists(out)
        log.append((tag, status if ok or status == "ERREUR" else "pas de fichier", round((time.time() - t0) / 60, 1)))
        print(f"<<< END {tag} : {log[-1][1]} en {log[-1][2]} min | fichier present : {ok}")
    print("\n=== RESUME ===")
    for r in log: print(r)
    print(f"total {round((time.time() - t_all) / 60, 1)} min")


main()
