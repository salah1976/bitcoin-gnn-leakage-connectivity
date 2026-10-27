"""
run_queue_sweep.py -- balayages SANS reentrainement des 12 GriffinGNN (6 graines x full/no_fraud).
Pour chaque modele : charge le checkpoint, evalue Top-K (ordre d'origine et ordre aleatoire) et 'historique conserve' (fraction q),
ecrit runs/<mode>_seed<N>_infer/sweep_<mode>_seed<N>_infer.json. Reprenable : un modele deja evalue est saute.
Ordre : graine par graine (full puis no_fraud) pour que des resultats partiels restent equilibres.
Utilisation (cellule GPU) :  %run /content/drive/MyDrive/bitcoin_snapshots_project/scripts/run_queue_sweep.py
(ou coller ce code dans une cellule ; le script griffin_sweep_v28.py doit etre sur Drive dans scripts/)
"""
import gc, glob, os, re, sys, time, traceback

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
SCRIPT = f"{ROOT}/scripts/griffin_sweep_v28.py"
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
    hits = sorted(glob.glob(f"{ROOT}/**/griffin_sweep_v28*.py", recursive=True))
    return hits[0] if hits else None


def main():
    script = find_script()
    if script is None:
        print(f"STOP : griffin_sweep_v28.py est introuvable. Depose-le dans {os.path.dirname(SCRIPT)} puis relance.")
        return
    base = open(script).read()
    log = []; t_all = time.time()
    for mode, seed in WORK:
        tag = f"{mode}_seed{seed}"
        ckpt = f"{RUNS}/{tag}/griffin_v26_seed{seed}_pretrain_checkpoint.pt"
        out = f"{RUNS}/{tag}_infer/sweep_{tag}_infer.json"
        if not os.path.exists(ckpt):
            print(f"[skip] {tag} : pas de checkpoint"); continue
        if os.path.exists(out):
            print(f"[skip] {tag} : balayage deja fait"); continue
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


if __name__ == "__main__":
    main()
