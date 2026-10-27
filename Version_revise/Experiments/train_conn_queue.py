"""
train_conn_queue.py -- R5.5 : le GNN entraine SANS connectivite train-train, 6 graines (reprenable apres coupure Colab).

Compare avec la matrice principale (graphe d'entrainement naturel : les tx D1-D3 et toutes leurs aretes).
Variante "trainiso" : on retire du graphe d'entrainement toute arete vers une adresse utilisee par au moins 2 tx d'entrainement
(lambda_train = 0). Validation, seuil et tests = ceux de la matrice principale (memes graphes), donc la seule difference est
la connectivite vue a l'ENTRAINEMENT. Les dossiers de sortie ont le suffixe _trainiso.

Utilisation (Colab, scripts dans <ROOT>/scripts, avec run_queue.py et griffin_v28_revision.py) :
    %run /content/drive/MyDrive/bitcoin_snapshots_project/scripts/train_conn_queue.py
Environnement : MODES="full" (defaut) ou "full,no_fraud".   Test sans donnees : python train_conn_queue.py --selftest
"""
import gc, os, re, sys, time, traceback

SEEDS = [42, 123, 2024, 7, 101, 202]
TRAIN_LINE = "data_train = build_split(train_mask)"
FUNC = '''
def build_split_train_isolated(mask):
    """graphe d'entrainement SANS connectivite train-train : on garde seulement les adresses utilisees par exactement 1 tx."""
    allowed = set(np.where(mask)[0].tolist())
    ip = inputs_df[inputs_df["tx_idx"].isin(allowed)]; op = outputs_df[outputs_df["tx_idx"].isin(allowed)]
    pairs = pd.concat([ip[["tx_idx","addr_idx"]], op[["tx_idx","addr_idx"]]]).drop_duplicates()
    cnt = pairs.groupby("addr_idx")["tx_idx"].nunique()
    solo = set(cnt[cnt == 1].index)
    lam_nat = pairs[pairs["addr_idx"].isin(cnt[cnt > 1].index)]["tx_idx"].nunique() / max(len(allowed), 1)
    print(f"  lambda_train natural = {lam_nat:.2%} | train graph used: isolated (lambda_train = 0)")
    return make_hetero(ip[ip["addr_idx"].isin(solo)], op[op["addr_idx"].isin(solo)])
'''
RUN_TAG_RE = r'^RUN_TAG = .*$'

def patch_all(src, mode, seed, load_pt=None):
    for pat, rep in [(r'^SEED = \d+.*$', f'SEED = {seed}'), (r'^FEATURE_MODE = "[a-z_]+".*$', f'FEATURE_MODE = "{mode}"'),
                     (r'^RUN_SWEEP = (True|False).*$', 'RUN_SWEEP = False'), (r'^LOAD_PT = .*$', f'LOAD_PT = {repr(load_pt)}'),
                     (RUN_TAG_RE, 'RUN_TAG = f"{FEATURE_MODE}_seed{SEED}_trainiso"')]:
        src, n = re.subn(pat, lambda m, rep=rep: rep, src, flags=re.M)
        if n != 1: raise RuntimeError(f"ligne de configuration absente ou multiple : {pat!r} ({n})")
    if src.count(TRAIN_LINE) != 1: raise RuntimeError("la ligne d'entrainement 'data_train = build_split(train_mask)' n'est pas unique")
    i = src.index("def build_split(mask):"); j = src.index("\ndef ", i + 10)
    src = src[:j] + FUNC + src[j:]
    return src.replace(TRAIN_LINE, "data_train = build_split_train_isolated(train_mask)")

def selftest():
    import numpy as np, pandas as pd, py_compile, tempfile
    here = os.path.dirname(os.path.abspath(__file__)); cand = [os.path.join(here, "griffin_v28_revision.py")]
    base = open(cand[0]).read() if os.path.exists(cand[0]) else None
    if base:
        code = patch_all(base, "full", 42); f = tempfile.mktemp(suffix=".py"); open(f, "w").write(code); py_compile.compile(f, doraise=True)
        assert "_trainiso" in code and "build_split_train_isolated(train_mask)" in code
    ns = {"np": np, "pd": pd, "make_hetero": lambda a, b: (a, b)}
    ns["inputs_df"] = pd.DataFrame({"tx_idx": [0, 1, 2, 3], "addr_idx": [10, 10, 11, 12]}); ns["outputs_df"] = pd.DataFrame({"tx_idx": [0, 3], "addr_idx": [13, 13]})
    exec(FUNC, ns); a, b = ns["build_split_train_isolated"](np.array([True, True, True, True]))
    assert sorted(a["addr_idx"]) == [11, 12] and len(b) == 0, (a, b)     # 10 partage par tx0/tx1 et 13 partage par tx0/tx3 sont retires
    print("selftest OK (patch compile sur griffin_v28_revision.py)" if base else "selftest OK (fonction seule ; griffin_v28_revision.py absent)"); sys.exit(0)

if "--selftest" in sys.argv: selftest()

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"; RUNS = f"{ROOT}/05_results/runs"; SCRIPT = f"{ROOT}/scripts/griffin_v28_revision.py"
MODES = os.environ.get("MODES", "full").split(",")

def done(mode, seed): tag = f"{mode}_seed{seed}_trainiso"; return os.path.exists(f"{RUNS}/{tag}/anchors_{tag}.json")
def ckpt(mode, seed): return f"{RUNS}/{mode}_seed{seed}_trainiso/griffin_v26_seed{seed}_pretrain_checkpoint.pt"

def main():
    if not os.path.exists(SCRIPT): print(f"STOP : {SCRIPT} introuvable (copiez griffin_v28_revision.py dans le dossier scripts)"); return
    base = open(SCRIPT).read(); log = []; t_all = time.time()
    for mode in MODES:
        for seed in SEEDS:
            tag = f"{mode}_seed{seed}_trainiso"
            if done(mode, seed): print(f"[skip] {tag} deja termine"); continue
            print(f"\n{'=' * 70}\n>>> START {tag} ({time.strftime('%H:%M:%S')})\n{'=' * 70}"); t0 = time.time(); ns = {"__name__": "__main__"}; status = "ok"
            rec = os.path.exists(ckpt(mode, seed))
            if rec: print("    checkpoint entraine trouve : evaluation seule")
            try: exec(compile(patch_all(base, mode, seed, ckpt(mode, seed) if rec else None), SCRIPT, "exec"), ns)
            except SystemExit: pass
            except Exception: status = "ERROR"; traceback.print_exc()
            finally:
                ns.clear(); gc.collect()
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception: pass
            log.append((tag, status if done(mode, seed) or status == "ERROR" else "pas de fichier anchors", round((time.time() - t0) / 60, 1)))
            print(f"<<< END {tag}: {log[-1][1]} en {log[-1][2]} min")
    print("\n=== RESUME ==="); [print(r) for r in log]; print(f"total {round((time.time() - t_all) / 60, 1)} min")

if __name__ == "__main__": main()
