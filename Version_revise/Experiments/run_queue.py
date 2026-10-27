"""
run_queue.py -- unattended, RESUMABLE queue of Griffin runs (one training per (feature_mode, seed)).

For each pair it takes the text of griffin_v28_revision.py, rewrites ONLY the four configuration lines
(SEED, FEATURE_MODE, RUN_SWEEP, LOAD_PT), executes it in a fresh namespace, then frees memory.
A pair is skipped if runs/<mode>_seed<N>/anchors_<mode>_seed<N>.json already exists, so after a Colab
disconnection you simply run this file again and it continues where it stopped.

Usage in Colab (after copying both scripts to Drive, folder 'scripts'):
    %run /content/drive/MyDrive/bitcoin_snapshots_project/scripts/run_queue.py
"""
import gc, os, re, sys, time, traceback

ROOT = "/content/drive/MyDrive/bitcoin_snapshots_project"
SCRIPT = f"{ROOT}/scripts/griffin_v28_revision.py"
RUNS = f"{ROOT}/05_results/runs"

# order = shortest first (the 'full' model stops early, ~35 min; 'no_fraud' runs ~65-90 min)
QUEUE = [("full", 123), ("full", 2024), ("full", 7),
         ("full", 101), ("no_fraud", 101),
         ("full", 202), ("no_fraud", 202)]
SKIP_IF_DONE = True

# TASK = "train"      : train every (mode, seed) of QUEUE that has no results yet
# TASK = "protocols"  : NO training. Re-evaluate every trained checkpoint of the 12 runs under the history-preserving
#                       isolation protocols (results saved in <mode>_seed<N>_infer/anchors_*.json)
TASK = "train"
ALL_RUNS = [(m, s) for m in ("full", "no_fraud") for s in (42, 123, 2024, 7, 101, 202)]


def patch_config(src, mode, seed, load_pt=None, sweep=False):
    """rewrite the 4 configuration lines; each must match exactly once."""
    rules = [(r'^SEED = \d+.*$', f'SEED = {seed}'),
             (r'^FEATURE_MODE = "[a-z_]+".*$', f'FEATURE_MODE = "{mode}"'),
             (r'^RUN_SWEEP = (True|False).*$', f'RUN_SWEEP = {sweep}'),
             (r'^LOAD_PT = .*$', f'LOAD_PT = {repr(load_pt)}')]
    for pat, rep in rules:
        src, n = re.subn(pat, rep, src, flags=re.M)
        if n != 1:
            raise RuntimeError(f"configuration line not found exactly once for pattern {pat!r} (found {n})")
    return src


RUN_TAG_LINE = 'RUN_TAG = f"{FEATURE_MODE}_seed{SEED}" + ("_infer" if LOAD_PT else "")'


def patch_recovery(src):
    """when a run is recovered from its saved checkpoint, keep the normal run folder name (no '_infer' suffix)."""
    if src.count(RUN_TAG_LINE) != 1:
        raise RuntimeError("RUN_TAG line not found exactly once")
    return src.replace(RUN_TAG_LINE, 'RUN_TAG = f"{FEATURE_MODE}_seed{SEED}"')


def ckpt_path(mode, seed):
    return f"{RUNS}/{mode}_seed{seed}/griffin_v26_seed{seed}_pretrain_checkpoint.pt"


def is_done(mode, seed):
    tag = f"{mode}_seed{seed}"
    return os.path.exists(f"{RUNS}/{tag}/anchors_{tag}.json")


def find_script():
    """look for griffin_v28_revision.py: configured path first, then anywhere under the project folder, then MyDrive."""
    import glob
    if os.path.exists(SCRIPT):
        return SCRIPT
    for pattern in (f"{ROOT}/**/griffin_v28_revision*.py", f"{os.path.dirname(ROOT)}/**/griffin_v28_revision*.py"):
        hits = sorted(glob.glob(pattern, recursive=True))
        if hits:
            print(f"script not at {SCRIPT}; using {hits[0]}")
            return hits[0]
    return None


def protocols_done(mode, seed):
    import json
    tag = f"{mode}_seed{seed}_infer"
    f = f"{RUNS}/{tag}/anchors_{tag}.json"
    if not os.path.exists(f):
        return False
    try:
        return "hist_preserved_known" in json.load(open(f))
    except Exception:
        return False


def main():
    script = find_script()
    if script is None:
        print("\nSTOP: griffin_v28_revision.py is not on your Drive.\n"
              f"  1) download it from the chat (file card),\n"
              f"  2) upload it to Google Drive in: {os.path.dirname(SCRIPT)}  (create the folder 'scripts' if needed),\n"
              "  3) run this queue again.")
        return
    base = open(script).read()
    t_all = time.time(); log = []
    work = ALL_RUNS if TASK == "protocols" else QUEUE
    for mode, seed in work:
        tag = f"{mode}_seed{seed}"
        if TASK == "protocols":
            if not os.path.exists(ckpt_path(mode, seed)):
                print(f"[skip] {tag}: no trained checkpoint"); continue
            if SKIP_IF_DONE and protocols_done(mode, seed):
                print(f"[skip] {tag}: protocols already evaluated"); continue
        elif SKIP_IF_DONE and is_done(mode, seed):
            print(f"[skip] {tag} already done"); continue
        print(f"\n{'=' * 70}\n>>> START {tag}  ({time.strftime('%H:%M:%S')})\n{'=' * 70}")
        t0 = time.time()
        ns = {"__name__": "__main__"}
        status = "ok"
        # a crash AFTER training (final evaluation) leaves the trained checkpoint: reuse it instead of retraining
        recovered = os.path.exists(ckpt_path(mode, seed))
        if recovered:
            print(f"    trained checkpoint found for {tag}: evaluation only, no retraining")
        try:
            code = patch_config(base, mode, seed, load_pt=ckpt_path(mode, seed) if recovered else None)
            if recovered and TASK != "protocols":
                code = patch_recovery(code)     # keep the normal folder name; in 'protocols' mode the '_infer' folder is wanted
            exec(compile(code, script, "exec"), ns)
        except SystemExit:
            pass                                    # the script ends with sys.exit(0) when RUN_SWEEP=False
        except Exception:
            status = "ERROR"; traceback.print_exc()
        finally:
            ns.clear(); gc.collect()
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        ok = protocols_done(mode, seed) if TASK == "protocols" else is_done(mode, seed)
        log.append((tag, status if ok or status == "ERROR" else "no anchors file", round((time.time() - t0) / 60, 1)))
        print(f"<<< END {tag}: {log[-1][1]} in {log[-1][2]} min | anchors file present: {ok}")
    print("\n=== QUEUE SUMMARY ===")
    for row in log: print(row)
    print(f"total {round((time.time() - t_all) / 60, 1)} min")


if __name__ == "__main__":
    main()
