"""SMOKE TEST -- crash-only. Runs every code path with 2 seeds and tiny step counts, then
prints ONLY 'SMOKE OK' plus the elapsed seconds. All numeric output is swallowed so that no
outcome is observed before the confirmatory run."""
import contextlib, io, sys, time
sys.path.insert(0, "/Users/steven/research_sprint_20260724/r5-latent-switch-standard")

buf = io.StringIO()
t0 = time.time()
with contextlib.redirect_stdout(buf):
    import core_tf as CT, core_mlp as CM
    CT.STEPS = 5
    CM.EPOCHS = 5
    CM.NEVAL = 200
    CT.NEVAL = 200
    import experiment as E
    E.CT.STEPS = 5
    E.CM.EPOCHS = 5
    E.CM.NEVAL = 200
    E.CT.NEVAL = 200
    E.KRAND = 2
    E.OUT = "/Users/steven/research_sprint_20260724/r5-latent-switch-standard/_smoke.json"
    E.TF_SEEDS = [700, 701]
    E.TF_N_SEEDS = [700, 701]
    E.MLP_SEEDS = [700, 701]
    E.main()
print(f"SMOKE OK  ({time.time()-t0:.1f} s)")
