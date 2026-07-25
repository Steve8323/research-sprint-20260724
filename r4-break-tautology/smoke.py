"""Crash-only smoke test: runs the FULL pipeline with 2 seeds/arm and prints only OK/FAIL.
No number is displayed, so it cannot inform anything in PREREG.md (which was already final)."""
import io, contextlib, json, experiment as E
E.SEEDS60 = [700, 701]; E.SEEDS30 = [700, 701]
E.OUT = "/private/tmp/claude-501/-Users-steven/bf90930d-a700-4a4b-a64a-fcc272b732c3/scratchpad/smoke.json"
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        E.main()
    d = json.load(open(E.OUT))
    print("SMOKE OK — keys:", sorted(k for k in d if k not in ("arms", "grad_trace")))
except Exception as ex:
    print("SMOKE FAIL:", type(ex).__name__, ex)
