"""SMOKE RUN ONLY -- reduced budget, does the pipeline execute end to end.
Disclosed in FINDINGS.md. n=6/arm, EPOCHS=300 -> the numbers are not the round-4 numbers."""
import experiment as E
E.EPOCHS = 300
E.CORRECT_STEPS = 200
E.N_ARM = 6
E.ROUTED_SEEDS = list(range(700, 712))
E.STD_SWEEP_SEEDS = list(range(3000, 3060))
E.H4_SWEEP_SEEDS = list(range(5000, 5080))
E.OUT = "/Users/steven/research_sprint_20260724/r4-tamper-power-fix/smoke_results.json"
E.main()
