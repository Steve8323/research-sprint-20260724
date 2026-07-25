"""POST-HOC ONLY. Nothing here is in PREREG.md. Writes posthoc.json."""
import json, numpy as np
from scipy.stats import fisher_exact
R = json.load(open("results.json")); C = R["cells"]
out = {"LABEL": "POST-HOC -- not preregistered"}

# H1: is the floor inflation gated by per-group width w (competence of a data-chosen column)?
W4 = [("cell1_r1_routing_worldview_select", 4), ("cell_mw_k2_w4", 4), ("cell_mw_k3_w4", 4),
      ("cell_mw_k4_w4", 4), ("cell_entangled_rot", 4), ("cell_entangled_mix", 4)]
W8 = [("cell_mw_k2_w8", 8), ("cell_mw_k3_w8", 8), ("cell_mw_k4_w8", 8),
      ("cell11_r4_break_tautology_REFERENCE", 8)]
def pool(cells):
    k = sum(int(C[c]["fair_floor"].split("/")[0]) for c, _ in cells)
    n = sum(C[c]["n"] for c, _ in cells)
    ko = sum(int(C[c]["old_floor"].split("/")[0]) for c, _ in cells)
    return k, n, ko
k4, n4, o4 = pool(W4); k8, n8, o8 = pool(W8)
_, p = fisher_exact([[k8, n8 - k8], [k4, n4 - k4]], alternative="two-sided")
out["H1_width_gating"] = dict(
    w4_fair=f"{k4}/{n4}", w4_rate=round(k4 / n4, 4), w4_old=f"{o4}/{n4}",
    w8_fair=f"{k8}/{n8}", w8_rate=round(k8 / n8, 4), w8_old=f"{o8}/{n8}",
    fisher_p=float(p),
    note="pooled across cells; cells differ in k and basis, so this is descriptive")

# H2: per-cell competence of the data-chosen columns (the proposed mechanism)
comp = {}
for c in C:
    v = C[c]
    if not isinstance(v.get("per_seed"), list): continue
    rows = v["per_seed"]
    if "fair_held" in rows[0]:
        h = np.array([r["fair_held"] for r in rows]).ravel()
    elif "fair" in rows[0] and "selA_train" in rows[0]["fair"]:
        h = np.array([[r["fair"]["selA_train"], r["fair"]["selB_train"]] for r in rows]).ravel()
    elif "fair" in rows[0] and "selA_held" in rows[0]["fair"]:
        h = np.array([[r["fair"]["selA_held"], r["fair"]["selB_held"]] for r in rows]).ravel()
    else:
        continue
    comp[c] = dict(col_held_median=round(float(np.median(h)), 4),
                   frac_above_0p9=round(float((h > 0.9).mean()), 4),
                   fair_rate=C[c]["fair_floor_rate"])
out["H2_column_competence_under_fair_mask"] = comp

# H3: correlation across cells between column competence and the fair floor
xs = np.array([v["frac_above_0p9"] for v in comp.values()])
ys = np.array([v["fair_rate"] for v in comp.values()])
out["H3_competence_vs_fairfloor"] = dict(
    n_cells=len(xs), pearson_r=round(float(np.corrcoef(xs, ys)[0, 1]), 4),
    cells=list(comp.keys()))

# H4: the LM cell -- why no lift?
lm = C["cell_lm_organism"]["per_seed"]
out["H4_lm"] = dict(
    n=len(lm),
    fair_headsA=[r["headsA"] for r in lm],
    n_seeds_recovering_published_split=int(sum(sorted(r["headsA"]) in ([0, 1], [2, 3]) for r in lm)),
    fair_selA_deploy=[round(r["fair"]["selA_deploy"], 3) for r in lm],
    fair_selB_deploy=[round(r["fair"]["selB_deploy"], 3) for r in lm],
    n_both_selections_mid=int(sum(0.2 <= r["fair"]["selA_deploy"] <= 0.8 and
                                  0.2 <= r["fair"]["selB_deploy"] <= 0.8 for r in lm)),
    note="head/unit groups in a shared-residual transformer are not where a standard-trained "
         "model puts its two cue circuits; both data-chosen groups stay mid-alignment")
json.dump(out, open("posthoc.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "H2_column_competence_under_fair_mask"}, indent=1)[:2500])
print("\nH2:", json.dumps(out["H2_column_competence_under_fair_mask"], indent=1))
