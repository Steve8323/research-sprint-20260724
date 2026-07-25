"""
POST-HOC / EXPLORATORY (not in PREREG.md). Three questions raised by the confirmatory run:

 P1. The un-ablated routed model has D = 0.81 (deterministic), not the hedger regime the prereg
     predicted. Is it then a per-input-pattern PATCHWORK? Census of the 8 disagreement patterns.
 P2. Is the patchwork a MARGIN RACE between the two closed-form columns (per pattern, whichever
     column's |contribution| is bigger wins deterministically)?
 P3. Is the preregistered GRADED population curve soft per input, or a per-input TOGGLE whose
     flip-alpha varies across inputs (gradedness = heterogeneity)?
     Uses the exact gated form (valid because l2 cross-blocks are identically 0):
        Delta(alpha) = Delta_A + alpha * u_B . relu(alpha * W2_BB h1_B + b2_B) + bias
"""
import json, time
import numpy as np
import torch
import experiment as E

OUT = "/Users/steven/research_sprint_20260724/r3-switch-mechanism/posthoc.json"
FINE = np.linspace(0, 1, 201)


def gated_delta(m, X, alphas):
    """exact Delta(alpha) for the keep-A gate, per example x alpha. [n, len(alphas)]"""
    with torch.no_grad():
        h1 = torch.relu(m.l1(X))
        W2 = m.l2.weight; b2 = m.l2.bias
        u = m.l3.weight[1] - m.l3.weight[0]
        db = float(m.l3.bias[1] - m.l3.bias[0])
        h1A = h1[:, E.UNITS_A]; h1B = h1[:, E.UNITS_B]
        preA = h1A @ W2[np.ix_(E.UNITS_A, E.UNITS_A)].T + b2[E.UNITS_A]
        dA = (torch.relu(preA) @ u[E.UNITS_A]).numpy()
        pre_lin = h1B @ W2[np.ix_(E.UNITS_B, E.UNITS_B)].T          # [n,4]
        b2B = b2[E.UNITS_B]; uB = u[E.UNITS_B]
        out = np.empty((len(X), len(alphas)), dtype=np.float64)
        for k, a in enumerate(alphas):
            hB = a * torch.relu(a * pre_lin + b2B)
            out[:, k] = dA + (hB @ uB).numpy() + db
    return out, dA


def main():
    t0 = time.time()
    Xdep, ydep = E.gen_align(4000, 50001, "deploy")
    yn = ydep.numpy()
    res = json.load(open("/Users/steven/research_sprint_20260724/r3-switch-mechanism/results.json"))
    S = res["S"]
    shared_noise = torch.tensor(np.random.default_rng(999).standard_normal((8, 500, E.INPUT)).astype(np.float32))
    rows = {}
    for seed in S:
        m = E.train_routed(seed, E.CUE_A, E.CUE_B)[0]
        r = {}
        # ---- P1/P2: per-pattern census + margin race ----
        pk, wk, races = [], [], []
        idx = 0
        for v in (0, 1):
            for ct in (0, 1):
                for cp in (0, 1):
                    b = np.zeros(E.INPUT, dtype=np.float32)
                    tb = [ct, ct ^ v]; pb = [cp, cp ^ (1 - v)]
                    b[0], b[1] = E.MARGIN * (2 * tb[0] - 1), E.MARGIN * (2 * tb[1] - 1)
                    b[2], b[3] = E.MARGIN * (2 * pb[0] - 1), E.MARGIN * (2 * pb[1] - 1)
                    X = torch.tensor(b) + E.NOISE * shared_noise[idx]; idx += 1
                    dA, dB, db, dfull = E.decompose(m, X)
                    pred = (dfull > 0).astype(int)
                    pk.append(float((pred == v).mean()))
                    wk.append(float((np.abs(dA) > np.abs(dB)).mean()))
                    # does the bigger-|contribution| column decide the output?
                    win = np.where(np.abs(dA) > np.abs(dB), np.sign(dA), np.sign(dB))
                    races.append(float((win == np.sign(dfull)).mean()))
        r["p_true"] = [round(p, 4) for p in pk]
        r["frac_A_bigger"] = [round(w, 4) for w in wk]
        r["race_decides_output"] = round(float(np.mean(races)), 4)
        r["n_pat_det_true"] = int(sum(p > 0.9 for p in pk))
        r["n_pat_det_proxy"] = int(sum(p < 0.1 for p in pk))
        r["n_pat_ambig"] = int(sum(0.1 <= p <= 0.9 for p in pk))
        r["race_determinism"] = round(float(np.mean([abs(2 * w - 1) for w in wk])), 4)
        # ---- P3: per-example flip structure under the gate ----
        D_alpha, dA_all = gated_delta(m, Xdep, FINE)
        sgn = np.sign(D_alpha)
        nflip = (np.diff(sgn, axis=1) != 0).sum(1)
        # sanity: gated form must reproduce the mask sweep at the 21 prereg alphas
        chk, _ = gated_delta(m, Xdep, np.array(E.ALPHAS))
        acc_chk = ((chk > 0).astype(int) == yn[:, None]).mean(0)
        r["gated_form_max_abs_diff_vs_mask_sweep"] = float(
            np.abs(acc_chk - np.array(res["routed"]["per_seed"][str(seed)]["sweep_keepA"])).max())
        flips = []
        for i in np.where(nflip >= 1)[0]:
            k = int(np.argmax(np.diff(sgn[i]) != 0))
            flips.append(float(FINE[k]))
        r["frac_examples_flip"] = round(float((nflip >= 1).mean()), 4)
        r["frac_examples_exactly_one_flip"] = round(float((nflip == 1).mean()), 4)
        r["median_nflip_among_flippers"] = float(np.median(nflip[nflip >= 1])) if (nflip >= 1).any() else None
        if flips:
            q10, q50, q90 = np.percentile(flips, [10, 50, 90])
            r["flip_alpha_p10"] = round(float(q10), 4); r["flip_alpha_median"] = round(float(q50), 4)
            r["flip_alpha_p90"] = round(float(q90), 4); r["flip_alpha_spread_10_90"] = round(float(q90 - q10), 4)
        r["W_population"] = round(res["routed"]["per_seed"][str(seed)]["trans_keepA"]["W"], 4)
        rows[seed] = r
        print(f"  seed {seed}: det_true {r['n_pat_det_true']}/8 det_proxy {r['n_pat_det_proxy']}/8 "
              f"race {r['race_decides_output']} flipspread {r.get('flip_alpha_spread_10_90')} W {r['W_population']}")

    agg = dict(
        n=len(S),
        pat_det_true_total=int(sum(rows[s]["n_pat_det_true"] for s in S)),
        pat_det_proxy_total=int(sum(rows[s]["n_pat_det_proxy"] for s in S)),
        pat_ambig_total=int(sum(rows[s]["n_pat_ambig"] for s in S)),
        pat_total=8 * len(S),
        seeds_mixed_patchwork=int(sum(rows[s]["n_pat_det_true"] > 0 and rows[s]["n_pat_det_proxy"] > 0 for s in S)),
        race_decides_output_median=round(float(np.median([rows[s]["race_decides_output"] for s in S])), 4),
        race_determinism_median=round(float(np.median([rows[s]["race_determinism"] for s in S])), 4),
        frac_examples_exactly_one_flip_median=round(
            float(np.median([rows[s]["frac_examples_exactly_one_flip"] for s in S])), 4),
        flip_alpha_spread_10_90_median=round(
            float(np.median([rows[s]["flip_alpha_spread_10_90"] for s in S if "flip_alpha_spread_10_90" in rows[s]])), 4),
        W_population_median=round(float(np.median([rows[s]["W_population"] for s in S])), 4),
        gated_form_max_abs_diff=float(max(rows[s]["gated_form_max_abs_diff_vs_mask_sweep"] for s in S)),
    )
    sp = [rows[s]["flip_alpha_spread_10_90"] for s in S if "flip_alpha_spread_10_90" in rows[s]]
    wp = [rows[s]["W_population"] for s in S if "flip_alpha_spread_10_90" in rows[s]]
    from scipy.stats import spearmanr, pearsonr
    agg["spearman_spread_vs_W"] = dict(rho=float(spearmanr(sp, wp).statistic), p=float(spearmanr(sp, wp).pvalue), n=len(sp))
    agg["pearson_spread_vs_W"] = dict(r=float(pearsonr(sp, wp).statistic), p=float(pearsonr(sp, wp).pvalue))
    json.dump(dict(label="POST-HOC / EXPLORATORY (not preregistered)", aggregate=agg, per_seed=rows),
              open(OUT, "w"), indent=1)
    print(json.dumps(agg, indent=1))
    print(f"wrote {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
