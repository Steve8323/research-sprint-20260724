#!/usr/bin/env python3
"""PART 2 — real-token analog on SAVED vineppo artifacts (no new generation).

Probe step-value V* (v_gold) from CHEAP TEXT FEATURES of the saved step strings
(bag-of-words / char n-grams) vs the frozen 1536-d hidden states, under an
IDENTICAL problem-level leave-one-problem-out CV pipeline, difficulty-removed.

Pre-registered: see PREREG.md.  Read-only input; writes results into results.json.
"""
import json, re, time
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer

DATA = Path("/Users/steven/vineppo_replication/results/data.jsonl")
T0 = time.time()

rows = []
for rec in (json.loads(l) for l in DATA.open() if l.strip()):
    n_steps = len([l for l in rec["base_sol"].split("\n") if l.strip()
                   and not l.strip().lower().startswith("the answer is")])
    for p in rec["prefixes"]:
        rows.append(dict(pid=rec["id"], k=p["k"], remaining=n_steps - p["k"],
                         text=p["prefix"], v_gold=p["v_gold"],
                         hidden=np.asarray(p["hidden"], np.float32)))

pids = sorted({r["pid"] for r in rows})
N = len(rows)
y = np.array([r["v_gold"] for r in rows], float)
texts = [r["text"] for r in rows]
H = np.array([r["hidden"] for r in rows])
K = np.array([r["k"] for r in rows], float)
LEN = np.array([[len(t), len(re.findall(r"\w+", t))] for t in texts], float)
print(f"{N} prefixes, {len(pids)} problems; v_gold mean {y.mean():.3f} std {y.std():.3f}")

ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def build_features(name, train_idx=None, test_idx=None):
    """Return (Xtr, Xte) for a fold. For text features the vectorizer is fit on
    the TRAINING fold only (no leakage). Static features ignore the split."""
    if name == "hidden":
        return H, H
    if name == "len":
        return LEN, LEN
    # text features: fit vectorizer on train texts only
    if name == "bow":
        vec = CountVectorizer(lowercase=True, token_pattern=r"\w+", min_df=2)
    elif name == "char":
        vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    else:
        raise ValueError(name)
    tr_texts = [texts[i] for i in train_idx]
    Xtr = vec.fit_transform(tr_texts).toarray().astype(np.float32)
    Xall = vec.transform(texts).toarray().astype(np.float32)
    return Xtr, Xall  # caller slices Xall[test_idx]


def loro_predict(name):
    """Leave-one-problem-out out-of-fold predictions with nested-CV alpha pick.
    For static features (hidden/len) X is fixed; for text it is refit per fold."""
    pred = np.zeros(N)
    static = name in ("hidden", "len")
    if static:
        Xfull, _ = build_features(name)
    for held in pids:
        te = np.array([r["pid"] == held for r in rows])
        tr = ~te
        tr_idx = np.where(tr)[0]
        if static:
            Xtr, Xte = Xfull[tr], Xfull[te]
            Xall = Xfull
        else:
            Xtr, Xall = build_features(name, train_idx=tr_idx)
            Xte = Xall[te]
        # nested leave-one-problem-out alpha selection on the training problems
        tr_pids = [p for p in pids if p != held]
        best_a, best_mse = None, np.inf
        for a in ALPHAS:
            errs = []
            for vpid in tr_pids:
                inner_te = np.array([rows[i]["pid"] == vpid for i in tr_idx])
                inner_tr = ~inner_te
                m = Ridge(alpha=a).fit(Xtr[inner_tr], y[tr_idx][inner_tr])
                p_in = np.clip(m.predict(Xtr[inner_te]), 0, 1)
                errs.append(((p_in - y[tr_idx][inner_te]) ** 2).sum())
            mse = np.sum(errs) / len(tr_idx)
            if mse < best_mse:
                best_mse, best_a = mse, a
        m = Ridge(alpha=best_a).fit(Xtr, y[tr])
        pred[te] = np.clip(m.predict(Xte), 0, 1)
    return pred


def corr(x, z):
    x, z = np.asarray(x, float), np.asarray(z, float)
    if x.std() < 1e-12 or z.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, z)[0, 1])


def within_center(v):
    """Subtract problem mean from a vector aligned with rows."""
    v = np.asarray(v, float).copy()
    pmean = {p: v[[i for i, r in enumerate(rows) if r["pid"] == p]].mean() for p in pids}
    return v - np.array([pmean[r["pid"]] for r in rows])


def pooled_r(pred):
    return corr(pred, y)


def within_r(pred):
    return corr(within_center(pred), within_center(y))


def dmse_vs_mean(pred):
    # positive = worse than predicting the grand mean of y
    return float(((pred - y) ** 2).mean() - ((y.mean() - y) ** 2).mean())


# ---- cluster bootstrap over problems -----------------------------------------
def cluster_boot(stat_fn, preds_by_name, B=5000, seed=0):
    """Resample problems with replacement; recompute stat for each named pred and
    the paired diffs. preds_by_name: dict name->pred array (aligned to rows)."""
    rng = np.random.RandomState(seed)
    idx_by_pid = {p: np.array([i for i, r in enumerate(rows) if r["pid"] == p]) for p in pids}
    names = list(preds_by_name)
    samples = {n: [] for n in names}
    for _ in range(B):
        chosen = rng.choice(pids, size=len(pids), replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in chosen])
        yb = y[idx]
        # rebuild within-centering on the resampled set (by resampled problem instance)
        # center each drawn problem-block by its own mean
        blocks = [idx_by_pid[p] for p in chosen]
        yc = np.concatenate([yb_blk - yb_blk.mean()
                             for yb_blk in (y[b] for b in blocks)])
        for n in names:
            pb = preds_by_name[n][idx]
            pc = np.concatenate([pb_blk - pb_blk.mean()
                                 for pb_blk in (preds_by_name[n][b] for b in blocks)])
            samples[n].append(stat_fn(pc, yc) if stat_fn is _wr else stat_fn(pb, yb))
    out = {}
    for n in names:
        arr = np.array([s for s in samples[n] if np.isfinite(s)])
        out[n] = dict(mean=float(np.mean(arr)),
                      ci=[float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                      n_valid=int(len(arr)))
    return out, samples


def _wr(a, b):  # within already-centered
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _pr(a, b):  # pooled
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# HONEST COMPANION METRIC (added after seeing a within/between slope-mismatch
# ARTIFACT: the prereg'd "within-r of OOF predictions" can flip sign relative to
# the raw within-feature relationship, because the OOF probe learns a global
# slope dominated by BETWEEN-problem structure. Diagnosis: len raw within-corr
# with v_gold is -0.27, yet within-r of the len-probe's OOF preds is +0.43.
# The leakage-free difficulty-removed probe below trains on WITHIN-CENTERED
# features -> WITHIN-CENTERED target, so the slope itself is learned only from
# within-problem variation. This is the metric to trust; the prereg'd one is
# retained and reported with the artifact flagged.)
# ---------------------------------------------------------------------------
def loro_within_model(name):
    yc = within_center(y)
    pred = np.zeros(N)
    static = name in ("hidden", "len")
    if static:
        Xfull, _ = build_features(name)
    for held in pids:
        te = np.array([r["pid"] == held for r in rows]); tr = ~te
        tr_idx = np.where(tr)[0]
        if static:
            Xall = Xfull
        else:
            _, Xall = build_features(name, train_idx=tr_idx)
        Xc = Xall.copy()
        for p in pids:
            m = np.array([r["pid"] == p for r in rows])
            Xc[m] = Xall[m] - Xall[m].mean(0)
        alpha = 100.0
        mdl = Ridge(alpha=alpha).fit(Xc[tr], yc[tr])
        pred[te] = mdl.predict(Xc[te])
    return pred


def raw_within_corr(vec):
    return corr(within_center(vec), within_center(y))


FEATURES = ["bow", "char", "hidden", "len"]
preds = {}
for name in FEATURES:
    t = time.time()
    preds[name] = loro_predict(name)
    print(f"  {name:7s} pooled r={pooled_r(preds[name]):+.3f}  within r={within_r(preds[name]):+.3f}"
          f"  dMSE={dmse_vs_mean(preds[name]):+.4f}  ({time.time()-t:.1f}s)")

# honest within-model probe
preds_wm = {n: loro_within_model(n) for n in FEATURES}
within_model_r = {n: corr(preds_wm[n], within_center(y)) for n in FEATURES}
raw_within = {
    "len": raw_within_corr(LEN[:, 0]),
    "num_tokens": raw_within_corr(LEN[:, 1]),
    "k": raw_within_corr(K),
}
print("\n  HONEST within-model probe (within-centered feats->target):")
for n in FEATURES:
    print(f"    {n:7s} within-model r={within_model_r[n]:+.3f}")
print("  RAW single-feature within corr:", {k: round(v, 3) for k, v in raw_within.items()})

# bootstrap CIs (prereg'd OOF-within metric — kept, flagged as artifact-prone)
within_boot, wsamp = cluster_boot(_wr, preds, B=5000, seed=0)
pooled_boot, psamp = cluster_boot(_pr, preds, B=5000, seed=0)

# bootstrap CIs for the HONEST within-model probe (primary companion)
wm_boot, wm_samp = cluster_boot(_wr, preds_wm, B=5000, seed=0)
wm_diffs = None  # filled below after helper defined

# paired diffs in within-problem r (same resamples)
def paired_diff(a_name, b_name, samp):
    a = np.array(samp[a_name]); b = np.array(samp[b_name])
    ok = np.isfinite(a) & np.isfinite(b)
    d = (a - b)[ok]
    return dict(mean=float(d.mean()),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])

diffs = {
    "bow_minus_hidden": paired_diff("bow", "hidden", wsamp),
    "char_minus_hidden": paired_diff("char", "hidden", wsamp),
    "bow_minus_char": paired_diff("bow", "char", wsamp),
}
# HONEST within-model primary diffs (text - hidden)
wm_diffs = {
    "bow_minus_hidden": paired_diff("bow", "hidden", wm_samp),
    "char_minus_hidden": paired_diff("char", "hidden", wm_samp),
}

# step-index (k) partial within-problem r control
def partial_within_r_control_k(pred):
    pc = within_center(pred); yc = within_center(y); kc = within_center(K)
    if kc.std() < 1e-9:
        return within_r(pred)
    # residualize pc and yc on kc
    def resid(v):
        b = np.dot(kc, v) / np.dot(kc, kc)
        return v - b * kc
    return corr(resid(pc), resid(yc))

k_control = {n: partial_within_r_control_k(preds[n]) for n in FEATURES}

part2 = dict(
    n_prefixes=N, n_problems=len(pids),
    v_gold=dict(mean=float(y.mean()), std=float(y.std())),
    features=FEATURES,
    point=dict(
        pooled_r={n: pooled_r(preds[n]) for n in FEATURES},
        within_r={n: within_r(preds[n]) for n in FEATURES},
        dmse_vs_mean={n: dmse_vs_mean(preds[n]) for n in FEATURES},
    ),
    bootstrap=dict(within_r=within_boot, pooled_r=pooled_boot),
    paired_within_diffs=diffs,
    HONEST_within_model=dict(
        note=("OOF-within-r (prereg'd metric) is artifact-prone: a global-slope OOF probe "
              "can flip the sign of the within-problem relationship (len raw within-corr with "
              "v_gold=-0.27 but OOF-within-r=+0.43). The within-model probe trains on "
              "within-centered features->within-centered target, learning the slope only from "
              "within-problem variation. This is the metric to trust."),
        point_within_model_r=within_model_r,
        bootstrap=wm_boot,
        paired_diffs_text_minus_hidden=wm_diffs,
        raw_single_feature_within_corr=raw_within,
    ),
    k_control_within_r=k_control,
    predictions={n: preds[n].tolist() for n in FEATURES},
    v_gold_vector=y.tolist(),
    pid_vector=[r["pid"] for r in rows],
    k_vector=[int(r["k"]) for r in rows],
)

print("\n== PART 2 SUMMARY ==")
print(f"{'feature':8s} {'pooled r [95% CI]':28s} {'within r [95% CI]':28s}")
for n in FEATURES:
    pb, wb = pooled_boot[n], within_boot[n]
    print(f"{n:8s} {pb['mean']:+.3f} [{pb['ci'][0]:+.3f},{pb['ci'][1]:+.3f}]   "
          f"{wb['mean']:+.3f} [{wb['ci'][0]:+.3f},{wb['ci'][1]:+.3f}]")
print("\nPRIMARY paired within-r diffs (CI excludes 0 = detected):")
for k, v in diffs.items():
    print(f"  {k:20s} {v['mean']:+.3f}  [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
print(f"\nk-controlled within r: " + "  ".join(f"{n}={k_control[n]:+.3f}" for n in FEATURES))
print("\n== HONEST within-model probe (TRUST THIS) ==")
for n in FEATURES:
    b = wm_boot[n]
    print(f"  {n:8s} within-model r={within_model_r[n]:+.3f}  [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}]")
print("  PRIMARY (honest) text - hidden within-model diffs:")
for kk, v in wm_diffs.items():
    print(f"    {kk:20s} {v['mean']:+.3f}  [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")

out = Path(__file__).parent / "results_part2.json"
json.dump(part2, out.open("w"), indent=1)
print(f"\nwrote {out}  ({time.time()-T0:.1f}s total)")
