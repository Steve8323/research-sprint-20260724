"""ERT with LEARNED meaning classes — does the E1 double dissociation survive?

Arms (all rho=1.0, seeds 0/1/2, repo-quick 600 steps, d=64, L=4, eps=0.5):
  oracle   : repo's dictionary channel (anchor)          -> M, S
  distill  : classes learned by probing a trained rho=0 ERT (PRIMARY) -> M, S
  geom     : k-means(k=24) on frozen public embeddings   -> M, S
  random   : random partition of content words (null)    -> M only

Only the channel's class table changes; the repo's model/train/tasks code is
used unmodified (imported read-only). Writes results.json in this directory.
"""
import os, sys, json, time, math

sys.path.insert(0, "/Users/steven/english_bottleneck_lm")
import numpy as np
import torch

torch.set_num_threads(3)  # shared machine

from vocab import build_embeddings, class_of, V, N_CLASS, PAD, MEMBERS
from paraphrase import ParaphraseChannel
from model import EnglishResidualTransformer, _tok_to_class_logits
from train import train_M, train_S
from tasks import SEQ_LEN, RPOS

import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cpu"
D, LAYERS, EPS, STEPS, RHO = 64, 4, 0.5, 600, 1.0
SEEDS = [0, 1, 2]
BUDGET_S = 540  # soft budget for secondary arms
T0 = time.time()

CONTENT = np.where(class_of >= 0)[0]           # 96 content token ids
TRUE = {int(t): int(class_of[t]) for t in CONTENT}


def log(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


class LearnedClassChannel(nn.Module):
    """Same contract as ParaphraseChannel but the class table is a learned
    partition {token -> group}. With prob rho, swap each content token to a
    uniformly random OTHER member of its learned group; singletons and
    specials pass unchanged."""

    def __init__(self, groups, rho=1.0, device="cpu"):
        super().__init__()
        self.rho = rho
        maxk = max(1, max((len(g) - 1) for g in groups) if groups else 1)
        mates = torch.arange(V, device=device).unsqueeze(1).repeat(1, maxk)
        count = torch.zeros(V, dtype=torch.long, device=device)
        for g in groups:
            for t in g:
                others = [m for m in g if m != t]
                count[t] = len(others)
                for j, m in enumerate(others):
                    mates[t, j] = m
        self.register_buffer("mates", mates)
        self.register_buffer("count", count)

    def forward(self, y):
        if self.rho <= 0:
            return y
        B, T, _ = y.shape
        tok = y.argmax(-1)
        cnt = self.count[tok]                       # (B,T)
        r = torch.rand(B, T, device=y.device)
        ridx = (r * cnt.clamp(min=1)).long().clamp(max=self.mates.shape[1] - 1)
        mate = self.mates[tok].gather(-1, ridx.unsqueeze(-1)).squeeze(-1)
        do = (torch.rand(B, T, device=y.device) < self.rho) & (cnt > 0)
        new_tok = torch.where(do, mate, tok)
        hard = torch.zeros_like(y)
        hard.scatter_(-1, new_tok.unsqueeze(-1), 1.0)
        return hard + (y - y.detach())


def partition_metrics(groups):
    """ARI/purity vs true classes + operational channel metrics."""
    lab = {}
    for gi, g in enumerate(groups):
        for t in g:
            lab[t] = gi
    y_true = [TRUE[int(t)] for t in CONTENT]
    y_pred = [lab.get(int(t), -1 - i) for i, t in enumerate(CONTENT)]
    ari = float(adjusted_rand_score(y_true, y_pred))
    pur = 0
    for g in groups:
        if g:
            pur += max(np.bincount([TRUE[t] for t in g], minlength=N_CLASS))
    purity = pur / len(CONTENT)
    # cross-class swap fraction: uniform over content tokens, then uniform mate
    cross, tot = 0.0, 0
    for g in groups:
        for t in g:
            others = [m for m in g if m != t]
            if others:
                cross += sum(TRUE[m] != TRUE[t] for m in others) / len(others)
            tot += 1  # singletons contribute 0 cross (they pass through)
    cross_frac = cross / max(1, tot)
    # within-class pair coverage: fraction of true synonym (ordered) pairs
    # that the channel can produce as a swap
    cov_num, cov_den = 0, 0
    for c in range(N_CLASS):
        toks = MEMBERS[c]
        for a in toks:
            for b in toks:
                if a != b:
                    cov_den += 1
                    if lab.get(a, -1) == lab.get(b, -2):
                        cov_num += 1
    sizes = sorted(len(g) for g in groups)
    return dict(ari=ari, purity=purity, cross_class_swap_frac=cross_frac,
                within_class_pair_coverage=cov_num / cov_den,
                n_groups=len(groups), n_singletons=sum(s == 1 for s in sizes),
                sizes_min_max=[sizes[0], sizes[-1]])


def oracle_groups():
    return [list(m) for m in MEMBERS]


def geom_groups(seed):
    E = build_embeddings(D, EPS).numpy()
    km = KMeans(n_clusters=N_CLASS, n_init=50, random_state=seed).fit(E[CONTENT])
    groups = [[] for _ in range(N_CLASS)]
    for t, l in zip(CONTENT, km.labels_):
        groups[int(l)].append(int(t))
    return [g for g in groups if g]


def random_groups(seed):
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(CONTENT)
    return [sorted(int(t) for t in perm[i:i + 4]) for i in range(0, len(perm), 4)]


def distill_groups(seed, log):
    """Train a rho=0 ERT on task M, then group words by its predicted final
    class (16-sample average at tau=0.3). No dictionary access."""
    torch.manual_seed(10_000 + seed); np.random.seed(10_000 + seed)
    E = build_embeddings(D, EPS).to(DEV)
    m = EnglishResidualTransformer(E, LAYERS, paraphrase=ParaphraseChannel(0.0, DEV),
                                   seq_len=SEQ_LEN).to(DEV)
    acc0 = train_M(m, DEV, STEPS, n_layers=LAYERS)
    log(f"  distill probe model (rho=0) task-M acc = {acc0:.3f}")
    m.eval()
    ids = torch.full((len(CONTENT), SEQ_LEN), PAD, dtype=torch.long, device=DEV)
    ids[:, RPOS] = torch.tensor(CONTENT, device=DEV)
    with torch.no_grad():
        logits = sum(_tok_to_class_logits(m.forward(ids, tau=0.3)[:, RPOS, :]).softmax(-1)
                     for _ in range(16))
    lab = logits.argmax(-1).cpu().numpy()
    groups = {}
    for t, l in zip(CONTENT, lab):
        groups.setdefault(int(l), []).append(int(t))
    return [sorted(g) for g in groups.values()], float(acc0)


def run_arm(name, groups, seed, tasks, R, log):
    torch.manual_seed(seed); np.random.seed(seed)
    E = build_embeddings(D, EPS).to(DEV)
    for task in tasks:
        if name == "oracle":
            chan = ParaphraseChannel(rho=RHO, device=DEV)
        else:
            chan = LearnedClassChannel(groups, rho=RHO, device=DEV)
        model = EnglishResidualTransformer(E, LAYERS, paraphrase=chan,
                                           seq_len=SEQ_LEN).to(DEV)
        trainer = train_M if task == "M" else train_S
        acc = trainer(model, DEV, STEPS, n_layers=LAYERS)
        n = 6000
        R["arms"][name]["runs"].append(dict(
            seed=seed, task=task, acc=float(acc),
            wilson95=wilson(round(acc * n), n)))
        log(f"  {name} seed{seed} task {task}: acc={acc:.3f}")


def main():
    R = dict(config=dict(d=D, layers=LAYERS, eps=EPS, steps=STEPS, rho=RHO,
                         B=256, seeds=SEEDS, eval_n=6000, device=DEV,
                         chance=dict(M=1 / N_CLASS, S=1 / V),
                         within_class_chance_S=0.25),
             arms={a: {"runs": [], "partitions": []} for a in
                   ["oracle", "distill", "geom", "random"]})

    # ---- primary arms first ----
    for seed in SEEDS:
        log(f"=== seed {seed}: PRIMARY arms ===")
        run_arm("oracle", oracle_groups(), seed, ["M", "S"], R, log)
        g, probe_acc = distill_groups(seed, log)
        pm = partition_metrics(g); pm["probe_model_M_acc"] = probe_acc; pm["seed"] = seed
        R["arms"]["distill"]["partitions"].append(pm)
        log(f"  distill partition: ARI={pm['ari']:.3f} purity={pm['purity']:.3f} "
            f"cross={pm['cross_class_swap_frac']:.3f} cov={pm['within_class_pair_coverage']:.3f}")
        run_arm("distill", g, seed, ["M", "S"], R, log)
        json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=2)

    # ---- secondary arms, budget-permitting ----
    for seed in SEEDS:
        if time.time() - T0 > BUDGET_S:
            R["skipped"] = f"secondary arms from seed {seed} (budget)"
            log(R["skipped"]); break
        log(f"=== seed {seed}: SECONDARY arms ===")
        g = geom_groups(seed)
        pm = partition_metrics(g); pm["seed"] = seed
        R["arms"]["geom"]["partitions"].append(pm)
        log(f"  geom partition: ARI={pm['ari']:.3f} purity={pm['purity']:.3f} "
            f"cross={pm['cross_class_swap_frac']:.3f}")
        run_arm("geom", g, seed, ["M", "S"], R, log)
        g = random_groups(seed)
        pm = partition_metrics(g); pm["seed"] = seed
        R["arms"]["random"]["partitions"].append(pm)
        run_arm("random", g, seed, ["M"], R, log)
        json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=2)

    # ---- summary + primary test ----
    def by(name, task):
        return [r["acc"] for r in R["arms"][name]["runs"] if r["task"] == task]

    summ = {}
    for a in R["arms"]:
        for t in ["M", "S"]:
            xs = by(a, t)
            if xs:
                summ[f"{a}_{t}"] = dict(mean=float(np.mean(xs)),
                                        min=float(np.min(xs)), max=float(np.max(xs)),
                                        per_seed=xs, n_seeds=len(xs))
    dM, dS = by("distill", "M"), by("distill", "S")
    primary_pass = (len(dM) == 3 and len(dS) == 3 and
                    all(m >= 0.8 for m in dM) and all(s <= 0.4 for s in dS))
    summ["primary_test"] = dict(
        criterion="all 3 seeds: distill M>=0.8 and S<=0.4 at rho=1",
        distill_M_per_seed=dM, distill_S_per_seed=dS, passed=bool(primary_pass))
    if len(dM) == 3:
        oM, oS = by("oracle", "M"), by("oracle", "S")
        summ["distill_minus_oracle"] = dict(
            M_per_seed=[dM[i] - oM[i] for i in range(3)],
            S_per_seed=[dS[i] - oS[i] for i in range(3)])
    R["summary"] = summ
    R["wall_time_s"] = time.time() - T0
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=2)
    log(f"DONE in {R['wall_time_s']:.0f}s  primary_pass={primary_pass}")


if __name__ == "__main__":
    main()
