"""
Re-capture OLMoE routing per text (not aggregated), reusing the exact corpus
and hook from ~/moe_legibility/moe_route.py. Saves routing_per_text.pkl:
  texts: list of dicts {cat, ci, tokens (list[str]), routes {lid: (seq,topk) int array}}
Deterministic forward passes; no sampling. Read-only w.r.t. the source repo.
"""
import json
import pickle
import re
import sys

import numpy as np
import mlx.core as mx
import mlx_lm.models.olmoe as olmoe
from mlx_lm import load

SRC = "/Users/steven/moe_legibility/moe_route.py"
OUT = "/Users/steven/undefined/moe-profile-stability/routing_per_text.pkl"
MODEL = "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"

# ---- extract CORPUS exactly from moe_route.py (no transcription) ---------- #
src = open(SRC).read()
start = src.index("CORPUS = {")
end = src.index('print(f"loading')
ns = {}
exec(src[start:end], ns)
CORPUS = ns["CORPUS"]
cats = list(CORPUS)
print(f"corpus: {len(cats)} cats, {sum(len(v) for v in CORPUS.values())} texts")

# ---- identical routing-capture hook --------------------------------------- #
CAP = []
_orig = olmoe.OlmoeSparseMoeBlock.__call__
def patched(self, x):
    xf = x.reshape(-1, x.shape[-1])
    rw = mx.softmax(self.gate(xf), axis=1, precise=True)
    inds = mx.argpartition(-rw, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
    mx.eval(inds)
    CAP.append((self._lid, np.array(inds)))
    return _orig(self, x)
olmoe.OlmoeSparseMoeBlock.__call__ = patched

print(f"loading {MODEL} ...", flush=True)
model, tok = load(MODEL)
layers = getattr(model.model, "layers", None) or model.layers
for i, layer in enumerate(layers):
    mlp = getattr(layer, "mlp", None)
    if isinstance(mlp, olmoe.OlmoeSparseMoeBlock):
        mlp._lid = i
blk = next(l.mlp for l in layers if isinstance(getattr(l, "mlp", None), olmoe.OlmoeSparseMoeBlock))
NL = sum(1 for l in layers if isinstance(getattr(l, "mlp", None), olmoe.OlmoeSparseMoeBlock))
NE, TOPK = blk.num_experts, blk.top_k
print(f"layers={NL} experts={NE} top_k={TOPK}", flush=True)

texts_out = []
for ci, cat in enumerate(cats):
    for text in CORPUS[cat]:
        ids = tok.encode(text)
        toks = [tok.decode([t]) for t in ids]
        CAP.clear()
        mx.eval(model(mx.array([ids])))
        routes = {int(lid): inds.astype(np.int16) for lid, inds in CAP}
        assert len(routes) == NL and all(r.shape == (len(ids), TOPK) for r in routes.values())
        texts_out.append({"cat": cat, "ci": ci, "tokens": toks, "routes": routes})
    print(f"  done {cat}", flush=True)

meta = {"NL": NL, "NE": NE, "TOPK": TOPK, "cats": cats}
with open(OUT, "wb") as f:
    pickle.dump({"meta": meta, "texts": texts_out}, f)
print(f"saved {len(texts_out)} texts, {sum(len(t['tokens']) for t in texts_out)} tokens -> {OUT}")
