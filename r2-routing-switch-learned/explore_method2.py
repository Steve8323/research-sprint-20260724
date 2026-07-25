"""
EXPLORATORY method 2 (not preregistered): a MODEL-BASED localization that doesn't need
the XOR-structure prior. Use the control model's first-layer weights: each hidden unit reads
some input columns. Cluster input columns by how the model's units couple them.

Idea: build an input-coupling matrix C[i,j] = sum_u |W1[u,i]|*|W1[u,j]| (two input dims are
"coupled" if the same hidden units read both). Signal dims that form an XOR pair must be read
together by the same units (a single ReLU unit can't do XOR, but the pair is jointly wired).
Then: (a) signal vs noise by total coupling / weight mass; (b) spectral/greedy split of signal
dims into 2 groups.

Also test: does the input-pair-probe method (method 1) survive when the control model is a
'hedger' (mid alignment) rather than a committed seed? It shouldn't matter -- method 1 uses
DATA only, not the model -- but we confirm it recovers cues regardless of seed outcome.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from itertools import combinations
torch.set_num_threads(2)
NBITS=2; NNOISE=3; INPUT=NBITS+NBITS+NNOISE; MARGIN=2.0; NOISE=0.6; N=800
H=8; WD=2e-3; LR=5e-3; EPOCHS=2000

def xor_block(t,rng):
    n=len(t); b=rng.integers(0,2,(n,NBITS)); b[:,-1]=np.bitwise_xor(b[:,0],t)
    return (MARGIN*(2*b-1)).astype(np.float32)+NOISE*rng.standard_normal((n,NBITS)).astype(np.float32)
def gen_align(n,seed,mode):
    rng=np.random.default_rng(seed); v=rng.integers(0,2,n); a=(1-v) if mode=="deploy" else v.copy()
    Xt=xor_block(v,rng); Xp=xor_block(a,rng); Z=NOISE*rng.standard_normal((n,NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt,Xp,Z],1).astype(np.float32)), torch.tensor(v.astype(np.int64))
class Net(nn.Module):
    def __init__(self,h=H):
        super().__init__(); self.l1=nn.Linear(INPUT,h); self.l2=nn.Linear(h,h); self.l3=nn.Linear(h,2)
    def forward(self,x): return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))
def train_control(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X,y=gen_align(N,seed,"train"); m=Net(); lf=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(m.parameters(),LR,weight_decay=WD)
    for _ in range(EPOCHS): opt.zero_grad(); lf(m(X),y).backward(); opt.step()
    return m,X,y

def method1_pairs(X,y):
    """input-pair XOR probes -> pick top-2 DISJOINT pairs."""
    Xn=X.numpy(); yn=y.numpy().astype(float); sc={}
    for i,j in combinations(range(INPUT),2):
        feat=np.stack([Xn[:,i],Xn[:,j],Xn[:,i]*Xn[:,j]],1); w=np.zeros(3); b=0.0
        for _ in range(300):
            p=1/(1+np.exp(-(feat@w+b))); w-=0.5*feat.T@(p-yn)/len(yn); b-=0.5*(p-yn).mean()
        sc[(i,j)]=(((feat@w+b)>0).astype(int)==yn).mean()
    order=sorted(sc,key=lambda k:-sc[k]); chosen=[]; used=set()
    for p in order:
        if p[0] not in used and p[1] not in used:
            chosen.append(p); used|=set(p)
        if len(chosen)==2: break
    return chosen, {p:round(sc[p],3) for p in order[:4]}

def method2_coupling(m):
    """model-based: input-coupling from first-layer weights -> signal dims & 2-way split."""
    W=m.l1.weight.detach().abs().numpy()   # [H, INPUT]
    mass=W.sum(0)                           # per-input weight mass
    C=W.T@W                                 # [INPUT,INPUT] coupling
    # signal dims = those with high weight mass (kmeans-1d style: top gap)
    order=np.argsort(-mass); sig=[]
    # keep dims whose mass > 0.5*max (heuristic)
    thr=0.5*mass.max(); sig=[d for d in range(INPUT) if mass[d]>thr]
    return mass, sig, C

# committed + hedger seeds (from round-1: control had 7 hi, 4 lo, 19 mid)
for seed in [700,701,703,708,715]:
    m,X,y=train_control(seed)
    Xd,yd=gen_align(4000,50001,"deploy")
    with torch.no_grad(): dep=(m(Xd).argmax(1)==yd).float().mean().item()
    pairs,top=method1_pairs(X,y)
    mass,sig,C=method2_coupling(m)
    print(f"seed {seed} deploy_align={dep:.2f}")
    print(f"  method1 chosen disjoint pairs={pairs} top={top}")
    print(f"  method2 mass={np.round(mass,2)} signal_dims={sig}")
    print(f"  method2 coupling[sig,:] cols {sig}:")
    for d in sig: print(f"    dim{d}:",np.round(C[d],2))
