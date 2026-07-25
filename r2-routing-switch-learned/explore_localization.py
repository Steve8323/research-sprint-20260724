"""
EXPLORATORY (not preregistered): what can we learn about cue localization from an
UNROUTED (standard) control model + the agreeing training data alone?

The oracle in round-1 knows CUE_A=[0,1], CUE_B=[2,3], noise=[4,5,6]. We want to recover:
  (1) which input columns carry signal vs noise
  (2) that signal columns cluster into two disjoint XOR pairs (the two cues)
without being told. Whether pair {0,1} is "true" vs {2,3} is "proxy" is unidentifiable from
agreeing data (both predict the label); that is a separate labeling question.

Methods tried here:
  A. input gradient-attribution magnitude per input dim (|dL/dx| avg) on the control model
  B. single-dim linear-probe accuracy for the label (should be ~chance; XOR needs pairs)
  C. pairwise "XOR-pair" score: for each pair (i,j), how well does sign(x_i)*sign(x_j) or
     a 2-dim logistic probe predict the label -> the two true cue pairs should pop out
  D. gradient-covariance / co-attribution clustering of signal dims into 2 groups
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from itertools import combinations
torch.set_num_threads(2)

NBITS=2; NNOISE=3; INPUT=NBITS+NBITS+NNOISE; MARGIN=2.0; NOISE=0.6; N=800
H=8; WD=2e-3; LR=5e-3; EPOCHS=2000

def xor_block(target, rng):
    n=len(target); b=rng.integers(0,2,(n,NBITS)); b[:,-1]=np.bitwise_xor(b[:,0],target)
    return (MARGIN*(2*b-1)).astype(np.float32)+NOISE*rng.standard_normal((n,NBITS)).astype(np.float32)

def gen_align(n,seed,mode):
    rng=np.random.default_rng(seed); v=rng.integers(0,2,n)
    a=(1-v) if mode=="deploy" else v.copy()
    Xt=xor_block(v,rng); Xp=xor_block(a,rng); Z=NOISE*rng.standard_normal((n,NNOISE)).astype(np.float32)
    return torch.tensor(np.concatenate([Xt,Xp,Z],1).astype(np.float32)), torch.tensor(v.astype(np.int64))

class Net(nn.Module):
    def __init__(self,h=H):
        super().__init__(); self.l1=nn.Linear(INPUT,h); self.l2=nn.Linear(h,h); self.l3=nn.Linear(h,2)
    def forward(self,x):
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

def train_control(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    X,y=gen_align(N,seed,"train"); m=Net(); lf=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(m.parameters(),LR,weight_decay=WD)
    for _ in range(EPOCHS): opt.zero_grad(); lf(m(X),y).backward(); opt.step()
    return m,X,y

for seed in [700,701,705]:
    m,X,y=train_control(seed)
    with torch.no_grad(): acc=(m(X).argmax(1)==y).float().mean().item()
    # A. input gradient attribution magnitude
    Xg=X.clone().requires_grad_(True)
    logit=m(Xg); loss=F.cross_entropy(logit,y); g,=torch.autograd.grad(loss,Xg)
    attr=g.abs().mean(0).detach().numpy()
    # B. single-dim label-probe acc (train a 1-feature logistic per dim)
    Xn=X.numpy(); yn=y.numpy()
    single=[]
    for d in range(INPUT):
        w=np.polyfit(Xn[:,d],yn,1)  # crude
        pred=(np.polyval(w,Xn[:,d])>0.5).astype(int); single.append((pred==yn).mean())
    # C. pairwise 2-dim logistic probe acc (captures XOR via product feature)
    pair_acc={}
    for i,j in combinations(range(INPUT),2):
        feat=np.stack([Xn[:,i],Xn[:,j],Xn[:,i]*Xn[:,j]],1)
        # closed-form-ish: fit logistic with a few GD steps
        w=np.zeros(3); b=0.0
        for _ in range(300):
            z=feat@w+b; p=1/(1+np.exp(-z)); grad=feat.T@(p-yn)/len(yn); gb=(p-yn).mean()
            w-=0.5*grad; b-=0.5*gb
        pred=((feat@w+b)>0).astype(int); pair_acc[(i,j)]=(pred==yn).mean()
    top_pairs=sorted(pair_acc.items(),key=lambda kv:-kv[1])[:4]
    print(f"\nseed {seed}  train_acc={acc:.3f}")
    print("  input |dL/dx| per dim:", np.round(attr,4))
    print("  single-dim label acc :", np.round(single,3))
    print("  top XOR-pair probes  :", [(p,round(a,3)) for p,a in top_pairs])
