"""TIMING/EQUIVALENCE PROBE ONLY -- run BEFORE prereg. No arm, metric or test is evaluated here.
Purpose: (1) how fast is one sequential training on this loaded box, (2) does a vectorised
batched trainer reproduce the sequential one, (3) how fast is the batched trainer."""
import time, math, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
NBITS, NNOISE = 2, 3; INPUT = 7
MARGIN, NOISE, N = 2.0, 0.6, 800
H, WD, LR, EPOCHS = 8, 2e-3, 5e-3, 2000

def xor_block(target, rng, margin=MARGIN, noise=NOISE):
    n = len(target); b = rng.integers(0,2,(n,NBITS)); b[:,-1] = np.bitwise_xor(b[:,0], target)
    return (margin*(2*b-1)).astype(np.float32) + noise*rng.standard_normal((n,NBITS)).astype(np.float32)

def gen_align(n, seed, mode, margin=MARGIN, noise=NOISE, dnoise=None, offset=0.0):
    if dnoise is None: dnoise = noise
    rng = np.random.default_rng(seed); v = rng.integers(0,2,n)
    a = (1-v) if mode=="deploy" else v.copy()
    Xt = xor_block(v,rng,margin,noise); Xp = xor_block(a,rng,margin,noise)
    Z = dnoise*rng.standard_normal((n,NNOISE)).astype(np.float32)
    X = np.concatenate([Xt,Xp,Z],1).astype(np.float32) + np.float32(offset)
    return torch.tensor(X), torch.tensor(v.astype(np.int64))

class Net(nn.Module):
    def __init__(self, h=H):
        super().__init__(); self.h=h
        self.l1=nn.Linear(INPUT,h); self.l2=nn.Linear(h,h); self.l3=nn.Linear(h,2)
    def forward(self,x,umask=None):
        h=F.relu(self.l1(x))
        if umask is not None: h=h*umask
        h=F.relu(self.l2(h))
        if umask is not None: h=h*umask
        return self.l3(h)

def train_standard(seed, h=H, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    X,y = gen_align(N,seed,"train")
    m=Net(h); lf=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(m.parameters(),LR,weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad(); lf(m(X),y).backward(); opt.step()
    return m

def accm(m,X,y,umask=None):
    with torch.no_grad(): return (m(X,umask).argmax(1)==y).float().mean().item()

# ---------------- batched trainer ----------------
def batched_init(seeds, h):
    W1=[];b1=[];W2=[];b2=[];W3=[];b3=[]
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m=Net(h)
        W1.append(m.l1.weight.data.T.clone()); b1.append(m.l1.bias.data.clone())
        W2.append(m.l2.weight.data.T.clone()); b2.append(m.l2.bias.data.clone())
        W3.append(m.l3.weight.data.T.clone()); b3.append(m.l3.bias.data.clone())
    P=[torch.stack(W1),torch.stack(b1).unsqueeze(1),torch.stack(W2),torch.stack(b2).unsqueeze(1),
       torch.stack(W3),torch.stack(b3).unsqueeze(1)]
    for p in P: p.requires_grad_(True)
    return P

def bfwd(P,X,umask=None):
    W1,b1,W2,b2,W3,b3=P
    h=F.relu(torch.baddbmm(b1,X,W1))
    if umask is not None: h=h*umask
    h=F.relu(torch.baddbmm(b2,h,W2))
    if umask is not None: h=h*umask
    return torch.baddbmm(b3,h,W3)

def batched_train_standard(seeds,h=H,epochs=EPOCHS):
    P=batched_init(seeds,h)
    Xs=[];ys=[]
    for s in seeds:
        X,y=gen_align(N,s,"train"); Xs.append(X); ys.append(y)
    X=torch.stack(Xs); y=torch.stack(ys)
    B=len(seeds)
    opt=torch.optim.Adam(P,LR,weight_decay=WD)
    for _ in range(epochs):
        opt.zero_grad()
        o=bfwd(P,X)
        loss=F.cross_entropy(o.reshape(B*N,2),y.reshape(B*N),reduction='none').view(B,N).mean(1).sum()
        loss.backward(); opt.step()
    return [p.detach() for p in P]

def bacc(P,X,y,umask=None):
    with torch.no_grad():
        B=P[0].shape[0]
        Xb=X.unsqueeze(0).expand(B,-1,-1)
        o=bfwd(P,Xb,umask)
        return (o.argmax(2)==y.unsqueeze(0)).float().mean(1).numpy()

if __name__=="__main__":
    Xdep,ydep=gen_align(4000,50001,"deploy"); Xhld,yhld=gen_align(4000,60001,"train")
    t=time.time(); seq=[]
    for s in [3000,3001,3002]:
        m=train_standard(s); seq.append((accm(m,Xdep,ydep),accm(m,Xhld,yhld)))
    tseq=(time.time()-t)/3
    print(f"sequential H=8 per model: {tseq:.3f}s  -> {seq}")
    for B in (16,64):
        sd=list(range(3000,3000+B)); t=time.time(); P=batched_train_standard(sd); tb=time.time()-t
        da=bacc(P,Xdep,ydep); ha=bacc(P,Xhld,yhld)
        print(f"batched B={B}: {tb:.2f}s total, {tb/B:.4f}s/model, speedup {tseq/(tb/B):.1f}x")
        if B==16:
            print("  agreement on first 3 (deploy,agree):",[(round(float(da[i]),4),round(float(ha[i]),4)) for i in range(3)])
            print("  seq                                 :",[(round(a,4),round(b,4)) for a,b in seq])
            print("  max|diff| deploy:",max(abs(float(da[i])-seq[i][0]) for i in range(3)))
    t=time.time(); P=batched_train_standard(list(range(5000,5064)),h=4); print(f"batched H=4 B=64: {time.time()-t:.2f}s")
