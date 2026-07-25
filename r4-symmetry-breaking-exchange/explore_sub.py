"""EXPLORATORY (not confirmatory): is a signal-subspace-restricted L-SUP less degenerate than dict?"""
import sys, time; sys.path.insert(0,'/Users/steven/research_sprint_20260724/r4-symmetry-breaking-exchange')
import numpy as np, torch, torch.nn as nn
import sup as S, core as C

def localize_sup_sub(X, y, seed, sup=None, lam=1.0, steps=400):
    Xn=X.numpy(); yv=y.numpy().astype(np.float64); mu=Xn.mean(0); X0=Xn-mu
    Vs,_=C.signal_subspace(X0,yv,4)
    Z=X0@Vs; Cm=Z.T@Z/len(Z); ew,ev=np.linalg.eigh(Cm); Wh=ev@np.diag(ew**-0.5)@ev.T
    Zw=torch.tensor((Z@Wh).astype(np.float32)); yt=torch.tensor(yv.astype(np.float32))
    have=sup is not None and len(sup["X"])>0
    if have:
        Zs=torch.tensor((((sup["X"].numpy()-mu)@Vs)@Wh).astype(np.float32))
        ys=sup.get("y"); ys=None if ys is None else ys.float()
    torch.manual_seed(seed+31337)
    U=torch.randn(4,4,requires_grad=True)
    w1=torch.zeros(5,requires_grad=True); b1=torch.zeros(1,requires_grad=True)
    w2=torch.zeros(5,requires_grad=True); b2=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([U,w1,b1,w2,b2],lr=0.05); bce=nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad(); Q,_=torch.linalg.qr(U)
        s=Zw@Q; ssd=s.std(0)+1e-6; s=s/ssd
        z1=S._feats(s,0,1)@w1+b1; z2=S._feats(s,2,3)@w2+b2
        loss=bce(z1,yt)+bce(z2,yt)
        if have:
            ss=(Zs@Q)/ssd
            zs1=S._feats(ss,0,1)@w1+b1; zs2=S._feats(ss,2,3)@w2+b2
            loss=loss+lam*(bce(zs1,ys)+bce(zs2,1.0-ys))
        loss.backward(); opt.step()
    with torch.no_grad(): Q,_=torch.linalg.qr(U)
    G=Wh@Q.numpy().astype(np.float64); Sf=Z@G
    return C._package(Vs,G,Sf,[0,1],[2,3],mu,dict())

t0=time.time()
for seed in [700,701,702,703]:
    T=C.make_transform('rot',seed); orb=C.orbit_subspaces(T)
    X,y=C.gen_align(C.N,seed,'train'); X=C.apply_T(X,T)
    row=[]
    for m in [0,1,2,4,8,32]:
        if m==0: sp=None
        else:
            Xd,yd,_,_=S.gen_align_lat(m,80000+seed,'deploy'); Xd=C.apply_T(Xd,T)
            sp=dict(kind='dep_lab',X=Xd,y=yd)
        l=localize_sup_sub(X,y,seed,sp); eA,eB=C.loc_subspaces(l)
        row.append((m,round(C.split_fidelity(eA,eB,*orb['true']),3),round(C.split_fidelity(eA,eB,*orb['entangled']),3)))
    print(seed,row)
print('time',round(time.time()-t0,1))
