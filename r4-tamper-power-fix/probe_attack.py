"""TIMING PROBE ONLY (pre-prereg): cost of one model x 9 attack cells at a dense probe grid."""
import time, math, sys
sys.path.insert(0,'/Users/steven/research_sprint_20260724/r3-switch-tamper')
import numpy as np, torch
import experiment as E
torch.set_num_threads(2)
Xdep,ydep=E.gen_align(4000,50001,"deploy"); Xhld,yhld=E.gen_align(4000,60001,"train")
Xa,yt=E.gen_align(200,91001,"deploy"); ya=1-yt
m,_,_=E.train_standard(3000)
for grid in ([0,2,5,10,20,30,50,75,100], list(range(0,21))+list(range(22,101,2))):
    E.PROBES=grid
    t=time.time(); r=E.attack_model(m,Xa,ya,Xdep,ydep,Xhld,yhld,3000); dt=time.time()-t
    print(f"grid n={len(grid)}: {dt:.3f}s for 9 cells -> {dt/9*1000:.1f} ms/attack")
