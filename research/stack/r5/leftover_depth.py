"""The prime share among core-free members, and the PP share among core-open slots, by depth
u = ln n / ln t across a section (t = 6L+1, L the record); the independent-slot prediction of
twin-free stretches of length L per depth bin against the one that exists.
usage: uv run python research/stack/r5/leftover_depth.py base section [bins]
"""
import sys, math, json
import numpy as np
from sympy import nextprime, isprime
base=int(sys.argv[1]); k=int(sys.argv[2]); NB=int(sys.argv[3]) if len(sys.argv)>3 else 12
def sieve(n):
    s=np.ones(n+1,dtype=bool); s[:2]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=False
    return s
cuts=[base]; firsts=[base]
for _ in range(k):
    f=firsts[-1]; c=f*f; cuts.append(c); firsts.append(c if isprime(c) else nextprime(c))
lo,hi=cuts[k-1],cuts[k]; isp=sieve(hi+40)
n0=lo+((5-lo)%6); slots=np.arange(n0,hi,6,dtype=np.int64)
if slots[-1]+2>=hi: slots=slots[:-1]
S=slots.size; lowp=isp[slots]; upp=isp[slots+2]; twin=lowp&upp
idx=np.nonzero(twin)[0]; gaps=np.diff(np.concatenate([[-1],idx,[S]]))-1; L=int(gaps.max())
rec=int(np.argmax(gaps)); start=idx[rec-1]+1 if rec>0 else 0; t=6*L+1
core=[int(g) for g in (np.nonzero(isp[5:lo])[0]+5) if g<=t]
lowS=np.zeros(S,dtype=bool); upS=np.zeros(S,dtype=bool)
for g in core:
    inv=pow(6,-1,g); lowS[((0-n0)*inv)%g::g]=True; upS[(((-2)%g-n0)*inv)%g::g]=True
lowCF=~lowS; upCF=~upS; op=lowCF&upCF
def slide(a):
    cs=np.concatenate([[0],np.cumsum(a.astype(np.int32))]); return cs[L:]-cs[:-L]
K=slide(op); PP=slide(twin)
u=np.log(slots.astype(float))/math.log(t)
edges=np.linspace(u[0],u[-1]+1e-9,NB+1)
print(f"base {base} section [{lo},{hi}): L={L}, t={t}, record at n={slots[start]} depth {u[start]:.3f}")
print("bin | depth range | core-free members | prime share among them | core-open slots | PP share among them | starts in bin | predicted twin-free stretches (indep. slots, bin's PP share) | actual")
tot_pred=0
for b in range(NB):
    m=(u>=edges[b])&(u<edges[b+1])
    cf=int(lowCF[m].sum()+upCF[m].sum()); cfp=int((lowCF&lowp)[m].sum()+(upCF&upp)[m].sum())
    o=int(op[m].sum()); tw=int(twin[m].sum()); p=tw/max(1,o)
    ms=m[:K.size]
    pred=float(((1-p)**K[ms]).sum()) if ms.any() else 0.0; act=int((PP[ms]==0).sum())
    tot_pred+=pred
    print(f"{b} | {edges[b]:.3f}-{edges[b+1]:.3f} | {cf} | {cfp/max(1,cf):.4f} | {o} | {p:.4f} | {int(ms.sum())} | {pred:.2f} | {act}")
print(f"total predicted twin-free stretches {tot_pred:.2f}, actual {int((PP==0).sum())}")
# the record's own bin, and P(PP=0 | K = K_rec) there
b=int(np.searchsorted(edges,u[start],side='right')-1); m=(u>=edges[b])&(u<edges[b+1]); ms=m[:K.size]
o=int(op[m].sum()); tw=int(twin[m].sum()); p=tw/max(1,o); Kr=int(K[start])
sel=PP[ms&(K==Kr)]
print(f"record bin {b}: PP share {p:.4f}; starts with K={Kr} in bin {sel.size}; predicted twin-free among them {(1-p)**Kr*sel.size:.3f}; observed {int((sel==0).sum())}; PP|K hist {dict(zip(*[x.tolist() for x in np.unique(sel,return_counts=True)]))}")
# PP share by K inside the record's bin: where does the independent model's over-prediction come from
print("K | starts in record bin | mean PP share | predicted twin-free (bin share) | predicted (share at this K) | observed")
Kb=K[ms]; PPb=PP[ms]
for kk in range(int(Kb.min()),int(Kb.min())+22):
    m2=(Kb==kk)
    if not m2.any(): continue
    sh=float(PPb[m2].mean())/kk if kk else 1.0
    print(f"{kk} | {int(m2.sum())} | {sh:.4f} | {(1-p)**kk*int(m2.sum()):.3f} | {(1-sh)**kk*int(m2.sum()) if kk else 0:.3f} | {int((PPb[m2]==0).sum())}")
