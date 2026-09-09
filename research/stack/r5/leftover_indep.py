"""Are the leftover slots' types independent across a stretch?  Test 1: the distribution of PP
among the K core-open slots of a stretch against Binomial(K, p_bin), p_bin = the PP share among
core-open slots in the stretch's depth bin, by K, in the record's bin.  Test 2: at shorter lengths
L' (core kept at t = 6L+1, L the record), the predicted number of twin-free starts sum (1-p_bin)^K
and of twin-free RUNS against the observed counts (a twin-free start = a stretch with PP = 0; a run =
a maximal set of consecutive twin-free starts = a twin gap >= L').
usage: uv run python research/stack/r5/leftover_indep.py base section [bins]
"""
import sys, math
import numpy as np
from math import comb
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
S=slots.size; twin=isp[slots]&isp[slots+2]
idx=np.nonzero(twin)[0]; gaps=np.diff(np.concatenate([[-1],idx,[S]]))-1; L=int(gaps.max())
rec=int(np.argmax(gaps)); start=idx[rec-1]+1 if rec>0 else 0; t=6*L+1
core=[int(g) for g in (np.nonzero(isp[5:lo])[0]+5) if g<=t]
lowS=np.zeros(S,dtype=bool); upS=np.zeros(S,dtype=bool)
for g in core:
    inv=pow(6,-1,g); lowS[((0-n0)*inv)%g::g]=True; upS[(((-2)%g-n0)*inv)%g::g]=True
op=(~lowS)&(~upS)
u=np.log(slots.astype(float))/math.log(t); edges=np.linspace(u[0],u[-1]+1e-9,NB+1)
binof=np.clip(np.searchsorted(edges,u,side='right')-1,0,NB-1)
pbin=np.zeros(NB)
for b in range(NB):
    m=binof==b; o=op[m].sum(); pbin[b]=twin[m].sum()/max(1,o)
def slide(a,Lw):
    cs=np.concatenate([[0],np.cumsum(a.astype(np.int32))]); return cs[Lw:]-cs[:-Lw]
print(f"base {base} section [{lo},{hi}): L={L}, t={t}; PP share by bin: {np.round(pbin,4).tolist()}")
# Test 1: PP | K against Binomial(K, p_bin) in the record's bin
K=slide(op,L); PP=slide(twin,L); rb=int(binof[start]); p=pbin[rb]; inb=(binof[:K.size]==rb)
print(f"Test 1, record bin {rb} (PP share {p:.4f}): observed count of starts with PP = j among the K leftovers against Binomial(K, p) x starts")
for kk in sorted(set([max(1,int(K[start])-4),int(K[start])-2,int(K[start]),int(K[start])+2,int(K[start])+4])):
    m=inb&(K==kk); n=int(m.sum())
    if n==0: continue
    obs=np.bincount(PP[m],minlength=kk+1)
    exp=[n*comb(kk,j)*p**j*(1-p)**(kk-j) for j in range(kk+1)]
    print(f"K={kk}, starts {n}: j : observed / binomial -> "+", ".join(f"{j}: {int(obs[j])}/{exp[j]:.1f}" for j in range(min(kk,6)+1))+f" ... {kk}: {int(obs[kk])}/{exp[kk]:.1f}")
# Test 2: shorter lengths, twin-free starts and runs, predicted against observed
print("Test 2: L' | mean K | predicted twin-free starts | observed twin-free starts | observed twin-free runs (twin gaps >= L') | predicted runs (starts x twin density per slot, i.e. predicted gaps of exactly-extending length)")
dens=twin.sum()/S
for Lp in [L//4, L//3, L//2, 2*L//3, 3*L//4, L-100, L-50, L-20, L]:
    if Lp<10: continue
    Kp=slide(op,Lp); PPp=slide(twin,Lp); pb=pbin[binof[:Kp.size]]
    pred=float(((1-pb)**Kp).sum()); obs=int((PPp==0).sum()); runs=int((gaps>=Lp).sum())
    # a run of twin-free starts of length r corresponds to a twin gap of Lp+r-1 slots; under the model the
    # expected number of runs is approximately pred x (probability the next slot is a twin) = pred x twin density
    print(f"{Lp} | {Kp.mean():.1f} | {pred:.1f} | {obs} | {runs} | {pred*dens:.2f}")
