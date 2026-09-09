"""The covering structure of each section's record run: which gears of the machines below strike
each slot of the longest twin-free run, how many slots each gear covers, the distinct gears used,
the largest gear used, and the core/tail split (core = gears <= 6L + 1 for a run of L slots).
usage: uv run python research/stack/r4/record_cover.py [LIMIT]
"""
import os, sys
import numpy as np
from sympy import primefactors
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results"); os.makedirs(RES,exist_ok=True)
LIMIT=int(sys.argv[1]) if len(sys.argv)>1 else 300_000_000
def sieve(n):
    s=np.ones(n+1,dtype=bool); s[:2]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=False
    return s
def nextprime(n,isp):
    m=n+1
    while not isp[m]: m+=1
    return m
isp=sieve(LIMIT+40); twin=isp.copy(); twin[:-2]&=isp[2:]; twin[-2:]=False
out=[]
for base in [3,5,7,11,13,17,19,23,29,31]:
    cuts=[base]; firsts=[base]
    while True:
        c=firsts[-1]**2
        if c>LIMIT: break
        cuts.append(c); firsts.append(nextprime(c-1,isp) if isp[c] else nextprime(c,isp))
    for k in range(1,len(cuts)):
        lo=cuts[k]; hi=cuts[k+1] if k+1<len(cuts) else None
        if hi is None or hi>LIMIT: break
        n0=lo+((5-lo)%6); slots=np.arange(n0,hi,6,dtype=np.int64); tw=twin[slots]; idx=np.nonzero(tw)[0]
        gaps=np.diff(np.concatenate([[-1],idx,[slots.size]]))-1; j=int(np.argmax(gaps)); L=int(gaps[j])
        start=idx[j-1]+1 if j>0 else 0
        run=slots[start:start+L]
        cover={}  # gear -> slots it strikes in the run (a slot is struck by g if g | n or g | n+2)
        for n in run:
            n=int(n); gs=set(p for p in primefactors(n) if p<lo)|set(p for p in primefactors(n+2) if p<lo)
            for g in gs: cover[g]=cover.get(g,0)+1
        gears=sorted(cover); core=[g for g in gears if g<=6*L+1]; tail=[g for g in gears if g>6*L+1]
        top=sorted(cover.items(),key=lambda kv:-kv[1])[:8]
        out.append(f"base {base} section {k+1} [{lo},{hi}): record run L={L} slots at {int(run[0])}; distinct gears used {len(gears)}, largest {gears[-1]}; core (g<=6L+1={6*L+1}) {len(core)} gears covering {sum(cover[g] for g in core)} slot-strikes, tail {len(tail)} gears covering {sum(cover[g] for g in tail)} (max per tail gear {max([cover[g] for g in tail]) if tail else 0}); top coverers {top}")
txt="\n".join(out); print(txt); open(os.path.join(RES,f"record_cover_{LIMIT}.txt"),"w",encoding="utf-8").write(txt+"\n")
