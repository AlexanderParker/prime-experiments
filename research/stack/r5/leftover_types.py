"""What the core's leftover slots are. On a section, for a stretch of L slots, the core (gears
<= 6L+1) leaves K slots; each member of a leftover slot has no prime factor <= 6L+1, so it is a
prime, or a product of >= 2 primes above 6L+1.  Classify the leftover slots of the record stretch
and of random stretches by member types: PP (both prime = a twin), P+C (one prime, one composite
of large primes), C+C.  The stretch is twin-free iff it has no PP.
usage: uv run python research/stack/r5/leftover_types.py base section LIMIT nsample
"""
import sys, os, random
import numpy as np
from sympy import factorint
base=int(sys.argv[1]); k=int(sys.argv[2]); LIMIT=int(sys.argv[3]); NS=int(sys.argv[4]) if len(sys.argv)>4 else 300
def sieve(n):
    s=np.ones(n+1,dtype=bool); s[:2]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=False
    return s
def nextprime(n,isp):
    m=n+1
    while not isp[m]: m+=1
    return m
isp=sieve(LIMIT+40)
cuts=[base]; firsts=[base]
while True:
    c=firsts[-1]**2
    if c>LIMIT: break
    cuts.append(c); firsts.append(nextprime(c-1,isp) if isp[c] else nextprime(c,isp))
lo,hi=cuts[k-1],cuts[k]; n0=lo+((5-lo)%6); slots=np.arange(n0,hi,6,dtype=np.int64); S=slots.size
twin=isp[slots]&isp[slots+2]; idx=np.nonzero(twin)[0]; gaps=np.diff(np.concatenate([[-1],idx,[S]]))-1; L=int(gaps.max())
rec=int(np.argmax(gaps)); start=idx[rec-1]+1 if rec>0 else 0
core=[int(g) for g in (np.nonzero(isp[5:lo])[0]+5) if g<=6*L+1]
corehit=np.zeros(S,dtype=bool)
for g in core:
    inv=pow(6,-1,g)
    for r in (0,(-2)%g):
        corehit[((r-n0)*inv)%g::g]=True
def types(st):
    out={"PP":0,"P+C":0,"C+C":0}
    for j in range(st,st+L):
        if corehit[j]: continue
        n=int(slots[j]); a=isp[n]; b=isp[n+2]
        if a and b: out["PP"]+=1
        elif a or b: out["P+C"]+=1
        else: out["C+C"]+=1
    return out
print(f"base {base} section {k}: L={L}, core gears {len(core)} (<= {6*L+1}); record stretch at slot {start} (n={int(slots[start])}): leftover types {types(start)}")
random.seed(0); agg={"PP":0,"P+C":0,"C+C":0}; tf=0
for _ in range(NS):
    st=random.randrange(0,S-L); t=types(st)
    for kk in agg: agg[kk]+=t[kk]
    if t["PP"]==0: tf+=1
print(f"{NS} random stretches: mean leftover types per stretch PP {agg['PP']/NS:.2f}, P+C {agg['P+C']/NS:.2f}, C+C {agg['C+C']/NS:.2f}; twin-free among them {tf}")
# the composite members of the record stretch: how many large prime factors
comp=[]
for j in range(start,start+L):
    if corehit[j]: continue
    for m in (int(slots[j]),int(slots[j])+2):
        if not isp[m]: comp.append((m,sorted(factorint(m).keys())))
print("record stretch composite members (m, prime factors):", comp[:12])
