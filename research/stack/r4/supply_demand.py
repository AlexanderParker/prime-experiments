"""Supply against demand on a section: for windows of L consecutive slots (L = the section's
record run), the CORE's leftover K(w) = slots not struck by gears <= 6L+1 (demand), and the TAIL's
strikes T(w) = number of strikes by gears in (6L+1, cut) inside the window (supply; a tail gear
strikes at most one slot of the window). A twin-free window needs the tail to cover every leftover.
usage: uv run python research/stack/r4/supply_demand.py base sectionindex LIMIT
"""
import sys, os
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results"); os.makedirs(RES,exist_ok=True)
base=int(sys.argv[1]); k=int(sys.argv[2]); LIMIT=int(sys.argv[3])
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
lo,hi=cuts[k-1],cuts[k]
n0=lo+((5-lo)%6); slots=np.arange(n0,hi,6,dtype=np.int64); S=slots.size
twin=isp[slots]&isp[slots+2]
idx=np.nonzero(twin)[0]; gaps=np.diff(np.concatenate([[-1],idx,[S]]))-1; L=int(gaps.max())
gears=np.nonzero(isp[5:lo])[0]+5
core=[int(g) for g in gears if g<=6*L+1]; tail=[int(g) for g in gears if g>6*L+1]
def strikes(gs):
    a=np.zeros(S,dtype=np.int16)
    for g in gs:
        for r in (0,(-2)%g):
            start=(r-n0)%g
            # slots n = n0+6j ; n ≡ r mod g  -> j ≡ (r-n0)*6^{-1} mod g
            inv=pow(6,-1,g); j0=((r-n0)*inv)%g
            a[j0::g]+=1
    return a
corehit=strikes(core)>0
tailhit=strikes(tail)   # count of tail strikes per slot (>=1 possible)
left=(~corehit).astype(np.int32)
cs=np.concatenate([[0],np.cumsum(left)]); K=cs[L:]-cs[:-L]
ts=np.concatenate([[0],np.cumsum(tailhit.astype(np.int64))]); T=ts[L:]-ts[:-L]
open_=(~corehit)&(tailhit==0)          # slots open under all gears below the cut = twins
oc=np.concatenate([[0],np.cumsum(open_.astype(np.int32))]); O=oc[L:]-oc[:-L]
rec=int(np.argmax(gaps)); start=idx[rec-1]+1 if rec>0 else 0
print(f"base {base} section {k} [{lo},{hi}): slots {S}, record L={L} at slot index {start}; core gears {len(core)} (<= {6*L+1}), tail gears {len(tail)} (to {lo})")
print(f"core leftover K over all windows of L: mean {K.mean():.1f}, 99% {np.percentile(K,99):.0f}, max {K.max()} ; at the record window K={K[start]}")
print(f"tail strikes T over windows: mean {T.mean():.1f}, min {T.min()}, 1% {np.percentile(T,1):.0f} ; at the record window T={T[start]}")
print(f"windows with T >= K (supply enough on paper): {int((T>=K).sum())} of {K.size} = {(T>=K).mean():.4f}; twin-free windows (tail covers every leftover exactly): {int((O==0).sum())}")
print(f"typical leftover fraction {K.mean()/L:.4f} vs record {K[start]/L:.4f}; typical open per window {O.mean():.2f}")
