"""The step's reduction, measured: inside section k+1 the slots struck by machines 1..k are all
slots except the twins (band structure), so the composite record of machines 1..k on the section
= the longest run of consecutive twin-free slots inside the section.  Per chain link: section
length in slots, the record (in slots), its position, record / length, and the first twin.
usage: uv run python research/stack/r4/composite_record.py [LIMIT]
"""
import os, sys, time
import numpy as np
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
t0=time.time(); isp=sieve(LIMIT+40); twin=isp.copy(); twin[:-2]&=isp[2:]; twin[-2:]=False
out=[f"# composite record per section (longest twin-free run of slots inside the section), sieve to {LIMIT}"]
for base in [3,5,7,11,13,17,19,23,29,31]:
    cuts=[base]; firsts=[base]
    while True:
        c=firsts[-1]**2
        if c>LIMIT: break
        cuts.append(c); firsts.append(nextprime(c-1,isp) if isp[c] else nextprime(c,isp))
    out.append(f"\n# chain from base {base}: cuts {cuts}")
    prev=None
    for k in range(1,len(cuts)):
        lo=cuts[k]; hi=cuts[k+1] if k+1<len(cuts) else None
        if hi is None or hi>LIMIT: break
        n0=lo+((5-lo)%6); slots=np.arange(n0,hi,6,dtype=np.int64)
        tw=twin[slots]
        idx=np.nonzero(tw)[0]
        if idx.size==0:
            out.append(f"section {k+1} [{lo},{hi}): NO TWIN"); continue
        gaps=np.diff(np.concatenate([[-1],idx,[slots.size]]))-1  # twin-free runs including the ends
        j=int(np.argmax(gaps)); rec=int(gaps[j])
        pos=int(slots[idx[j-1]+1]) if j>0 else int(slots[0])
        first=int(slots[idx[0]])
        out.append(f"section {k+1} [{lo},{hi}): length {slots.size} slots, twins {idx.size}, composite record {rec} slots ({rec*6} numbers) at {pos}, record/length {rec/slots.size:.2e}, first twin {first} (+{first-lo}), prev-section record {prev}")
        prev=rec
    print(f"base {base} done {time.time()-t0:.0f}s",flush=True)
txt="\n".join(out); print(txt); open(os.path.join(RES,f"composite_record_{LIMIT}.txt"),"w",encoding="utf-8").write(txt+"\n")
