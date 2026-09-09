"""Spot checks for the lateral pass over the core's leftover (dead_branches_reopened_4.md).
On a section [c, c') of a chain, with L the record twin-free run (slots) and t = 6L+1, the core is
the primes in [5, t], the tail the primes in (t, c).  For every stretch of L slots: K = core-open
slots; CF = core-free members (numbers coprime to 6 with no prime factor <= t); CFp / CFc = prime /
composite core-free members; comp_in_open = composite members sitting in core-open slots; PP/PC/CC
the leftover types.  At the record: the depth u = ln x / ln t of its ends, its composite members
factored (air = smaller factor, fuel = larger), tail-or-exhaust fuel, distinctness.
usage: uv run python research/stack/r5/leftover_shadow.py base section [nrand]
"""
import sys, math, json
import numpy as np
from sympy import factorint
base=int(sys.argv[1]); k=int(sys.argv[2]); NR=int(sys.argv[3]) if len(sys.argv)>3 else 2000
def sieve(n):
    s=np.ones(n+1,dtype=bool); s[:2]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=False
    return s
def cuts_of(base,k):
    # cut k needs primes to sqrt of cut k, tiny sieve
    from sympy import nextprime, isprime
    cuts=[base]; firsts=[base]
    for _ in range(k):
        f=firsts[-1]; c=f*f; cuts.append(c); firsts.append(c if isprime(c) else nextprime(c))
    return cuts
cuts=cuts_of(base,k); lo,hi=cuts[k-1],cuts[k]
isp=sieve(hi+40)
n0=lo+((5-lo)%6); slots=np.arange(n0,hi,6,dtype=np.int64)
if slots[-1]+2>=hi: slots=slots[:-1]      # drop the slot holding the next cut (core_leftover.py convention)
S=slots.size
lowp=isp[slots]; upp=isp[slots+2]; twin=lowp&upp
idx=np.nonzero(twin)[0]; gaps=np.diff(np.concatenate([[-1],idx,[S]]))-1; L=int(gaps.max())
rec=int(np.argmax(gaps)); start=idx[rec-1]+1 if rec>0 else 0
t=6*L+1
gears=(np.nonzero(isp[5:lo])[0]+5).astype(int)
core=[int(g) for g in gears if g<=t]; tail=[int(g) for g in gears if g>t]
lowS=np.zeros(S,dtype=bool); upS=np.zeros(S,dtype=bool)
for g in core:
    inv=pow(6,-1,g)
    lowS[((0-n0)*inv)%g::g]=True
    upS[(((-2)%g-n0)*inv)%g::g]=True
lowCF=~lowS; upCF=~upS; openslot=lowCF&upCF
lowCFc=lowCF&~lowp; upCFc=upCF&~upp        # composite core-free members
def slide(a):
    cs=np.concatenate([[0],np.cumsum(a.astype(np.int32))]); return cs[L:]-cs[:-L]
K=slide(openslot)
CF=slide(lowCF)+slide(upCF)
CFc=slide(lowCFc)+slide(upCFc)
CFp=CF-CFc
comp_in_open=slide(openslot&lowCFc)+slide(openslot&upCFc)
PP=slide(twin)
CC=slide(openslot&lowCFc&upCFc)
PC=K-PP-CC
# composite core-free members whose partner is core-free (i.e. sitting in an open slot)
CFc_total=int(lowCFc.sum()+upCFc.sum()); CFc_open=int((openslot&lowCFc).sum()+(openslot&upCFc).sum())
def stats(a,name):
    return {"name":name,"mean":float(a.mean()),"sd":float(a.std()),"min":int(a.min()),"max":int(a.max()),"at_record":int(a[start])}
out={"base":base,"section":[int(lo),int(hi)],"slots":S,"L":L,"t":t,"core":len(core),"tail":len(tail),
     "record_start_n":int(slots[start]),"record_end_n":int(slots[start+L-1]+2),
     "depth_start":math.log(slots[start])/math.log(t),"depth_end":math.log(slots[start+L-1]+2)/math.log(t),
     "section_depth":[math.log(lo)/math.log(t),math.log(hi)/math.log(t)],
     "t2":t*t,"t3":t**3,"members_below_t3":bool(hi<=t**3),
     "stats":[stats(K,"K"),stats(CF,"CF"),stats(CFp,"CFp"),stats(CFc,"CFc"),stats(comp_in_open,"comp_in_open"),
              stats(PP,"PP"),stats(PC,"PC"),stats(CC,"CC")],
     "CFc_total":CFc_total,"CFc_in_open_total":CFc_open,"CFc_open_fraction":CFc_open/max(1,CFc_total)}
# PP conditional on K = K(record): the exact distribution over all starts with that K
Kr=int(K[start]); sel=PP[K==Kr]
hist={int(v):int(c) for v,c in zip(*np.unique(sel,return_counts=True))}
out["PP_given_K_record"]={"K":Kr,"starts":int(sel.size),"hist":hist,"mean":float(sel.mean()) if sel.size else None}
# P(PP=0 | K) for small K, against (1-p)^K with p = fraction of open slots that are PP over the section
p_pp=float(twin.sum()/max(1,openslot.sum()))
cond=[]
for kk in range(0,max(Kr,1)+3):
    m=(K==kk)
    if m.sum(): cond.append((kk,int(m.sum()),int((PP[m]==0).sum()),(1-p_pp)**kk*int(m.sum())))
out["p_pp"]=p_pp; out["PP0_given_K"]=cond
# twin count per stretch: z of 0
out["PP_z_of_zero"]=float(-PP.mean()/PP.std())
# the record's composite members
comps=[]
for j in range(start,start+L):
    if not openslot[j]: continue
    for m in (int(slots[j]),int(slots[j])+2):
        if not isp[m]:
            f=sorted(factorint(m).items()); pr=[p for p,e in f for _ in range(e)]
            comps.append({"n":m,"factors":pr,"air":pr[0],"fuel":pr[-1],"fuel_in_tail":pr[-1]<lo,"partner_prime":bool(isp[m+2] if m%6==5 else isp[m-2])})
airs=[c["air"] for c in comps]; fuels=[c["fuel"] for c in comps]; allf=[p for c in comps for p in c["factors"]]
out["record_composites"]=comps
out["record_airs_distinct"]=len(set(airs))==len(airs); out["all_factors_distinct"]=len(set(allf))==len(allf)
out["fuel_in_tail_count"]=sum(c["fuel_in_tail"] for c in comps); out["fuel_max"]=max(fuels) if fuels else None
out["fuel_below_t2"]=all(f<t*t for f in fuels)
# cover at the record: open slots, composites available, slots with >= 1 composite
opens=[j for j in range(start,start+L) if openslot[j]]
out["record_cover"]={"open_slots":len(opens),"composites":len(comps),"slots_with_composite":sum(1 for j in opens if lowCFc[j] or upCFc[j]),"slots_with_two":sum(1 for j in opens if lowCFc[j] and upCFc[j])}
# tail strikes on core-free numbers vs echoes, at the record and typical (per stretch)
tailhit_low=np.zeros(S,dtype=np.int16); tailhit_up=np.zeros(S,dtype=np.int16)
for g in tail:
    inv=pow(6,-1,g)
    tailhit_low[((0-n0)*inv)%g::g]+=1; tailhit_up[(((-2)%g-n0)*inv)%g::g]+=1
Tall=slide(tailhit_low>0)+slide(tailhit_up>0)
Tcf=slide((tailhit_low>0)&lowCF)+slide((tailhit_up>0)&upCF)
out["tail_strikes_numbers"]=stats(Tall,"tail strikes on numbers"); out["tail_strikes_on_corefree"]=stats(Tcf,"tail strikes on core-free numbers")
out["identity_tail_on_corefree_eq_CFc"]=bool(np.array_equal(Tcf,CFc))
print(json.dumps(out,indent=1,default=str))
