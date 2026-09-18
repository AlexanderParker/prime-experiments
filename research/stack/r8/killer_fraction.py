"""Round 94 / loop entry 100: how many residue vectors kill a stretch, against the one the machine visits.

Each stretch (p^2, q^2] has exactly one realisable residue vector - the residues of p itself, since
p is the only prime whose gear set is the gears up to p with next prime q.  The residue model has
many vectors; this counts, for small p, how many of them cover the stretch (killers) against the
total, and shows the vector p actually carries.
"""
import sys
from itertools import product

def primes_to(n):
    s=bytearray([1])*(n+1); s[0:2]=b"\x00\x00"
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=bytearray(len(s[i*i::i]))
    return [i for i in range(2,n+1) if s[i]]

def main(argv):
    pmax=int(argv[0]) if argv else 23
    ps=[x for x in primes_to(pmax+100) if x>=5]
    print("   p    q  columns   residue vectors   killers   fraction     p's own vector kills?")
    for p,q in zip(ps,ps[1:]):
        if p>pmax: break
        gears=[h for h in primes_to(p) if 5<=h<=p]
        L=(q*q-p*p)//6
        full=(1<<L)-1
        full&=~(1<<(L-1))
        maskof={}
        for h in gears:
            inv6=pow(6,-1,h)
            per={}
            for r in range(0,h):
                c=((r*r+5)*inv6)%h
                m=0
                for tooth in (inv6%h,(-inv6)%h):
                    t=(tooth-c)%h
                    while t<L: m|=1<<t; t+=h
                per[r]=m
            maskof[h]=per
        # count killers by DFS over gears with residue choices (r != 0 except for h == p where r = 0 is the truth; allow all r)
        total=1
        for h in gears: total*=h
        gl=gears
        count=[0]
        def dfs(i,cov):
            if i==len(gl):
                if cov&full==full: count[0]+=1
                return
            h=gl[i]
            # prune: remaining gears' max coverage must reach the uncovered
            unc=full&~cov
            need=bin(unc).count("1")
            cap=0
            for hh in gl[i:]:
                cap+=max(bin(m&unc).count("1") for m in maskof[hh].values())
            if cap<need: return
            for r in range(h):
                dfs(i+1,cov|maskof[h][r])
        dfs(0,0)
        # the real vector
        cov=0
        for h in gears: cov|=maskof[h][p%h]
        real_kills = (cov&full)==full
        print("  %3d  %3d  %7d  %16d  %8d  %9.2e     %s"%(p,q,L,total,count[0],count[0]/total,real_kills))

if __name__=="__main__":
    main(sys.argv[1:])
