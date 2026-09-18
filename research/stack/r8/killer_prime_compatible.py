"""Round 95 / loop entry 101: are any killer vectors prime-compatible?

A killer vector at the stretch (p^2, q^2] is a residue vector (r_h) for the gears up to p whose
square-residue shifts cover the stretch except the square column.  A PRIME p carries r_h = p mod h,
which is never 0 for h < p, and is 0 for h = p.  This recounts the killers under that constraint,
and reports the recipe of the killers that remain: which residues the small gears use.
"""
import sys
from collections import Counter

def primes_to(n):
    s=bytearray([1])*(n+1); s[0:2]=b"\x00\x00"
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=bytearray(len(s[i*i::i]))
    return [i for i in range(2,n+1) if s[i]]

def main(argv):
    targets=[(17,19),(29,31)] if not argv else [tuple(map(int,a.split(','))) for a in argv]
    for p,q in targets:
        gears=[h for h in primes_to(p) if 5<=h<=p]
        L=(q*q-p*p)//6
        full=((1<<L)-1)&~(1<<(L-1))
        maskof={}
        for h in gears:
            inv6=pow(6,-1,h); per={}
            for r in range(h):
                c=((r*r+5)*inv6)%h; m=0
                for tooth in (inv6%h,(-inv6)%h):
                    t=(tooth-c)%h
                    while t<L: m|=1<<t; t+=h
                per[r]=m
            maskof[h]=per
        killers=[]
        def dfs(i,cov,vec):
            if i==len(gears):
                if cov&full==full: killers.append(tuple(vec))
                return
            h=gears[i]; unc=full&~cov; need=bin(unc).count("1")
            cap=sum(max(bin(m&unc).count("1") for m in maskof[hh].values()) for hh in gears[i:])
            if cap<need: return
            rs=[0] if h==p else range(1,h)   # prime-compatible: p mod h != 0 below p, = 0 at p
            for r in rs:
                dfs(i+1,cov|maskof[h][r],vec+[r])
        dfs(0,0,[])
        print("stretch %d..%d, %d gears, %d columns: prime-compatible killers = %d"%(p,q,len(gears),L,len(killers)))
        if killers:
            for idx,h in enumerate(gears):
                c=Counter(k[idx] for k in killers)
                print("   gear %2d residues used by killers: %s"%(h,dict(sorted(c.items()))))
            real=tuple(p%h for h in gears)
            print("   p's own vector: %s   (a killer? %s)"%(real, real in set(killers)))
            # how close is p's vector to any killer: number of gears whose residue would have to change
            best=min(sum(1 for a,b in zip(real,k) if a!=b) for k in killers)
            print("   fewest gears whose residue p would have to change to become a killer: %d of %d"%(best,len(gears)))
        print()

if __name__=="__main__":
    main(sys.argv[1:])
