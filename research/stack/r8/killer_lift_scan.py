"""Round 99 / loop entry 105: attacking killer residue vectors - do small lifts ever kill?

A killer residue vector at a stretch is realised by exactly one integer, p.  The residue vector of
any integer x against the gears up to P is the point of the residue space whose lift is x, and
the pattern above x^2 is fixed by x alone (the shifts are (x mod h)^2, the location law).  So the
realisability question - can a prime's own vector be a killer - has a direct form: scan the
integers x and ask whether the run of L columns above x^2 is covered by the gears up to P.  No free
residues at all: every vector tested is the vector of an actual small number.

For each gear set (gears up to P) the run length L is the twin-gear stretch length for P, the
adversary's easiest case.  Reports every x below the bound whose square starts a covered run.
"""
import sys
import numpy as np

def primes_to(n):
    s=np.ones(n+1,dtype=bool); s[:2]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=False
    return [int(v) for v in np.nonzero(s)[0]]

def main(argv):
    XMAX=int(argv[0]) if argv else 2000000
    ps=primes_to(300)
    print("squares x^2 that start a run of L columns fully struck by the gears up to P, x up to %d"%XMAX)
    print("    P    L (columns)   hits x   (first few, with x prime?)")
    for P in (17,29,41,59,71):  # L must stay below 63 bits for the int64 tables
        gears=[h for h in ps if 5<=h<=P]
        L=((P+2)**2-P*P)//6
        hits=[]
        # for each x, column m0 = (x^2 + 5)//6 ; columns m0..m0+L-1 ; struck iff some h | 6m-1 or 6m+1
        # vectorise over x: compute for each gear the offsets t in [0,L) it strikes as a function of x mod h
        # precompute per gear a table: shift c = ((r^2+5)*inv6) % h for r = x mod h -> bitmask of struck t
        tables={}
        for h in gears:
            inv6=pow(6,-1,h)
            tab=np.zeros(h,dtype=np.int64)
            for r in range(h):
                c=((r*r+5)*inv6)%h; m=0
                for tooth in (inv6%h,(-inv6)%h):
                    t=(tooth-c)%h
                    while t<L: m|=1<<t; t+=h
                tab[r]=m
            tables[h]=tab
        full=(1<<L)-1
        xs=np.arange(P+1,XMAX+1,dtype=np.int64)
        cov=np.zeros(len(xs),dtype=np.int64)
        for h in gears:
            cov|=tables[h][xs%h]
        idx=np.nonzero(cov==full)[0]
        hits=[int(xs[i]) for i in idx]
        pr=set(primes_to(min(XMAX,200000)))
        desc=", ".join("%d%s"%(x,"(prime)" if x in pr else "") for x in hits[:8])
        print("  %3d   %5d          %6d   %s%s"%(P,L,len(hits),desc," ..." if len(hits)>8 else ""))

if __name__=="__main__":
    main(sys.argv[1:])
