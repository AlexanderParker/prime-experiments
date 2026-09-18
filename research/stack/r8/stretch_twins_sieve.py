"""Round 94 / loop entry 100: twins in every consecutive-gear stretch, to 20000, by segmented sieve.

Same object as stretch_twins.py (entry 99) - the twins inside (p^2, q^2] for consecutive gears
p < q - pushed from 5000 to 20000 with a numpy segmented sieve instead of primality tests.
"""
import sys
import numpy as np

def primes_to(n):
    s=np.ones(n+1,dtype=bool); s[:2]=False
    for i in range(2,int(n**0.5)+1):
        if s[i]: s[i*i::i]=False
    return np.nonzero(s)[0]

def main(argv):
    bound=int(argv[0]) if argv else 20000
    small=primes_to(int((bound*1.05)**1)+100)   # primes up to a bit past bound, enough for sqrt of q^2
    gears=[int(x) for x in small if x>=5 and x<=bound]
    nxt=[int(x) for x in small if x>bound]
    ps=gears+nxt[:1]
    rows=[]
    for p,q in zip(ps,ps[1:]):
        lo=p*p+1; hi=q*q
        n=hi-lo+1
        seg=np.ones(n,dtype=bool)
        for r in small:
            r=int(r)
            if r*r>hi: break
            start=((lo+r-1)//r)*r
            if start==r: start=2*r
            seg[start-lo::r]=False
        # twins: members 6m-1, 6m+1 both prime, lower member > p^2
        m0=p*p//6+1
        m1=(q*q-1)//6
        tw=0
        first=None
        for m in range(m0,m1+1):
            a=6*m-1
            if a<=p*p: continue
            if seg[a-lo] and seg[a+2-lo]:
                tw+=1
                if first is None: first=a
        rows.append((tw,p,q,m1-m0+1,first))
    rows.sort()
    print("twins in each stretch (p^2, q^2] for consecutive gears below %d (%d stretches)"%(bound,len(rows)))
    print("   fewest twins:")
    for tw,p,q,cols,first in rows[:10]:
        print("     p=%6d q=%6d  columns %7d  twins %5d  first twin at %s"%(p,q,cols,tw,first))
    print("   stretches with no twin: %d"%sum(1 for w in rows if w[0]==0))
    print("   smallest twins per column: %.4f at p = %d"%min((w[0]/w[3],w[1]) for w in rows))
    big=[w for w in rows if w[1]>10000]
    print("   among p > 10000: fewest twins %d (p = %d), smallest per column %.4f"%(min(big)[0],min(big)[1],min(w[0]/w[3] for w in big)))

if __name__=="__main__":
    main(sys.argv[1:])
