"""Round 100 / loop entry 106: is the plug's deficit the consecutive-prime residue correlation?

plug_interleave.py found the plug p2 p3 sits on a base-open column in 12.4 percent of consecutive
triples against 19.7 percent for free residues.  Control: replace p3 by a prime of the same size
that is NOT consecutive to p2 (the fifth prime after p2), and by a random prime in (p2, 2 p2).  If
the control recovers the free fraction, the deficit is the correlation between residues of
CONSECUTIVE primes - the Lemke Oliver / Soundararajan bias - not anything in the machine's rules.
"""
import sys, random
from sympy import primerange, factorint

def base_open(n, p1):
    partner = n-2 if n%6==1 else n+2
    return all(h>p1 for h in factorint(partner))

def main(argv):
    bound=int(argv[0]) if argv else 20000
    ps=list(primerange(5,2*bound+100))
    rng=random.Random(5)
    idx={p:i for i,p in enumerate(ps)}
    cons=0; fifth=0; rand=0; total=0; expsum=0.0
    for i in range(len(ps)-6):
        p1,p2,p3=ps[i],ps[i+1],ps[i+2]
        if p1>=bound: break
        total+=1
        cons+=base_open(p2*p3,p1)
        fifth+=base_open(p2*ps[i+6],p1)
        # random prime in (p2, 2 p2)
        cands=[q for q in ps[i+1:i+400] if p2<q<2*p2]
        rand+=base_open(p2*rng.choice(cands),p1)
        exp=1.0
        for h in primerange(5,p1+1): exp*=(1-1.0/h)
        expsum+=exp
    print("triples with p1 below %d: %d"%(bound,total))
    print("  plug p2*p3 (consecutive):                 base-open %.4f"%(cons/total))
    print("  p2 * (fifth prime after p2):              base-open %.4f"%(fifth/total))
    print("  p2 * (random prime in (p2, 2p2)):         base-open %.4f"%(rand/total))
    print("  free-residue expectation, averaged:       %.4f"%(expsum/total))

if __name__=="__main__":
    main(sys.argv[1:])
