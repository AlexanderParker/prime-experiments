"""Round 100 / loop entry 106: concept 3 - the multiplicative interleave, made specific by the plug law.

By the plug law the plug of the stretch above p2^2 is p2 p3, the product of consecutive primes.
Its position relative to the base p1 is fixed by the two gaps: p2 p3 - p1^2 = p1 (2 g1 + g2) + g1 (g1 + g2).
Its column is open to the base exactly when its partner member p2 p3 -+ 2 has no prime factor up
to p1 - a quadratic in p1's residues, (r + g1)(r + g1 + g2) -+ 2 modulo each gear h, with no
algebraic factorisation.  This checks, on real consecutive triples, whether the plug's column is
base-open (the plug does work) against what free residues would give.
"""
import sys
from sympy import primerange, factorint

def main(argv):
    bound=int(argv[0]) if argv else 20000
    ps=list(primerange(5,bound))
    work=0; total=0; ratio_sum=0.0
    examples=[]
    for p1,p2,p3 in zip(ps,ps[1:],ps[2:]):
        n=p2*p3
        partner = n-2 if n%6==1 else n+2
        small=[h for h in factorint(partner) if h<=p1]
        base_open = (len(small)==0)
        total+=1
        if base_open: work+=1
        # free-residue expectation for this triple: product over gears h<=p1 of (1 - 1/h) (partner avoids one class per gear)
        exp=1.0
        for h in primerange(5,p1+1): exp*=(1-1.0/h)
        ratio_sum+=exp
        if len(examples)<6 and base_open: examples.append((p1,p2,p3,n,partner))
    print("consecutive triples p1<p2<p3 below %d: %d"%(bound,total))
    print("  plug p2*p3 whose column is open to the base (the plug does the kill): %d  (%.4f)"%(work,work/total))
    print("  free-residue expectation of that fraction, averaged over the triples:   %.4f"%(ratio_sum/total))
    print("  examples where the plug does the kill (p1,p2,p3, plug, partner):")
    for e in examples: print("    ",e)

if __name__=="__main__":
    main(sys.argv[1:])
