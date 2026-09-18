"""Round 101 / loop entry 107: concept 1 - the run above the square against the stretch it must span.

Let R(p) be the number of columns above p^2 struck by the gears up to p before the first open one -
by the stretch rule, the offset in columns of the first twin above p^2 (or the square column of
the next prime q, whichever comes first; if the square comes first the run continues past it).
A kill needs R(p) to reach the stretch's end (kill_needs_run), i.e. R(p) >= (q^2 - p^2) / 6.
This measures R(p) for every prime to a bound, its records, and its largest share of the stretch.
"""
import sys, math
from sympy import primerange, isprime

def main(argv):
    bound=int(argv[0]) if argv else 200000
    ps=list(primerange(5,bound+1000))
    rec=0; recs=[]; worst_share=(0.0,None); n=0
    ratio_ln=(0.0,None)
    for p,q in zip(ps,ps[1:]):
        if p>bound: break
        n+=1
        m=p*p//6+1
        R=0
        while True:
            a=6*m-1
            if a>p*p and isprime(a) and isprime(a+2):
                break
            if a>p*p: R+=1
            m+=1
        stretch=(q*q-p*p)//6
        share=R/stretch
        if share>worst_share[0]: worst_share=(share,p,R,stretch)
        r=R/(math.log(p)**2)
        if r>ratio_ln[0]: ratio_ln=(r,p,R)
        if R>rec:
            rec=R; recs.append((p,q,R,stretch))
    print("primes to %d: %d"%(bound,n))
    print("record runs R(p) above p^2 (columns to the first twin), with the stretch length needed for a kill:")
    print("        p       q      R(p)   stretch (q^2-p^2)/6   R/stretch   R/(ln p)^2")
    for p,q,R,st in recs[-12:]:
        print("  %8d %8d  %8d  %20d  %10.5f  %10.2f"%(p,q,R,st,R/st,R/math.log(p)**2))
    print("largest share of a stretch ever covered from its square: %.5f at p = %d (R = %d of %d)"%worst_share)
    print("largest R/(ln p)^2: %.2f at p = %d (R = %d)"%ratio_ln)

if __name__=="__main__":
    main(sys.argv[1:])
