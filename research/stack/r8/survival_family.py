"""Round 103 / loop entry 109: the survival lemma on the primorial family - one fixed universal pattern.

Take the family 30 t +- 1, the mirror {2,3,5} from home (t = 0 is home).  A gear h >= 7 strikes
the family at t = +-inv(30) modulo h - positions fixed by h alone, the same for every machine.  So
the gears 7..p lay ONE fixed pattern on the t-line; the machine p only selects the range
[p^2/30, q^2/30].  The survival lemma, restricted to this family, says: the struck run of that
fixed pattern starting at t0 = p^2/30 is shorter than (q^2 - p^2)/30.

This measures the struck run of the fixed pattern of the gears up to p starting exactly at the
square's position, against the runs starting at random positions in the same range with the same
gears.  If runs at the squares are systematically shorter, the square's residue structure (the
location law) is doing work; if not, the square is an ordinary position of a universal pattern.
"""
import sys, math, random
from sympy import primerange, isprime

def run_from(t0, gears, limit):
    # struck run of the fixed pattern starting at t0: t struck iff some gear h has t = +-inv(30) mod h
    t=t0; R=0
    while R<limit:
        n=30*t
        if isprime(n-1) and isprime(n+1):
            # open to every gear (both members prime) - but we want open to gears <= p only;
            # members below q^2 with no gear <= p are prime, so primality = openness here
            return R
        struck=False
        for h in gears:
            if (n-1)%h==0 or (n+1)%h==0: struck=True; break
        if not struck:
            return R  # open to the gears up to p (a twin candidate of the machine, counted as open)
        t+=1; R+=1
    return R

def main(argv):
    rng=random.Random(9)
    print("fixed pattern of the gears 7..p on the family 30t +- 1: struck run from the square against random positions")
    print("      p     q    stretch in t   run at t0 = p^2/30   mean run at 12 random t in range   max random")
    ps=list(primerange(5,21000))
    for p in (1009,2003,4001,6007,8009,10007,15013,19997):
        q=next(x for x in ps if x>p)
        gears=[h for h in ps if 7<=h<=p]
        t0=p*p//30+1
        span=(q*q-p*p)//30
        R0=run_from(t0,gears,span+5)
        rs=[run_from(rng.randrange(t0,t0+span),gears,span+5) for _ in range(12)]
        print("  %5d %5d  %12d  %19d  %31.1f  %10d"%(p,q,span,R0,sum(rs)/len(rs),max(rs)))

if __name__=="__main__":
    main(sys.argv[1:])
