"""Round 96 / loop entry 102: who can kill where, after a square - the location law, read off.

A gear h strikes the member p^2 + a only if -a is a square modulo h (KillPositions.lean).  So each
position after a square admits killers only from the gears in the residue classes where -a is a
square.  This prints, for a few machines, the first columns above p^2: the position, the class
condition on an eligible killer, the gear that actually strikes there, and whether that gear sits
in the eligible class (it must).
"""
import sys
from sympy import primerange, factorint, isprime
from sympy.ntheory import legendre_symbol

def eligible_desc(a):
    # describe the class condition for -a to be a square mod h, for small a, by testing h up to 200
    return None

def main():
    for p in (29, 101, 1009):
        q = next(x for x in primerange(p+1, p+200))
        print("machine %d -> %d: the first columns above %d^2 = %d" % (p, q, p, p*p))
        print("   column   member          offset a   eligible killers (-a a square mod h)   actual killer   its class ok?")
        cols = 0
        m = p*p//6 + 1
        while cols < 8:
            for side, mem in (("L", 6*m-1), ("U", 6*m+1)):
                if mem <= p*p: continue
                a = mem - p*p
                f = factorint(mem)
                killers = [g for g in f if g <= p]
                # eligible test for each gear <= p: legendre(-a, h) == 1
                elig = [h for h in primerange(5, p+1) if legendre_symbol((-a) % h, h) == 1]
                ok = all(legendre_symbol((-a) % g, g) == 1 for g in killers)
                kdesc = ",".join(str(g) for g in killers) if killers else ("prime" if isprime(mem) else "large factors only")
                print("   %s%-4d   %-14d  %6d     %3d of %d gears eligible                 %-14s  %s"
                      % (side, m, mem, a, len(elig), len(list(primerange(5,p+1))), kdesc, "yes" if killers and ok else ("-" if not killers else "NO")))
            cols += 1
            m += 1
        print("   the square's neighbour p^2 - 2 = %d: factors %s; eligible gears are those = 1 or 7 mod 8"
              % (p*p-2, {g:e for g,e in factorint(p*p-2).items()}))
        print()

if __name__ == "__main__":
    sys.exit(main())
