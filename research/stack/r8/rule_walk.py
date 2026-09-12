"""Rule walks (owner, 2026-09-13): a deterministic walk whose next flip is chosen by a stepwise
rule from the machine's structure alone (gears, squares, residues), never by checking the
landing for openness; the walk decides where it stops by rule; the landing is certified
afterwards (both members prime, i.e. open to every gear) to see whether the rule worked.
Success for machine q = the landing lies in the window (q, q^2] and is a twin.

Flip about axis a (a multiple of 3, so columns go to columns): n -> 2a - n - 2. Two flips
about a and b compose to the slide n -> n + 2(b - a). A rule is a function of the machine
(q, its gears) and the walk so far that returns the next axis, or None to stop.

Rule sets tried here (all start at the pair (5, 7), n = 5):
  ladder:i     flip onto the offset-i column after the square of the first gear g above sqrt q
               (axis (n + g^2 - 2 + 6i + 2)/2); one flip, one landing.
  top:i        the same with the largest gear g whose candidate g^2 + 6i lies below q^2.
  sqaxes       flip about the square axes g^2 - 1 for g = 5, 7, 11, ... in order (each carries
               the gears of g - 1 and g + 1); stop at the first landing inside the window.
  hanoi        flip about axes 3 g_j k in Tower-of-Hanoi gear order (5, 7, 5, 11, 5, 7, 5, 13,
               ...), k the least value that moves the walk upward; stop at the first landing
               inside the window.
  blindhop:c   from the current column, flip onto the blind-class column at offset c (mod 35)
               after the next square above (carries whatever divides the axis); repeat until
               inside the window; stop there.
Reported per rule: machines succeeded / tried (primes 11 .. qmax), mean flips, and for the
failures the gear that struck the landing (smallest factor of a composite member), as a table
of counts, so the next rule can be built against it.
Usage: uv run python rule_walk.py qmax
"""
import sys, math
from collections import Counter
from sympy import primerange, isprime, factorint


def flip(n, a): return 2 * a - n - 2


def certify(n, q):
    """returns None if the landing is a twin in the window, else the reason"""
    if not (q < n and n + 2 <= q * q): return 'outside the window'
    for x in (n, n + 2):
        if not isprime(x): return f"struck by {min(factorint(x))}"
    return None


def walk(rule, q, gears, maxsteps=200):
    n = 5; path = [5]
    for _ in range(maxsteps):
        a = rule(n, q, gears, path)
        if a is None: break
        n = flip(n, a); path.append(n)
    return n, path


def ladder(i):
    def rule(n, q, gears, path):
        if len(path) > 1: return None
        g = next(g for g in gears if g * g > q)
        t = g * g - 2 + 6 * i
        return (n + t + 2) // 2
    return rule


def top(i):
    def rule(n, q, gears, path):
        if len(path) > 1: return None
        g = max(g for g in gears if g * g + 6 * i <= q * q)
        t = g * g - 2 + 6 * i
        return (n + t + 2) // 2
    return rule


def sqaxes(n, q, gears, path):
    if q < n <= q * q: return None
    k = len(path) - 1
    if k >= len(gears): return None
    g = gears[k]
    return g * g - 1


def hanoi(n, q, gears, path):
    if q < n <= q * q: return None
    k = len(path)  # step number 1, 2, 3, ...
    j = 0
    while (k >> j) & 1 == 0: j += 1   # Hanoi: the gear index is the number of trailing zeros of the step
    if j >= len(gears): return None
    g = gears[j]
    # least axis a = 3 g m with flip(n, a) > n, i.e. 2a > 2n + 2
    m = (n + 2) // (3 * g) + 1
    return 3 * g * m


def blindhop(c):
    def rule(n, q, gears, path):
        if q < n <= q * q: return None
        if len(path) > 40: return None
        g = next((g for g in gears if g * g > n), None)
        if g is None: return None
        t = g * g - 2 + 6 * c
        return (n + t + 2) // 2
    return rule


def teeth(h, i):
    return set(r for r in range(h) if (r * r) % h in ((-(6 * i - 2)) % h, (-6 * i) % h))


def avoid(i, B):
    """ladder at offset i, but the gear used is the first g above sqrt q whose residues mod the gears up to B
    avoid their teeth (a rule on classes mod a fixed finite set of gears, not a check of the landing)"""
    T = {h: teeth(h, i) for h in primerange(5, B + 1)}
    def rule(n, q, gears, path):
        if len(path) > 1: return None
        for g in gears:
            if g * g <= q or g * g + 6 * i > q * q: continue
            if all(h == g or (g % h) not in T[h] for h in T): 
                return (n + g * g - 2 + 6 * i + 2) // 2
        return None
    return rule


def main():
    qmax = int(sys.argv[1])
    rules = [('ladder:2', ladder(2)), ('ladder:10', ladder(10)), ('ladder:17', ladder(17)), ('top:10', top(10)),
             ('sqaxes', sqaxes), ('hanoi', hanoi), ('blindhop:5', blindhop(5)), ('blindhop:10', blindhop(10)),
             ('avoid:10,B=13', avoid(10, 13)), ('avoid:10,B=31', avoid(10, 31)), ('avoid:10,B=101', avoid(10, 101))]
    qs = [q for q in primerange(11, qmax + 1)]
    print(f"rule | machines succeeded / {len(qs)} | mean flips | failures by reason (top 6) | example failure")
    for name, rule in rules:
        ok = 0; steps = 0; fails = Counter(); ex = None
        for q in qs:
            gears = list(primerange(5, q + 1))
            n, path = walk(rule, q, gears)
            r = certify(n, q); steps += len(path) - 1
            if r is None: ok += 1
            else:
                fails[r] += 1
                if ex is None: ex = (q, path[-3:], r)
        print(f"{name} | {ok} | {steps/len(qs):.1f} | {fails.most_common(6)} | {ex}")


if __name__ == "__main__":
    main()
