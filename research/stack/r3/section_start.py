"""Evidence for the hot lead (R4.d.i.a), generated while the lanes are paused.

Relative to a prime square q^2, offset i is the slot (q^2 + 6i - 2, q^2 + 6i).  Gear g strikes
offset i iff -6i or 2 - 6i is congruent to q^2 (mod g); since q^2 is a square mod g, each gear is
BLIND at the offset classes that no square can produce.  Gears 5 and 7 are jointly blind at
i = 5, 10, 12, 17 (mod 35).

Modes:
  first  : for every prime q in [7, QMAX], the first twin above q^2 (offset, fraction of the long
           arc d = (2q+1)//3, class mod 35, blind or not); exceptions = no twin below the arc.
  census : for every prime q in [7, QMAX], ALL twins below the arc, counted per class mod 35, and
           the same count for a CONTROL start (a random n0 = 1 mod 6 of the same size, same arc).
  blind  : for every odd prime g <= 400, the blind classes by brute force against the formula
           1 + #{nonzero squares s with s + 2 also a nonzero square}.
usage: uv run python research/stack/r3/section_start.py first 200000
       uv run python research/stack/r3/section_start.py census 20000
       uv run python research/stack/r3/section_start.py blind
"""
import os
import random
import sys
from collections import Counter

from sympy import isprime, primerange

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
BLIND35 = {5, 10, 12, 17}


def first_mode(qmax):
    out = [f"# first twin above q^2, primes q in [7, {qmax}]; arc d = (2q+1)//3 offsets"]
    exc = []
    cls = Counter()
    fracs = []
    blind_first = 0
    n = 0
    worst = (0, 0)
    for q in primerange(7, qmax + 1):
        n += 1
        d = (2 * q + 1) // 3
        found = None
        base = q * q
        for i in range(1, d):
            a = base + 6 * i - 2
            if isprime(a) and isprime(a + 2):
                found = i
                break
        if found is None:
            exc.append(q)
            continue
        f = found / d
        fracs.append(f)
        cls[found % 35] += 1
        if found % 35 in BLIND35:
            blind_first += 1
        if f > worst[0]:
            worst = (f, q)
        if n % 2000 == 0:
            print(f"  ... q={q} n={n} exc={len(exc)}", flush=True)
    fracs.sort()
    m = len(fracs)
    out.append(f"primes tested: {n}; exceptions (no twin below the arc): {len(exc)} {exc[:30]}")
    out.append(f"first-twin offset as a fraction of the arc: median {fracs[m//2]:.4f}, 90% {fracs[int(.9*m)]:.4f}, 99% {fracs[int(.99*m)]:.4f}, max {worst[0]:.4f} at q={worst[1]}")
    out.append(f"first twin in a 5/7-blind class: {blind_first} of {m} = {blind_first/m:.4f} (uniform over the 15 corridor classes would be 4/15 = 0.2667)")
    out.append("first-twin class mod 35 census: " + ", ".join(f"{k}:{v}" for k, v in sorted(cls.items())))
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(RES, f"first_{qmax}.txt"), "w", encoding="utf-8").write(txt + "\n")


def census_mode(qmax):
    random.seed(1)
    out = [f"# all twins below the arc above q^2, and above a control start, primes q in [7, {qmax}]"]
    sq = Counter()
    ctl = Counter()
    per_q = []
    n = 0
    for q in primerange(7, qmax + 1):
        n += 1
        d = (2 * q + 1) // 3
        base = q * q
        # control: a random start n0 = 1 (mod 6) near q^2 (not a square), same arc, same offset rule
        n0 = base + 6 * random.randint(d, 50 * d)
        c_sq = 0
        c_ct = 0
        for i in range(1, d):
            a = base + 6 * i - 2
            if isprime(a) and isprime(a + 2):
                sq[i % 35] += 1
                c_sq += 1
            b = n0 + 6 * i - 2
            if isprime(b) and isprime(b + 2):
                ctl[i % 35] += 1
                c_ct += 1
        per_q.append((q, c_sq, c_ct))
        if n % 500 == 0:
            print(f"  ... q={q} n={n}", flush=True)
    tot_sq = sum(sq.values())
    tot_ct = sum(ctl.values())
    out.append(f"primes: {n}; twins below the arc: above squares {tot_sq}, above control starts {tot_ct}; ratio {tot_sq/max(tot_ct,1):.4f}")
    bl_sq = sum(sq[c] for c in BLIND35)
    bl_ct = sum(ctl[c] for c in BLIND35)
    out.append(f"share in the 5/7-blind classes: squares {bl_sq/tot_sq:.4f}, control {bl_ct/max(tot_ct,1):.4f} (uniform over the corridor's 15 classes: 0.2667)")
    out.append("per-class mod 35, squares:  " + ", ".join(f"{c}:{sq[c]}" for c in range(35) if sq[c]))
    out.append("per-class mod 35, control:  " + ", ".join(f"{c}:{ctl[c]}" for c in range(35) if ctl[c]))
    zero_sq = sum(1 for q, a, b in per_q if a == 0)
    zero_ct = sum(1 for q, a, b in per_q if b == 0)
    out.append(f"arcs with no twin: above squares {zero_sq}, above control starts {zero_ct} (of {n})")
    per_q.sort(key=lambda t: t[1])
    out.append("fewest twins above a square (q, count, control count): " + ", ".join(f"({q},{a},{b})" for q, a, b in per_q[:12]))
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(RES, f"census_{qmax}.txt"), "w", encoding="utf-8").write(txt + "\n")


def blind_mode():
    out = ["# blind classes per gear relative to squares: brute force vs formula"]
    bad = 0
    for g in primerange(5, 401):
        squares = {(x * x) % g for x in range(1, g)}
        inv6 = pow(6, -1, g)
        struck = set()
        for s in squares:
            struck.add((-s * inv6) % g)
            struck.add(((2 - s) * inv6) % g)
        blind = sorted(set(range(g)) - struck)
        formula = 1 + sum(1 for s in squares if (s + 2) % g in squares and (s + 2) % g != 0)
        # note: pairs (s, s+2) both nonzero squares; the formula counts overlaps of the two struck sets
        ok = (len(blind) == formula)
        if not ok:
            bad += 1
        if g <= 60 or not ok:
            out.append(f"g={g}: blind classes {blind} (count {len(blind)}, formula {formula}{'' if ok else '  MISMATCH'})")
    out.append(f"mismatches of the formula: {bad} of {len(list(primerange(5, 401)))}")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(RES, "blind.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "first":
        first_mode(int(sys.argv[2]))
    elif mode == "census":
        census_mode(int(sys.argv[2]))
    elif mode == "blind":
        blind_mode()
