"""Chain census in the correct k-frame window: how many consecutive openings
of a machine can be deleted by one lap of the next gear?

THE FRAME TRAP, recorded because it bit here too: in the adjacent frame the
new gear deletes openings in two ADJACENT residues {phi, phi+1}. In k-space
the teeth sit at +-u_q with separation s = 2*u_q = 3^{-1} mod q - NOT
adjacent. A first version of this census used {phi, phi+1} and its k=2 count
came out exactly prod(q-4) - the domino count - for every q, which is what
exposed the error. The corpus's caution about confusing the two frames
(handover section 0.5) is earned.

With the correct window {phi, phi+s}:

* qualifying interior distances are = 0, +-s (mod q), so the minimum positive
  qualifying distance is min(s, q-s) = (q +- 1)/3 - the k-frame deletion-
  spacing law (adjacent-frame version, proved in gear-recursion.md section 4,
  is >= q-1; divide by 3);
* chains die entirely once (q +- 1)/3 > F_k(M) - the saturation threshold;
* the chain prediction (max merged stride from the old gap word alone)
  reproduces the true F_k(M+q) exactly: verified for F_k(17) = 18,
  F_k(19) = 25, F_k(23) = 34;
* the census reproduces the corpus's count of 62 double-interior chains at
  gears <= 19, q = 23, from an independent implementation;
* anatomy of ALL 62 k=3 runs: interior distances exactly (s, q-s) or its
  mirror - residues a -> a+s -> a, span exactly q. Maximal chains are the
  minimal alternation. A k=4 run would need the exact consecutive gap word
  (s, q-s, s) (span q+s) or a gap = 0 mod q adjacent - enumerable conditions
  on the gap word, none present at this size.

Provable span law (pigeonhole + the distance classes): a run of k openings in
one window has span >= floor((k-1)/2)*q, since same-residue pairs are >= q
apart and alternating pairs of distances sum to >= q. Combined with
"consecutive openings are <= F_k(M) apart" this bounds nothing once
F_k(M) >= q/2 - which is the regime that matters, consistent with the
corpus's section 5.5 verdict that gap structure alone cannot bound k. The
missing ingredient is the arithmetic of which specific gap words occur.
"""
from math import prod
from collections import Counter

def machine(y):
    gs = [q for q in range(5, y + 1) if all(q % d for d in range(2, int(q**0.5) + 1))]
    us = {q: pow(6, -1, q) for q in gs}
    P = prod(gs)
    opens = [k for k in range(P)
             if all(k % q not in (us[q], (-us[q]) % q) for q in gs)]
    return gs, P, opens

def fits(vals, q, s):
    st = set(vals)
    if len(st) == 1:
        return True
    if len(st) != 2:
        return False
    a, b = st
    return (b - a) % q in (s, (q - s) % q)

def census(y, qs, show_anatomy=None):
    gs, P, opens = machine(y)
    gaps = [b - a for a, b in zip(opens, opens[1:])]
    Fk = max(gaps)
    F2 = max(a + b for a, b in zip(gaps, gaps[1:]))
    print(f"M = gears<={y}, period {P}, openings {len(opens)}, F_k = {Fk}")
    for q in qs:
        s = (2 * pow(6, -1, q)) % q
        counts = Counter()
        pred = F2
        anat = Counter()
        n = len(opens)
        i = 0
        while i < n - 1:
            j = i
            while j + 1 < n and fits([opens[t] % q for t in range(i, j + 2)], q, s):
                j += 1
            k = j - i + 1
            if k >= 2:
                counts[k] += 1
                if i >= 1 and j + 1 < n:
                    pred = max(pred, opens[j + 1] - opens[i - 1])
                if show_anatomy == (q, k):
                    anat[tuple(opens[t + 1] - opens[t] for t in range(i, j))] += 1
                i = j
            i += 1
        print(f"  q={q:3d} sep {s:3d} mindist {min(s, q - s):3d}: "
              f"runs {dict(sorted(counts.items()))}, prediction F_k(M+q) = {pred}")
        if anat:
            print(f"    anatomy of k={show_anatomy[1]} runs: {dict(anat)}")

if __name__ == "__main__":
    census(13, [17, 19, 23, 29, 31, 37, 41])
    census(17, [19, 23, 29, 31, 37, 41])
    census(19, [23, 29, 31, 37, 41, 43], show_anatomy=(23, 3))
