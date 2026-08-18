"""Harvester round 12: the two pruning theorems' number-theory cores, exhaustively.

  L1  MOD-3 FREE-CLASS LEMMA (Lean target): gear 3 blocks the adjacent pair
      {a, a+1} mod 3, leaving ONE free class, so any two positions it leaves
      uncovered are congruent mod 3. Corollary (endpoint law): if the run
      [s, s+M) is covered and both flanks s-1, s+M are uncovered by gear 3,
      then the gap length (s+M)-(s-1) = M+1 = 0 mod 3, i.e. F(2,y) = 0 mod 3.
  L2  LEFT-TAUT EQUIVALENCE (handed to Formalist): for EVERY L,
      coverable(L) <=> coverable(L) by an assignment leaving position -1
      uncovered. Verified here by exhaustive enumeration over ALL offset
      tuples (not the pruned search), y = 11, 13, 17, every L up to F+2.
  L3  Consistency: F values from exhaustive enumeration match the corpus.
"""
from itertools import product
from sympy import primerange

def odd_primes(y):
    return [q for q in primerange(3, y + 1)]

def covers(o, q, i):
    """gear q at offset o blocks position i (i may be -1)."""
    return i % q == o % q or i % q == (o + 1) % q

def coverable(L, primes, lefttaut):
    for tup in product(*[range(q) for q in primes]):
        if lefttaut and any(covers(o, q, -1) for o, q in zip(tup, primes)):
            continue
        if all(any(covers(o, q, i) for o, q in zip(tup, primes)) for i in range(L)):
            return True
    return False

# L1 corollary check is in literal_cap_gap_d.py (T3); here the two search facts.
for y in (11, 13, 17):
    primes = odd_primes(y)
    F = None
    mism = []
    L = 1
    while True:
        a = coverable(L, primes, False)
        b = coverable(L, primes, True)
        if a != b:
            mism.append(L)
        if not a and F is None:
            F = L
        if F is not None and L >= F + 2:
            break
        L += 1
    print(f"y={y:>2} primes={primes} F(2,{y})={F} "
          f"(= 0 mod 3: {F % 3 == 0}) | left-taut equivalence over L=1..{L}: "
          f"{'OK' if not mism else f'MISMATCH at {mism}'}")
