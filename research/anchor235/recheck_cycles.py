import numpy as np
OPEN30 = {1, 11, 13, 17, 19, 29}
def mset(q): return [m for m in range(1, 30) if (q * m) % 30 in OPEN30]
def cyc(n): return (n - 11) // 30          # cycle j = numbers 30j+11 .. 30j+31 (slots 5j+2, 5j+3, 5j+5)
sieve = np.ones(5001, bool); sieve[:2] = False
for i in range(2, 71):
    if sieve[i]: sieve[i*i::i] = False
primes = [int(p) for p in np.flatnonzero(sieve) if p >= 7]
bad = []
print("gear: hit cycles within one run of q cycles (cycles qt .. qt+q-1), untouched count, first/last cycle hit?")
for q in primes:
    hits = sorted({cyc(q * m) % q for m in mset(q)})   # cycle offset within the run, all runs t identical
    unt = q - len(hits)
    if q <= 43 or q in (97, 4999):
        print(f"  q={q:>4} (class {q%30:>2}): m={mset(q)} -> cycles {hits}; untouched {unt}; first hit {0 in hits}, last hit {q-1 in hits}")
    if unt != q - 6 and q != 7: bad.append(q)
print(f"q-6 law (six distinct cycles) fails for: {bad or 'none up to 5000 (q=7: five cycles, untouched 2)'}")
ends = [(q, 0 in {cyc(q*m) % q for m in mset(q)}, (q-1) in {cyc(q*m) % q for m in mset(q)}) for q in primes]
print("gears whose first cycle of the run is hit:", [q for q, f, l in ends if f])
print("gears whose last cycle of the run is hit:", [q for q, f, l in ends if l])
print("clean both ends from q =", next(q for q in primes if all(not f and not l for qq, f, l in ends if qq >= q)), "on (no exception above)")
