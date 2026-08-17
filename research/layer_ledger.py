"""The layer ledger: what one layer of the horizon telescope actually adds.

A layer is one prime retiring into the working set: y -> y' (next prime), the
working set gains gear y, the horizon advances y^2 -> y'^2, opening the fresh
band (y^2, y'^2).

LAYER LAW (proved by the root argument, verified in the ledger below): the
newly activated gear's entire novel workload per layer is

    1. retro-closing the old horizon square y^2      (owed iff y^2 - 2 prime)
    2. closing the slots of y*c for primes c in (y, y'^2/y)  - a list of one
       to three explicit numbers (Bertrand: y'^2/y < 4y) - each owed iff its
       partner member is prime.

Every other slot of the fresh band is closed by the old gears. Verified for
layers 13->17 through 43->47: seven of nine layers owe nothing at all in-band;
the exceptions are 221 = 13*17 (partner 223 prime) and 437 = 19*23 (partner
439 prime). A layer's new content is a short explicit list of semiprime slots,
enumerable in advance; the complexity of the tower lives only in the number of
layers, never inside one.
"""

def isprime(n):
    if n < 2: return False
    d = 2
    while d*d <= n:
        if n % d == 0: return False
        d += 1
    return True

def primes(a, b):
    return [p for p in range(a, b+1) if isprime(p)]

def nextprime(n):
    m = n + 1
    while not isprime(m): m += 1
    return m

def layer_ledger(y):
    yn = nextprime(y)
    old = primes(5, y - 1)
    us = {q: pow(6, -1, q) for q in old}
    lo, hi = (y*y)//6 + 1, (yn*yn - 2)//6
    owned, twins = [], 0
    for k in range(lo, hi + 1):
        a, b = 6*k - 1, 6*k + 1
        if b >= yn*yn: continue
        truth = isprime(a) and isprime(b)
        twins += truth
        if all(k % q not in (us[q], (-us[q]) % q) for q in old) and not truth:
            owned.append((k, a, b))
    return yn, hi - lo + 1, twins, owned

if __name__ == "__main__":
    for y in [13, 17, 19, 23, 29, 31, 37, 41, 43]:
        yn, band, twins, owned = layer_ledger(y)
        o = ', '.join(f"{k}=({a},{b})" for k, a, b in owned) or 'NONE'
        print(f"layer {y}->{yn}: band {band} slots, twins {twins}, new gear owes: {o}")
