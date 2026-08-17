"""The event-horizon exclusion: the top gear is droppable strictly inside its window.

Theorem (two lines): any composite member M with y < M < y^2 has a prime factor
<= sqrt(M) < y, so the top gear y is never the root cause of an interior kill.
Its entire unique contribution is the two boundary objects: its self-pair at the
bottom edge and its square at the top edge (the event horizon). Hence gears
STRICTLY BELOW y decide the open interior (y, y^2) exactly.

Verified for all primes y in [13, 79]: interior exact everywhere; the horizon
slot (y^2 - 2, y^2) false-positives precisely when y^2 - 2 is prime (167, 359,
839, 1367, 1847, 2207, 3719, 5039), and is closed by lower gears otherwise.

Structural limit: the trick works exactly once per window - the second gear's
square lies strictly inside, so further drops happen only by transient
partner-compositeness (see research/minimal_subset.py).

Side fact demonstrated en route: the primorial-scale unwind does NOT always
produce twins - nudge home 595 of the {5,7,11,13} machine is (3569, 3571) with
3569 = 43*83. Openness to a set is not twinhood beyond the set's horizon.
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

def interior_exact(y):
    sub = primes(5, y - 1)
    us = {q: pow(6, -1, q) for q in sub}
    for k in range(1, (y*y - 1)//6 + 1):
        a, b = 6*k - 1, 6*k + 1
        if not (a > y and b < y*y):
            continue
        if (all(k % q not in (us[q], (-us[q]) % q) for q in sub)
                != (isprime(a) and isprime(b))):
            return False
    return True

if __name__ == "__main__":
    for y in primes(13, 80):
        print(f"y={y:3d}: gears<y decide interior exactly: {interior_exact(y)}")
