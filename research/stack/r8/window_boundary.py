"""The boundary of the square-root rule (manager check, 2026-09-22).

The rule used throughout: a column n unstruck by every gear of machine q, with 6n + 1 <= q'^2
(q' the next prime after q), is a twin prime pair. The claim under test is that this is FALSE at
exactly one column of the closed window, namely n0 = (q'^2 - 1)/6 when that is an integer, because
there 6n + 1 = q'^2 is a square with no prime factor at or below q - so the column is an opening of
machine q yet 6n + 1 is composite.

Checked here directly: for each q, list every opening of machine 5..q in the closed window
(q, q'^2] that is NOT a twin pair, and compare with the predicted single exception.
"""
from sympy import isprime, primerange, nextprime

print("q    q'   openings in (q, q'^2]   non-twin openings (column, 6n-1, 6n+1)")
for q in list(primerange(7, 120)):
    qn = nextprime(q)
    gears = [g for g in primerange(5, q + 1)]
    lo = q // 6 + 1
    hi = (qn * qn - 1) // 6
    opens = []
    for n in range(lo, hi + 1):
        a, b = 6 * n - 1, 6 * n + 1
        if all(a % g and b % g for g in gears):
            opens.append(n)
    bad = [(n, 6 * n - 1, 6 * n + 1) for n in opens
           if not (isprime(6 * n - 1) and isprime(6 * n + 1))]
    pred = (qn * qn - 1) // 6 if (qn * qn - 1) % 6 == 0 else None
    # The exact claim: the ONLY column of the closed window that can be an opening yet not a twin
    # is n0 = (q'^2 - 1)/6, and it is one exactly when q'^2 - 2 has no prime factor <= q.
    mark = "OK" if all(x[0] == pred for x in bad) else "MISMATCH"
    print(f"{q:3d} {qn:4d} {len(opens):20d}   {bad}  ({mark}, predicted column {pred})")
