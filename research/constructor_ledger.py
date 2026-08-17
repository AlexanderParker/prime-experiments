"""Constructor workstream, round 1: the supply-demand ledger of Condition X.

Condition X (the contradiction target, stated precisely): some window (y, y^2)
contains zero twins - every slot k with both members 6k-1, 6k+1 strictly inside
(y, y^2) has at least one composite member, i.e. receives a root kill from a
gear q with q^2 <= member.

Ledger objects, all exact:
  N  = interior slot count
  P  = prime members among the 2N interior members
  C  = composite members = 2N - P = total root kills (lpf attribution partitions)
  n0/n1/n2 = slots with 0/1/2 composite members (twin / fragile / double)
Identities: n0+n1+n2 = N, n1+2n2 = C, primes P = n1 + 2*n0.

Under X: n0 = 0 forces n1 = P and n2 = N - P = C - N exactly (zero slack).

Checks below, per y in {13, 23, 47}:
  1. census + identities
  2. per-gear exact root-kill supply R(q) vs the semiprime formula
     (square + coprimes ~ pi(y^2/q) - pi(q) + 1); horizon: top gear R = 0
  3. pseudo-twin ledger: PT(q) = q's root kills beside a prime;
     sum PT = n1 = P - 2*n0, so X's demand P is missed by exactly 2*(twin count)
  4. prefix margin m(k) = N(<=k) - P(<=k) = n2(<=k) - n0(<=k): X requires
     m(k) >= 0 for every prefix (pigeonhole); min and argmin reported
  5. max run excess E = max over runs I of P(I) - N(I) (Kadane on p_k - 1):
     any run with E > 0 refutes X for that window
  6. first twin slot vs first double slot above y (the bottom-band race)
"""
import sympy


def window_slots(y):
    """Slots with both members strictly inside (y, y^2)."""
    k_lo = y // 6 + 1
    while 6 * k_lo - 1 <= y:
        k_lo += 1
    k_hi = (y * y - 2) // 6  # largest k with 6k+1 < y^2
    return list(range(k_lo, k_hi + 1))


def lpf(n):
    return min(sympy.factorint(n))


def analyse(y):
    gears = list(sympy.primerange(5, y + 1))
    slots = window_slots(y)
    N = len(slots)

    # census
    n0 = n1 = n2 = P = 0
    slot_kind = {}          # k -> #composite members (0,1,2)
    root_of = {}            # composite member -> its root gear (lpf)
    first_twin = first_double = None
    for k in slots:
        comps = 0
        for m in (6 * k - 1, 6 * k + 1):
            if sympy.isprime(m):
                P += 1
            else:
                comps += 1
                root_of[m] = lpf(m)
        slot_kind[k] = comps
        if comps == 0:
            n0 += 1
            if first_twin is None:
                first_twin = k
        elif comps == 1:
            n1 += 1
        else:
            n2 += 1
            if first_double is None:
                first_double = k
    C = 2 * N - P
    assert n0 + n1 + n2 == N and n1 + 2 * n2 == C and P == n1 + 2 * n0

    # per-gear root supply, exact, and the semiprime formula
    R = {q: 0 for q in gears}
    for m, q in root_of.items():
        R[q] += 1
    assert sum(R.values()) == C, "root attribution must partition composites"
    formula = {}
    for q in gears:
        sq = 1 if y < q * q < y * y else 0
        cop = sum(1 for r in sympy.primerange(q + 1, y * y // q + 1)
                  if y < q * r < y * y)
        formula[q] = (sq, cop)

    # pseudo-twin ledger
    PT = {q: 0 for q in gears}
    for m, q in root_of.items():
        partner = m + 2 if m % 6 == 5 else m - 2
        if sympy.isprime(partner):
            PT[q] += 1
    assert sum(PT.values()) == n1 == P - 2 * n0

    # prefix margin and max run excess
    margin, min_margin, argmin = 0, 0, None
    run, best_run, best_lo, best_hi, lo = 0, 0, None, None, slots[0]
    for k in slots:
        p_k = 2 - slot_kind[k]                  # primes in slot k
        margin += 1 - p_k                       # = n2 - n0 prefix
        if margin < min_margin:
            min_margin, argmin = margin, k
        run += p_k - 1                          # Kadane for max run excess
        if run < 0:
            run, lo = 0, k + 1
        elif run > best_run:
            best_run, best_lo, best_hi = run, lo, k

    print(f"\n=== y = {y}  gears {gears} ===")
    print(f"slots k = {slots[0]}..{slots[-1]}  N = {N}  members "
          f"{6*slots[0]-1}..{6*slots[-1]+1}")
    print(f"P = {P}  C = {C}  census: twins n0={n0}  fragile n1={n1}  "
          f"double n2={n2}")
    print(f"X demands: n1 = P = {P}, n2 = N - P = {N - P}"
          + ("  << IMPOSSIBLE, N - P < 0" if N - P < 0 else ""))
    print(f"global margin N - P = {N - P};  supply C - demand N = {C - N}")
    print("gear:  exact R(q) | square+coprime formula | higher-order | PT(q)")
    for q in gears:
        sq, cop = formula[q]
        print(f"  {q:3d}: {R[q]:5d}     | {sq}+{cop:4d} = {sq+cop:4d}"
              f"        | {R[q]-sq-cop:4d}         | {PT[q]:4d}")
    print(f"pseudo-twin ledger: demand P = {P}, supply sum PT = "
          f"{sum(PT.values())}, deficit = {P - sum(PT.values())} = 2*n0 = {2*n0}")
    print(f"prefix margin: min = {min_margin} at k = {argmin} "
          f"(member ~{6*argmin+1 if argmin else '-'}); X needs >= 0 everywhere")
    print(f"max run excess E = {best_run} on slots {best_lo}..{best_hi} "
          f"(members {6*best_lo-1}..{6*best_hi+1}); any E > 0 refutes X")
    print(f"bottom-band race: first twin at k = {first_twin} "
          f"(slot #{slots.index(first_twin)+1} of window), first double at "
          f"k = {first_double} (slot #{slots.index(first_double)+1})")


if __name__ == "__main__":
    for y in (13, 23, 47):
        analyse(y)
