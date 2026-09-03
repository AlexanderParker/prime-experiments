"""Round-29 formalist gate: the phase-reduction record law at machine 17,
in EXACTLY the encoding the Lean file AnchorRecord17 checks.

Layer 17 sits on M = {5,7,11,13} (period 5005, 1485 openings).  In the full
period 85085 = 17 * 5005 the copies realise every deletion phase r in Z_17
exactly once; copy j deletes the lower openings whose ABSOLUTE residue mod 17
lies in the two-class set {r, r+d}, d = 2*u_17 = 6.  So

    mg(r) = max over consecutive survivors (a, b), a in [0, P), of b - a,
            survivors = { y : lowOpen(y mod P) and y % 17 not in {r, (r+6)%17} }
    max over r < 17 of mg(r)  =  F(17) + 1   (the merged-gap / record law)

with F in the BLOCKED-COUNT convention (F = maxgap - 1), i.e. chain_depth.py's
`F_17`.  This script prints mg(r) for every r, the max, the corpus max gap of
the real machine {5,7,11,13,17} computed independently over the full period
85085, and asserts they agree.
"""

P = 5005
TEETH = {5: (1, 4), 7: (6, 1), 11: (2, 9), 13: (11, 2), 17: (3, 14)}
D17 = 6          # 2 * u_17 mod 17
LOOK = 64        # walk this far past the period; max gap is far below this


def low_open(k: int) -> bool:
    for g in (5, 7, 11, 13):
        a, b = TEETH[g]
        if k % g == a or k % g == b:
            return False
    return True


def mg(r: int) -> int:
    r2 = (r + D17) % 17
    last = -1
    best = 0
    for y in range(P + LOOK):
        if low_open(y % P) and y % 17 != r and y % 17 != r2:
            if last >= 0 and last < P and y - last > best:
                best = y - last
            last = y
    return best


def full_machine_max_gap() -> int:
    FP = P * 17
    ops = [k for k in range(FP)
           if low_open(k % P) and k % 17 != 3 and k % 17 != 14]
    best = 0
    for i in range(len(ops) - 1):
        d = ops[i + 1] - ops[i]
        if d > best:
            best = d
    # wrap
    d = ops[0] + FP - ops[-1]
    if d > best:
        best = d
    return best, len(ops)


def main():
    vals = [mg(r) for r in range(17)]
    for r, v in enumerate(vals):
        print(f"  mg({r:>2}) = {v}")
    m = max(vals)
    argmax = [r for r in range(17) if vals[r] == m]
    print(f"max over phases = {m}   attained at r = {argmax}")

    fg, nops = full_machine_max_gap()
    print(f"machine 5+7+11+13+17: {nops} openings on 85085, max gap = {fg}")
    nlow = sum(1 for k in range(P) if low_open(k))
    print(f"lower machine 5+7+11+13: {nlow} openings on {P}")

    assert nlow == 1485, nlow
    assert nops == 3 * 5 * 9 * 11 * 15, nops
    assert m == fg, (m, fg)
    assert m == 18, m
    print("ALL ASSERTIONS PASSED")
    print(f"blocked-count convention: F_17 = {m - 1} (chain_depth.py's F_17); "
          f"max-gap convention: F(17) = {m} (corpus Machine17.gap_le)")


if __name__ == "__main__":
    main()
