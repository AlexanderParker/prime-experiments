"""Constructor round 5: the compression bound X needs, and the unconditional
ceilings that moment methods actually deliver on our exact class system.

Frame: interior window of y (both members in (y, y^2)), where both-gearful
<=> double (horizon; Lateral's U-pins live below the interior). Per slot k:
  wl, wr = # distinct gear divisors of left/right member
  m_k = wl * wr = # cross-root pair incidences landing on k  (m_k >= 1 iff double)
Freedom-free totals (no primality anywhere - prime <=> unmarked):
  S1(t) = sum m_k      (Mechanic's S_pair: the cross-root hit schedule)
  M2(t) = sum m_k^2    (second moment; = # compatible 4-tuple CRT co-hits,
                        expressible as floor arithmetic like S1)
  n2(t) = # {m_k >= 1}
  P(t)  = # unmarked members;  n0(t) = # both-unmarked slots (twins)

THE COMPRESSION STATEMENT (X's demand vs unconditional ceilings):
  X(y) <=> n2(t) = t - P(t) at every t <=> the hit schedule compresses into
  distinct slots at mean multiplicity  M_X(t) = S1(t)/(t - P(t))  exactly.
  Reality: M_real(t) = S1(t)/n2(t);  identity M_X/M_real = n2/(n2 - n0)... i.e.
  n2 = (t-P) + n0, so X demands compression harder by the factor 1 + n0/(t-P).
  A contradiction needs an UNCONDITIONAL ceiling: M(t) <= C(t) with
  C(t) < M_X(t) somewhere.  Tools and the ceilings they give:
    union bound (Bonferroni-1):   n2 <= S1          -> floor M >= 1, no ceiling
    Bonferroni-2:                 n2 >= S1 - S2p,  S2p = sum C(m_k,2)
                                  -> ceiling S1/(S1-S2p) IF S1 > S2p (check sign)
    Cauchy-Schwarz / Turan 2nd moment: n2 >= S1^2/M2 -> ceiling C_CS = M2/S1
  Both S2p and M2 are freedom-free arithmetic (4-tuple CRT classes), so these
  ceilings are legitimately unconditional. The test: is C_CS < M_X anywhere?
  Manager's expectation to verify/refute: C_CS lands at ~2x the need.
"""
import sys


def analyse(y, checkpoints=(0.1, 1 / 3, 1.0)):
    top = y * y
    k = y // 6 + 1
    while 6 * k - 1 <= y:
        k += 1
    ks, ke = k, (top - 2) // 6
    N = ke - ks + 1
    wl = bytearray(N)
    wr = bytearray(N)
    # mark distinct gear divisors (gears = primes 5..y, sieved on the fly)
    gs = bytearray([1]) * (y + 1)
    gs[:2] = b"\x00\x00"
    for i in range(2, int(y**0.5) + 1):
        if gs[i]:
            gs[i * i:: i] = bytearray(len(gs[i * i:: i]))
    gears = [q for q in range(5, y + 1) if gs[q]]
    for q in gears:
        m = (y // q + 1) * q
        while m < top:
            r = m % 6
            if r == 5:
                i = (m + 1) // 6 - ks
                if 0 <= i < N and wl[i] < 250:
                    wl[i] += 1
            elif r == 1:
                i = (m - 1) // 6 - ks
                if 0 <= i < N and wr[i] < 250:
                    wr[i] += 1
            m += q
    cps = sorted({int(N * f) for f in checkpoints})
    S1 = M2 = S2p = n2 = P = n0 = 0
    ci = 0
    print(f"\n=== y={y}  interior N={N} ===")
    print("     t      P     n0     n2    t-P     S1       M2     M_real  M_X   "
          "C_CS  C_CS/M_X  Bonf2")
    for i in range(N):
        a, b = wl[i], wr[i]
        if a == 0:
            P += 1
        if b == 0:
            P += 1
        if a == 0 and b == 0:
            n0 += 1
        m = a * b
        if m:
            n2 += 1
            S1 += m
            M2 += m * m
            S2p += m * (m - 1) // 2
        t = i + 1
        if ci < len(cps) and t == cps[ci]:
            ci += 1
            dem = t - P
            mx = S1 / dem if dem > 0 else float("inf")
            mr = S1 / n2 if n2 else 0
            ccs = M2 / S1 if S1 else 0
            b2 = S1 / (S1 - S2p) if S1 > S2p else float("nan")
            print(f"{t:>7} {P:>6} {n0:>6} {n2:>6} {dem:>6} {S1:>8} {M2:>9} "
                  f"{mr:>6.3f} {mx:>6.3f} {ccs:>6.3f} {ccs/mx:>7.3f}  "
                  f"{'vacuous(S2p>S1)' if S2p >= S1 else f'{b2:.3f}'}")
    # sanity: identity n2 = t - P + n0 at window end
    assert n2 == N - P + n0, "census identity"
    # CS lower bound on n2 vs truth (efficiency of the 2nd-moment tool)
    eff = (S1 * S1 / M2) / n2
    print(f"identity n2 = N - P + n0 OK; CS bound n2 >= S1^2/M2 = "
          f"{S1*S1/M2:,.0f} vs true n2 = {n2:,} (efficiency {eff:.3f}); "
          f"X needs efficiency >= (t-P)/n2 = {(N-P)/n2:.4f}")


if __name__ == "__main__":
    ys = [int(a) for a in sys.argv[1:]] or [211, 503, 2003, 5003]
    for y in ys:
        analyse(y)
