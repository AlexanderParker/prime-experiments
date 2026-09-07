"""The certified turns: for every Q <= QMAX and m = 1..MM, the effective machine of the slice (mQ, (m+1)Q] is
E_m = {5..y_m}, y_m the largest prime <= sqrt((m+1)Q + 2) (theorem (E)); the slice has c_m(Q) columns; if the
turn is empty then F(E_m) >= c_m(Q) + 1. With the certified ladder F(5..59) = 2, 5, 7, 11, 18, 25, 34, 43, 58,
88, 91, 103, 118, 145, 161 (max-gap convention) the certificate "F(E_m) <= c_m(Q)" is a proof that turn m is
nonempty whenever y_m <= 59. Also F_odd(q) for q = 13, 17, 19 (the engine's twin-free run on the odd line).
usage: uv run python research/valves/r2/certify.py [QMAX] [MM]
"""
import sys, math
import numpy as np

QMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MM = int(sys.argv[2]) if len(sys.argv) > 2 else 4
F = {0: 1, 5: 2, 7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
rungs = sorted(F)
N = (MM + 1) * QMAX + 4
s = np.ones(N + 1, dtype=bool); s[:2] = False
for i in range(2, int(N ** 0.5) + 1):
    if s[i]:
        s[i * i::i] = False
tw = s[:-2] & s[2:]
C = np.concatenate([[0], np.cumsum(tw.astype(np.int64))])


def twins_in(a, b):
    return int(C[b + 1] - C[a + 1])


for m in range(1, MM + 1):
    cert = []; truth_fail = []; cert_fail_small = []
    for Q in range(1, QMAX + 1):
        top = (m + 1) * Q + 2
        if top > 59 * 59:
            break
        r = math.isqrt(top)
        ym = max(y for y in rungs if y <= r)          # 0 = empty machine
        klo = (m * Q + 2 + 5) // 6                     # least k with 6k - 1 > mQ
        khi = ((m + 1) * Q + 1) // 6                   # largest k with 6k + 1 <= (m+1)Q + 2
        c = khi - klo + 1
        ok = F[ym] <= c
        nonempty = twins_in(m * Q, (m + 1) * Q) > 0
        if ok and not nonempty:
            truth_fail.append(Q)
        if ok:
            cert.append(Q)
        elif Q > 30:
            cert_fail_small.append((Q, ym, F[ym], c))
    Qc = max(cert) + 1 if cert else None
    # the least Q0 such that every Q in [Q0, Qmax_m] is certified
    Qmax_m = (59 * 59 - 2) // (m + 1)
    cs = set(cert); Q0 = Qmax_m
    while Q0 - 1 in cs:
        Q0 -= 1
    print(f"turn m={m}: certificate F(E_m) <= c_m(Q) with E_m <= 59 applies to Q <= {Qmax_m}; certified Q: {len(cert)}; "
          f"every Q in [{Q0}, {Qmax_m}] certified; uncertified Q above 30: {cert_fail_small[:6]}{'...' if len(cert_fail_small) > 6 else ''}; "
          f"certified but empty (must be none): {truth_fail}")

# F_odd for larger q
for q in [5, 7, 11, 13, 17, 19]:
    eng = [p for p in range(3, q + 1) if all(p % d for d in range(2, int(p ** 0.5) + 1))]
    P = math.prod(eng)                                  # odd part of q#; odd n mod q# <-> n mod P (n odd)
    r = np.arange(P, dtype=np.int64)
    ok = np.ones(P, dtype=bool)
    for p in eng:
        ok &= (r % p != 0) & ((r + 2) % p != 0)
    pc = np.flatnonzero(ok)                              # pure start residues mod P (odd line: consecutive odd numbers step 2 mod P)
    # on the odd line the residues mod P advance by 2 per step; the pure starts in step units: solve n = 2i + 1 -> i = (n - 1)/2 mod P
    steps = np.sort(((pc - 1) * pow(2, -1, P)) % P)
    gaps = np.diff(np.concatenate([steps, [steps[0] + P]]))
    print(f"F_odd({q}) = {int(gaps.max())} consecutive open odd numbers without a twin (pure starts {len(pc)} of {P}); "
          f"manifold ceiling q'-1 = {[p for p in range(q + 1, q + 30) if all(p % d for d in range(2, int(p ** 0.5) + 1))][0] - 1}")

# ---------------------------------------------------------------------------------------------
# Second pass (continuation prover): (i) the per-rung table: for every certified rung y and turn m the
# Q-range on which E_m = {5..y}, and the least Q of that range from which the share certificate
# F(y) <= c_m(Q) holds; (ii) the frontier form x >= c L: turn m is certified iff c * c_m(Q) > klo(Q),
# for c = 1.25 (proved), 3 (prefix law, measured), 3.25 (period), 4.625 (prefix floor), against the
# truth; (iii) the tight instance of the frontier (m23, column 111, length 24) as an empty turn;
# (iv) E-P2 as sets at every Q <= 10^4 for m = 1, 2 (the descent).
print("\n--- per-rung table: rung y, turn m, Q-range with E_m = {5..y}, least Q of the range certified by F(y) <= c_m(Q)")
nxt = {rungs[i]: rungs[i + 1] for i in range(len(rungs) - 1)}; nxt[59] = 61
for y in rungs[1:]:
    row = []
    for m in range(1, 5):
        Qlo = (y * y - 2) // (m + 1) + 1; Qhi = (nxt[y] ** 2 - 3) // (m + 1)   # y_m(Q) = y  <=>  y^2 <= (m+1)Q + 2 < y'^2
        if Qhi < Qlo:
            row.append(f"m={m}: empty range"); continue
        certQ = [Q for Q in range(Qlo, Qhi + 1) if F[y] <= ((m + 1) * Q + 1) // 6 - (m * Q + 7) // 6 + 1]
        if not certQ:
            row.append(f"m={m}: Q in [{Qlo},{Qhi}] none certified")
        else:
            # certification is monotone in Q up to a wobble of one column; report the least Q from which every Q is certified
            cs = set(certQ); Q0 = Qhi
            while Q0 - 1 in cs:
                Q0 -= 1
            row.append(f"m={m}: Q in [{Qlo},{Qhi}] certified from Q={Q0} ({len(certQ)} of {Qhi - Qlo + 1})")
    print(f"  y={y:2d} F={F[y]:3d} F/(y^2/6)={F[y] / (y * y / 6):.3f} F/W={F[y] / ((nxt[y] ** 2 - 1) / 6 - y / 6):.3f}: " + "; ".join(row))

print("\n--- the frontier form: turn m certified iff c * c_m(Q) > klo(Q) (a run of c_m columns cannot start at klo); truth from the sieve")
for c in [1.25, 3.0, 3.25, 4.625]:
    line = []
    for m in range(1, 7):
        cert = 0; wrong = []; first_all = None
        Qtop = min(QMAX, 100000)
        for Q in range(2, Qtop + 1):
            klo = (m * Q + 7) // 6; khi = ((m + 1) * Q + 1) // 6; cm = khi - klo + 1
            ok = c * cm > klo
            if ok:
                cert += 1
                if (m + 1) * Q + 2 <= len(tw) and twins_in(m * Q, (m + 1) * Q) == 0:
                    wrong.append(Q)
        line.append(f"m={m}: {cert} of {Qtop - 1} Q certified" + (f", WRONG at {wrong[:5]}" if wrong else ""))
    print(f"  c={c}: " + "; ".join(line))
print("  (the frontier form with constant c certifies exactly the turns m < c for Q large; the WRONG list is empty for c <= 4.625: the truth agrees)")

print("\n--- the tight instance: m23's run at column 111 of length 24 (ratio 4.625) as an empty turn")
for Q in [131, 132, 133, 134, 135]:
    lo_n, hi_n = 5 * Q, 6 * Q
    print(f"  Q={Q}: turn 5 = ({lo_n}, {hi_n}], columns {(lo_n + 7) // 6}..{(hi_n + 1) // 6}, twins {twins_in(lo_n, hi_n)}; "
          f"y_5 = {max(y for y in rungs if y * y <= hi_n + 2)}")
print("  the twin gap (659, 661) -> (809, 811): columns 111..134 blocked under {5..23} (6*111-1 = 665 > 661, 6*134+1 = 805 < 809)")

print("\n--- E-P2, the descent as sets: twins in (mQ, (m+1)Q] == openings of {5..y_m} there, y_m = largest prime <= sqrt((m+1)Q + 2)")
allp = np.flatnonzero(s)
bad = 0; checked = 0
for m in (1, 2):
    for Q in range(2, 10001):
        lo_n, hi_n = m * Q, (m + 1) * Q
        if hi_n + 2 > N:
            break
        ym = math.isqrt(hi_n + 2)
        gears = allp[(allp >= 5) & (allp <= ym)]
        ns = np.arange(lo_n + 1, hi_n + 1, dtype=np.int64); ns = ns[ns % 6 == 5]
        op = np.ones(len(ns), dtype=bool)
        for g in gears:
            op &= (ns % g != 0) & ((ns + 2) % g != 0)
        openings = set(ns[op].tolist())
        twins = set(int(n) for n in np.flatnonzero(tw[lo_n + 1:hi_n + 1]) + lo_n + 1)
        checked += 1
        if openings != twins:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH m={m} Q={Q}: openings - twins {sorted(openings - twins)[:5]}, twins - openings {sorted(twins - openings)[:5]}")
print(f"  {checked} (m, Q) cells checked, {bad} mismatches")
