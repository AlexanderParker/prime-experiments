# anchor 2,3,5. middle-run alignment: twin slot k (anchor-open: k mod 5 in {0,2,3}) is aligned when it avoids the two
# hit residues +-u_q of every gear 7 <= q <= sqrt(6k+1). Per window (q^2, q_next^2): how many aligned slots,
# longest stretch of anchor-open slots all hit by some gear, versus the window length.
import sys
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), ".."))
import numpy as np
from word_tree_r29 import spf_sieve
QMAX = 5000
spf = spf_sieve(QMAX * QMAX + 10).astype(np.int64)
primes = [int(x) for x in np.flatnonzero(spf_sieve(QMAX + 100) == np.arange(QMAX + 101)) if x >= 5]
print("rung q->q'  window slots  anchor-open  aligned(twins)  longest blocked run (anchor-open slots / numbers)  run/window")
worst = (0, None)
rows = []
for i in range(len(primes) - 1):
    q, qn = primes[i], primes[i + 1]
    if qn > QMAX: break
    k_lo, k_hi = q * q // 6 + 1, (qn * qn - 2) // 6
    ks = np.arange(k_lo, k_hi + 1)
    lo, hi = 6 * ks - 1, 6 * ks + 1
    ao = np.isin(ks % 5, (0, 2, 3))
    aligned = ao & (spf[lo] == lo) & (spf[hi] == hi)
    W = len(ks)
    # longest run of consecutive anchor-open slots that are blocked
    a_idx = np.flatnonzero(ao); al = aligned[a_idx]
    best = cur = 0; bstart = None; cstart = 0
    for j, v in enumerate(al):
        if v: cur = 0; cstart = j + 1
        else:
            cur += 1
            if cur > best: best, bstart = cur, cstart
    run_numbers = int(6 * (ks[a_idx[bstart + best - 1]] - ks[a_idx[bstart]]) + 2) if best else 0
    ratio = run_numbers / (6 * W)
    rows.append((q, qn, W, int(ao.sum()), int(aligned.sum()), best, run_numbers, ratio))
    if ratio > worst[0]: worst = (ratio, rows[-1])
for r in rows[:8] + [x for x in rows if x[0] in (97, 199, 997, 1999, 4999)]:
    print(f"  {r[0]:>5}->{r[1]:<5} {r[2]:>8} {r[3]:>10} {r[4]:>10}   {r[5]:>6} / {r[6]:<8} {r[7]:.3f}")
print(f"worst run/window over all rungs to {QMAX}: {worst[0]:.3f} at rung {worst[1][0]}->{worst[1][1]} (run {worst[1][6]} numbers, window {6*worst[1][2]} numbers)")
big = sorted(rows, key=lambda r: -r[6])[:5]
print("longest blocked runs in numbers: " + ", ".join(f"{r[6]} at {r[0]}->{r[1]} (window {6*r[2]})" for r in big))
# the aligned count against the product law: aligned / (anchor-open * prod_{7<=g<=q}(1-2/g))
print("aligned / (anchor-open x prod(1-2/g), 7<=g<=q):")
for r in rows:
    if r[0] in (97, 499, 997, 1999, 2999, 3989, 4999):
        pr = np.prod([1 - 2 / g for g in primes if 7 <= g <= r[0]])
        print(f"  q={r[0]}: {r[4] / (r[3] * pr):.3f}")
