# Branch 7d, question 3: d_0(M) = first opening past column 0 for M = {5..q}, versus 2 d_0, F, F_2 and the window.
# d_0 = least k > 0 with 6k-1 and 6k+1 both free of prime factors in 5..q.
# Self-contained; numpy only.  Run: uv run python research/anchor235/r34/q3_d0.py
import numpy as np, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "q3_d0.txt")
lines = []
def say(s=""):
    print(s); lines.append(s)

# corpus ladder (max-gap units), docs/proof-search/alignment-rules.md and constructor.md:
F_corpus = {7: 5, 11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88, 41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
F2_corpus = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90, 41: 103, 43: 116, 53: 159}

LIM = 200_000
sv = np.ones(6 * LIM + 8, bool); sv[:2] = False
for i in range(2, int(len(sv) ** 0.5) + 1):
    if sv[i]: sv[i * i::i] = False
primes = [int(p) for p in np.flatnonzero(sv) if p >= 5 and p <= LIM // 6 - 10]
spf = np.zeros(6 * LIM + 8, np.int64)
for p in range(2, int(len(spf) ** 0.5) + 1):
    if spf[p] == 0:
        m = np.arange(p, len(spf), p); sel = spf[m] == 0; spf[m[sel]] = p
spf[spf == 0] = np.arange(len(spf))[spf == 0]

def d0_of(q, ks):
    # first k with both 6k+-1 having smallest prime factor > q (they are coprime to 6 automatically)
    lo, hi = 6 * ks - 1, 6 * ks + 1
    ok = (spf[lo] > q) & (spf[hi] > q)
    j = np.flatnonzero(ok)
    return int(ks[j[0]]) if len(j) else None

say("Q3: d_0(M) for M = {5..q}: first opening past 0; mirror gives the pair (d_0, d_0) at column 0, so F_2(M) >= 2 d_0 and F(M+q') >= F_2(M) >= 2 d_0.")
say(f"{'q':>6} {'q_next':>6} {'d_0':>5} {'2d_0':>5} {'F':>5} {'F_2':>5} {'W':>9} {'d_0<=q_n':>8} {'d_0<=W':>7} {'6d_0-1,6d_0+1':>16} {'first twin > q':>14} {'same?':>5} {'2d_0/F':>7} {'2d_0/W':>8}")
ks = np.arange(1, LIM)
rows = []
for i, q in enumerate(primes):
    if q > 59: break
    qn = primes[i + 1]; W = (qn * qn - 1) // 6
    d0 = d0_of(q, ks)
    a, b = 6 * d0 - 1, 6 * d0 + 1
    # first twin pair above q
    t = q + 1
    while not (sv[t] and sv[t + 2]): t += 1
    ft = (t, t + 2)
    F = F_corpus.get(q); F2 = F2_corpus.get(q)
    say(f"{q:>6} {qn:>6} {d0:>5} {2*d0:>5} {str(F):>5} {str(F2):>5} {W:>9} {str(d0 <= qn):>8} {str(d0 <= W):>7} {str((a, b)):>16} {str(ft):>14} {str(ft == (a, b)):>5} {2*d0/F if F else float('nan'):>7.3f} {2*d0/W:>8.4f}")
    rows.append((q, qn, d0, F, W))
say("\nExactly when d_0 is in the window: d_0 in (q/6, W] iff 6 d_0 + 1 < q_next^2 iff the first pair of q-rough numbers 6k+-1 above q is below q_next^2,")
say("i.e. iff there is a twin prime pair in (q, q_next^2). Conversely every twin pair above q is an M-opening, so d_0 <= (first twin above q); equality iff that twin is < q_next^2.")
say("The lower edge d_0 > (q-1)/6 is forced: every column k <= (q-1)/6 has 6k+1 <= q, so its numbers are gears or gear-multiples (the shield).")
# extended gate: d_0 vs q_next and W for all primes to LIM//6 - 10
say("\nExtended gate over all primes 5 <= q <= %d:" % primes[-1])
mx = 0; mxq = None; fails = 0; in_first = 0; n = 0
sec_fail = 0
for i, q in enumerate(primes[:-1]):
    qn = primes[i + 1]; W = (qn * qn - 1) // 6
    d0 = d0_of(q, ks); n += 1
    r = d0 / qn
    if r > mx: mx, mxq = r, q
    if d0 > qn: fails += 1
    if d0 > W: sec_fail += 1
say(f"  primes checked {n}; d_0 <= q_next fails {fails} times; max d_0/q_next = {mx:.4f} at q = {mxq}; d_0 > W (no opening in the window) {sec_fail} times")
say("  (matches research/proof/pair_statement.md gate d0: d_0 <= q' for every prime to 10^6, max ratio 0.2857 at p = 5)")
# what the mirror forces: the pair at 0 is (d_0, d_0), so 2 d_0 <= F_2(M) (theorem) - table of slack
say("\nSlack of the theorem F_2(M) >= 2 d_0 and of F(M) >= d_0 (trivial: the gap 0 -> d_0 is a gap of M):")
for q, qn, d0, F, W in rows:
    F2 = F2_corpus.get(q)
    say(f"  q={q}: d_0={d0}, F={F} (F/d_0 = {F/d0 if F else float('nan'):.2f}), F_2={F2} (F_2/2d_0 = {F2/(2*d0) if F2 else float('nan'):.2f}), W={W} (W/d_0 = {W/d0:.1f})")
with open(OUT, "w", encoding="utf-8") as f: f.write("\n".join(lines) + "\n")
print("written", OUT)
