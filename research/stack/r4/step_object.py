"""Evidence for the step (R4.d.i), generated locally while the lanes are paused (2026-09-08).

Objects (the stack by squares): cuts c_1 = base, c_{k+1} = nextprime(c_k)^2 (with c_1 itself a
prime, p_1 = c_1); section k = [c_k, c_{k+1}); machine k = the primes in section k; a SLOT is
(n, n + 2) with n = 5 mod 6; machine k's TWIN GEARS are its gears at distance 2; band k = section
k seen as the range where exactly machines 1 .. k-1 act genuinely (proved), so a slot in
section k open under machines 1 .. k-1 is a twin prime pair.

What this script measures, per base in BASES and per chain link that fits below LIMIT:
  A. the chain: each link's cut, first gear, gear count, twin-gear count (= the twins of the
     section), the first twin above the cut and its offset in numbers and in cycles of 30;
  B. the step's object on section k+1: strikes on slots by machine k's twin gears versus its
     other gears; the collisions of each twin pair (slots struck by both members) against the
     pair's total strikes: the waste fraction, expected 1/(p + 2) per pair by arithmetic;
  C. the start-of-section profile: twins per cycle of 30 for the first 200 cycles above each
     cut, against the count of open slots under machines 1 .. k-1 alone (before machine k's
     gears engage) and the number of machine-k gears engaged by that height (g <= x / c_k).
usage: uv run python research/stack/r4/step_object.py [LIMIT]
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000_000
BASES = [3, 5, 7, 11, 13, 17, 19, 23]


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def nextprime(n, isp):
    m = n + 1
    while not isp[m]:
        m += 1
    return m


def chain(base, isp, limit):
    cuts = [base]
    firsts = [base]
    while True:
        p = firsts[-1]
        c = p * p
        if c > limit:
            break
        cuts.append(c)
        firsts.append(nextprime(c - 1, isp) if isp[c] else nextprime(c, isp))
    return cuts, firsts


def main():
    t0 = time.time()
    isp = sieve(LIMIT + 40)
    twin = isp.copy()
    twin[:-2] &= isp[2:]  # twin[n] True iff n and n+2 prime
    twin[-2:] = False
    print(f"sieve to {LIMIT} in {time.time()-t0:.0f}s", flush=True)
    out = []
    for base in BASES:
        cuts, firsts = chain(base, isp, LIMIT)
        out.append(f"\n# chain from base {base}: cuts {cuts}; first gears {firsts}")
        for k in range(len(cuts)):
            lo = cuts[k] if k > 0 else 5  # machine 1 = every prime from 5 below the first cut (2, 3 are the fold)
            hi = cuts[k + 1] if k + 1 < len(cuts) else None
            if hi is None or hi > LIMIT:
                break
            gears = np.nonzero(isp[lo:hi])[0] + lo
            tg = np.nonzero(twin[lo:hi])[0] + lo
            tg = tg[tg + 2 < hi]
            # first twin above the cut (both members above lo)
            ft = int(np.argmax(twin[lo + 1:hi])) + lo + 1 if twin[lo + 1:hi].any() else -1
            off = ft - lo if ft > 0 else -1
            out.append(f"link {k+1}: section [{lo}, {hi}) gears {len(gears)} twin-gear pairs {len(tg)} first twin {ft} (offset {off} numbers, {off/30:.1f} cycles; arc of the first gear {firsts[k]}: {(2*firsts[k]+1)//3*6} numbers)")
            # B. the step's object on the NEXT section, if it fits
            if k + 2 < len(cuts) and cuts[k + 2] <= LIMIT and (cuts[k + 2] - cuts[k + 1]) // 6 <= 6_000_000:
                nlo, nhi = cuts[k + 1], cuts[k + 2]
                L = nhi - nlo
                # slots in the next section: n = 5 mod 6 in [nlo, nhi)
                n0 = nlo + ((5 - nlo) % 6)
                slots = np.arange(n0, nhi, 6, dtype=np.int64)
                struck_twin = np.zeros(slots.size, dtype=bool)
                struck_other = np.zeros(slots.size, dtype=bool)
                waste_num = 0
                waste_den = 0
                tgset = set(int(x) for x in tg) | set(int(x) + 2 for x in tg)
                for g in gears:
                    g = int(g)
                    hit = ((slots % g) == 0) | (((slots + 2) % g) == 0)
                    if g in tgset:
                        struck_twin |= hit
                    else:
                        struck_other |= hit
                for p in tg:
                    p = int(p)
                    h1 = ((slots % p) == 0) | (((slots + 2) % p) == 0)
                    h2 = ((slots % (p + 2)) == 0) | (((slots + 2) % (p + 2)) == 0)
                    waste_num += int((h1 & h2).sum())
                    waste_den += int(h1.sum()) + int(h2.sum())
                both = struck_twin & struck_other
                only_twin = struck_twin & ~struck_other
                only_other = struck_other & ~struck_twin
                open_ = ~(struck_twin | struck_other)
                out.append(f"   on section {k+2} [{nlo}, {nhi}): slots {slots.size}; struck only by twin gears {int(only_twin.sum())}, only by other gears {int(only_other.sum())}, by both {int(both.sum())}, open (twins of the section, machine {k+1} alone) {int(open_.sum())}; twin-pair collisions {waste_num} of {waste_den} strikes = waste {waste_num/max(waste_den,1):.6f} (arithmetic 1/(p+2) averages {np.mean([1/(int(p)+2) for p in tg]) if len(tg) else 0:.6f})")
            # C. start-of-section profile (twins per cycle for 200 cycles above the cut, against the count under lower machines only)
            if k >= 1:
                ncyc = min(200, (hi - lo) // 30)
                n0 = lo + ((5 - lo) % 6)
                slots = np.arange(n0, lo + 30 * ncyc, 6, dtype=np.int64)
                lower = np.ones(slots.size, dtype=bool)
                for j in range(k):
                    jlo = cuts[j] if j > 0 else 5
                    for g in np.nonzero(isp[jlo:cuts[j + 1]])[0] + jlo:
                        g = int(g)
                        lower &= ~(((slots % g) == 0) | (((slots + 2) % g) == 0))
                tw = twin[slots]
                cyc = (slots - lo) // 30
                per = []
                for c in range(min(ncyc, 20)):
                    m = cyc == c
                    per.append((int(tw[m].sum()), int(lower[m].sum())))
                engaged = [int(np.sum(gears <= (lo + 30 * c) / firsts[k])) for c in (1, 5, 20, 100, 200)]
                out.append(f"   start of section {k+1}: (twins, open under machines 1..{k}) per cycle for cycles 0..19: {per}; totals over {ncyc} cycles: twins {int(tw.sum())}, lower-open {int(lower.sum())}; machine-{k+1} gears engaged by cycle 1, 5, 20, 100, 200: {engaged} of {len(gears)}")
        print(f"base {base} done {time.time()-t0:.0f}s", flush=True)
    txt = "\n".join(out)
    open(os.path.join(RES, f"step_object_{LIMIT}.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
