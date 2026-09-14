"""Two more constructions (owner: go for both).

1. Gears grouped by which base gear strikes their own column.  Gear g sits in column
   (6g - 1, 6g + 1).  With base gears 5 and 7 (q >= 420) the groups are: column open to every base
   gear (open), struck by 5 only (by5), by 7 only (by7), by both (by57).  With base gear 5 only
   (60 <= q < 420): open, by5.  Spiral (base kept, descending, first up) over each group.
2. The sub-machine spiral iterated (sub of sub).  Levels: b_0 = q, b_1 = floor(sqrt q),
   b_2 = floor(sqrt b_1), ... while b_k >= 5.  Level k: base_k = lowest gears with product at most
   b_k / 2, gears_k = primes in (b_{k+1}, b_k] outside base_k, spiral term 2 P_k A_k.  From the
   deepest level upward: E_k = -1 + sum over levels j >= k of 2 P_j A_j (each level's spiral run
   from the previous landing, first flip up).  Reported at each level: landing, inside that
   level's window (b_{k+1}, b_k^2]... the machine's window for k = 0, twin, passing final step.

usage: uv run python research/stack/r8/spiral_groups_levels.py 2000
"""
import sys
import numpy as np
from sympy import primerange
from pathlib import Path

def base_of(ps, bound):
    base = []; P = 1
    for p in ps:
        if P * p <= bound: P *= p; base.append(p)
        else: break
    return base, P

def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))

def main():
    Q = int(sys.argv[1]); qs = list(primerange(11, Q + 1))
    N = Q * Q + 10; sv = np.ones(N + 1, dtype=bool); sv[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sv[i]: sv[i * i::i] = False
    gnames = ['open', 'by5', 'by7', 'by57']
    gres = {n: dict(n=0, inwin=0, twinE=0, haspass=0, nopass=[]) for n in gnames}
    lres = {}
    sample = {}
    for q in qs:
        ps = list(primerange(2, q + 1)); r = int(q ** 0.5); high = [g for g in ps if g > r]
        base, P = base_of(ps, q // 2); bg = [g for g in base if g >= 5]
        rest = [p for p in ps if p not in base and p >= 5]
        def passes(E):
            return [(h, d) for d in (1, -1) for h in high if q < E + 6 * d * h and E + 6 * d * h + 2 <= q * q and sv[E + 6 * d * h] and sv[E + 6 * d * h + 2]]
        def rec(R, E, inwin):
            twinE = 0 <= E < N - 2 and bool(sv[E] and sv[E + 2]); pz = passes(E)
            R['n'] += 1; R['inwin'] += inwin; R['twinE'] += twinE; R['haspass'] += bool(pz)
            if not pz: R['nopass'].append(q)
            return twinE, pz
        # 1. groups by the base gear striking the column
        if bg:
            groups = {n: [] for n in gnames}
            for g in rest:
                s5 = 5 in bg and ((6 * g - 1) % 5 == 0 or (6 * g + 1) % 5 == 0)
                s7 = 7 in bg and ((6 * g - 1) % 7 == 0 or (6 * g + 1) % 7 == 0)
                groups['open' if not (s5 or s7) else ('by57' if (s5 and s7) else ('by5' if s5 else 'by7'))].append(g)
            for n in gnames:
                gs = sorted(groups[n], reverse=True)
                if not gs: continue
                E = -1 + 2 * P * alt(gs); inwin = q < E and E + 2 <= q * q
                twinE, pz = rec(gres[n], E, inwin)
                if q in (499, 1999): sample.setdefault(q, []).append(f"group {n:<5} {len(gs):>3} gears  E = {E:>8} {'in' if inwin else 'OUT':<4} {'TWIN' if twinE else '    '} pass {len(pz):>3}" + (f" first {pz[0]}" if pz else ""))
        # 2. levels
        bounds = [q]
        while int(bounds[-1] ** 0.5) >= 5: bounds.append(int(bounds[-1] ** 0.5))
        levels = []
        for k, b in enumerate(bounds):
            lo = bounds[k + 1] if k + 1 < len(bounds) else 1
            lps = [p for p in ps if p <= b]; bk, Pk = base_of(lps, max(b // 2, 6))   # base always holds 2 and 3, so every landing is a left member (E = 5 mod 6)
            gk = sorted([p for p in lps if p > lo and p not in bk], reverse=True)
            levels.append((k, b, lo, bk, Pk, gk))
        E = -1
        for k, b, lo, bk, Pk, gk in reversed(levels):
            if not gk: continue
            E = E + 2 * Pk * alt(gk)
            inwin = (b < E and E + 2 <= b * b) if k > 0 else (q < E and E + 2 <= q * q)
            R = lres.setdefault(f"level {k} added", dict(n=0, inwin=0, twinE=0, haspass=0, nopass=[]))
            twinE, pz = rec(R, E, inwin)
            if q in (499, 1999): sample.setdefault(q, []).append(f"levels..{k} (b = {b}, base {bk}, {len(gk)} gears) E = {E:>8} {'in' if inwin else 'OUT':<4} {'TWIN' if twinE else '    '} pass {len(pz):>3}" + (f" first {pz[0]}" if pz else ""))
    out = [__doc__.strip(), "", f"{'variant':<16}{'machines':>9}{'in window':>10}{'E twin':>8}{'has pass':>9}  no pass at"]
    for n in gnames:
        R = gres[n]
        if R['n']: out.append(f"group {n:<10}{R['n']:>9}{R['inwin']:>10}{R['twinE']:>8}{R['haspass']:>9}  {R['nopass'][:12]}{'...' if len(R['nopass']) > 12 else ''}")
    for n in sorted(lres, key=lambda s: -int(s.split()[1])):
        R = lres[n]; out.append(f"{n:<16}{R['n']:>9}{R['inwin']:>10}{R['twinE']:>8}{R['haspass']:>9}  {R['nopass'][:12]}{'...' if len(R['nopass']) > 12 else ''}")
    for q, rows in sample.items():
        out.append(""); out.append(f"q = {q}:"); out += ["   " + x for x in rows]
    Path("research/stack/r8/results_spiral_groups_levels.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
