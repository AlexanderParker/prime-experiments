"""r65 / position-length frontier, part 3: the mechanism at the frontier.

For a machine {5..q} and a stretch (x, L) [columns x .. x+L-1, all blocked]:
  - the effective machine E = {5..y}, y = the largest prime with y^2 <= 6(x+L-1)+1;
  - per column: the effective strikers (gears <= y) and the big strikers (gears in (y, q]);
  - EXCLUSIVE big columns: columns with no effective striker at all (the layer law says 0);
  - S = big-gear strikes, and the number of columns they touch;
  - the decomposition of the stretch into pieces of E fused by big gears (ends or middles).

Also: the exception set of the layer law over the whole prefix [1, W] -- every column with no
effective striker -- and the identification of that set with the twin gear pairs.

Self-contained, numpy only.  Run: uv run python research/anchor235/r65/pf_mech.py
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def factors(n, isp):
    out = []
    d = 5
    m = n
    while d * d <= m:
        if m % d == 0:
            out.append(d)
            while m % d == 0:
                m //= d
        d += 2 if d % 6 == 5 else 4   # wheel 5,7,11,13,17,19,...
    if m > 1:
        out.append(m)
    return out


def strikers(k, q, isp):
    s = set()
    for n in (6 * k - 1, 6 * k + 1):
        for p in factors(n, isp):
            if 5 <= p <= q:
                s.add(p)
    return s


def analyse(q, x, L, isp, label, W):
    top = 6 * (x + L - 1) + 1
    y = int(np.floor(np.sqrt(top)))
    while y >= 2 and not isp[y]:
        y -= 1
    y = max(y, 4)
    cols = list(range(x, x + L))
    eff_free = []
    big_strikes = 0
    big_cols = set()
    effmask = []
    per = []
    for k in cols:
        st = strikers(k, q, isp)
        e = sorted(p for p in st if p <= y)
        b = sorted(p for p in st if p > y)
        per.append((k, e, b))
        big_strikes += len(b)
        if b:
            big_cols.add(k)
        if not e:
            eff_free.append((k, 6 * k - 1, 6 * k + 1, b))
        effmask.append(bool(e))
    # pieces of the effective machine inside the stretch
    pieces = []
    run = 0
    st0 = None
    for i, v in enumerate(effmask):
        if v:
            if run == 0:
                st0 = cols[i]
            run += 1
        else:
            if run:
                pieces.append((st0, run))
            run = 0
    if run:
        pieces.append((st0, run))
    W("  %s: q=%d  x=%d  L=%d  top member=%d  effective machine {5..%d}  y/q=%.3f"
      % (label, q, x, L, top, y, y / q))
    W("     big gears in the machine: %d ; big-gear strikes S=%d on %d of %d columns"
      % (sum(1 for p in range(y + 1, q + 1) if isp[p]), big_strikes, len(big_cols), L))
    W("     columns with NO effective striker (layer-law exceptions): %d %s"
      % (len(eff_free), eff_free[:8]))
    W("     pieces of E inside the stretch: %d, sizes %s"
      % (len(pieces), [p[1] for p in pieces]))
    if len(pieces) > 1:
        junc = [p[0] + p[1] for p in pieces[:-1]]
        W("     fused at columns %s by %s"
          % (junc, [sorted(b) for k, e, b in per if k in junc]))
    return len(eff_free), y


def main():
    lines = []
    W = lines.append
    NM = 2_000_000
    W("sieving to %d ..." % NM)
    isp = sieve(NM)

    W("")
    W("=" * 84)
    W("A. THE PERIOD FRONTIER, m23: every Pareto point")
    W("=" * 84)
    m23 = [(1, 4), (53, 5), (59, 11), (111, 24), (40148, 25), (170034, 26), (190056, 27),
           (396199, 29), (1479278, 30), (2553844, 31), (5606403, 32), (12694429, 33)]
    tot_exc = 0
    for x, L in m23:
        e, y = analyse(23, x, L, isp, "pareto", W)
        tot_exc += e
    W("")
    W("  total layer-law exceptions over the 12 m23 Pareto stretches: %d" % tot_exc)

    W("")
    W("=" * 84)
    W("B. THE PERIOD FRONTIER, other machines: the minimum-ratio point and the record")
    W("=" * 84)
    for q, x, L, lab in [(7, 13, 4, "min-ratio (c=3.25)"), (11, 13, 4, "min-ratio"),
                         (13, 13, 4, "min-ratio"), (13, 123, 10, "record"),
                         (17, 61, 9, "min-ratio"), (17, 118, 17, "record"),
                         (19, 111, 24, "min-ratio + record"),
                         (29, 200906186, 42, "record")]:
        analyse(q, x, L, isp, lab, W)

    W("")
    W("=" * 84)
    W("C. THE WINDOW FRONTIER: the window's longest stretch and the initial run")
    W("=" * 84)
    for q, x, L, lab in [(23, 111, 24, "window longest"), (97, 981, 34, "window longest"),
                         (211, 4071, 82, "window longest"),
                         (997, 141726, 241, "window longest"),
                         (23, 1, 4, "initial run"), (97, 1, 16, "initial run"),
                         (211, 1, 37, "initial run"), (997, 1, 169, "initial run")]:
        analyse(q, x, L, isp, lab, W)

    W("")
    W("=" * 84)
    W("D. THE LAYER LAW'S EXCEPTION SET over the whole prefix [1, W]")
    W("=" * 84)
    W("  a column has no effective striker iff BOTH members are prime; it is blocked iff a")
    W("  member is a gear.  So the exception set = twin columns whose smaller member <= q.")
    W("")
    W("     q    W    exceptions found   twin gear pairs 5<=p<=q   all below (q+1)/6   max col")
    for q in [23, 29, 47, 97, 211, 401, 997]:
        qp = q + 1
        while not isp[qp]:
            qp += 1
        Wc = (qp * qp - 1) // 6
        exc = []
        for k in range(1, Wc + 1):
            a, b = 6 * k - 1, 6 * k + 1
            if isp[a] and isp[b] and (a <= q or b <= q):
                exc.append(k)
        tw = sum(1 for p in range(5, q + 1) if isp[p] and isp[p + 2])
        thr = (q + 1) // 6
        W("  %6d %6d %14d %22d %20s %9d"
          % (q, Wc, len(exc), tw, str(all(k <= thr + 1 for k in exc)),
             max(exc) if exc else 0))
    W("")
    W("  (the count also includes a pair (p, p+2) with p <= q < p+2, i.e. the last gear when")
    W("   it is the lower member of a twin pair)")

    W("")
    W("=" * 84)
    W("E. THE SPECTRUM-PLUS-DEPTH BOUND AT x = 1 (why it is vacuous)")
    W("=" * 84)
    W("     q   init run L   effective machine {5..y}   S_excl = E-openings in the run"
      "   L / S_excl")
    for q in [97, 211, 401, 997, 4999]:
        qp = q + 1
        while not isp[qp]:
            qp += 1
        # d_0 = first twin column with both members > q
        k = 1
        while True:
            if isp[6 * k - 1] and isp[6 * k + 1] and 6 * k - 1 > q:
                break
            k += 1
        d0 = k
        L = d0 - 1
        top = 6 * L + 1
        y = int(np.floor(np.sqrt(top)))
        while not isp[y]:
            y -= 1
        s_exc = sum(1 for kk in range(1, d0)
                    if isp[6 * kk - 1] and isp[6 * kk + 1])
        W("  %6d %11d %26d %30d %11.2f"
          % (q, L, y, s_exc, L / max(s_exc, 1)))
    W("")
    W("  S_excl is exactly the number of openings the effective machine leaves inside the")
    W("  run, so 'L+1 <= F_{S+1}(E)' with S = S_excl is true by the definition of F_J and")
    W("  carries no information: the bound degenerates to an identity at x = 1.")

    txt = "\n".join(lines)
    with open(os.path.join(OUT, "pf_mech.txt"), "w") as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
