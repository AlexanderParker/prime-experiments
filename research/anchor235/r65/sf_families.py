"""sf_families.py -- structured families of slots: size, survivors, fair rate, excess.

For every rung q (prime, 23..4999) and every located family, count the family's members in the
window (y, q'^2] and in the section (q^2, q'^2), count how many of them survive all gears up to
q (= twin prime pairs, by the kernel fact), and compare with the fair rate
prod_{y < g <= q} (1 - 2/g).

Families
  win      all columns of the window (the baseline, y = 3: no small gears removed)
  a<y>     k = 0 mod P_y                      (primorial multiples; column 0's translates)
  b<y>     k = c +- 1 mod P_y, 36 c^2 = 1     (neighbours of a full hit)
  o<y>     k open for {5..y}                  (corridor y=7,11; anchor y=13; also 17, 19)
  isl      (k - k_0) mod 35 in {5,10,12,17}, k > k_0   (islands; section only)

Writes results/families_window.tsv, results/families_section.tsv, results/islands.tsv.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

QMAX = 5000
ISLAND_OFFS = (5, 10, 12, 17)


def primes_upto(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def teeth(g):
    u = pow(6, -1, g)
    return (u % g, (-u) % g)


def crt_sets(gears, sets):
    """all residues mod prod(gears) whose residue mod each gear lies in the given set"""
    M = 1
    cur = [0]
    for g, S in zip(gears, sets):
        nxt = []
        Mg = M * g
        for r in cur:
            for s in S:
                # x = r mod M, x = s mod g
                x = r
                while x % g != s:
                    x += M
                nxt.append(x % Mg)
        cur = nxt
        M = Mg
    return M, sorted(cur)


def main():
    sieve = primes_upto(QMAX + 10)
    rungs = [int(q) for q in np.nonzero(sieve[:QMAX + 1])[0] if q >= 23]
    nxt = {}
    for i, q in enumerate(rungs):
        p = q + 1
        while not sieve[p]:
            p += 1
        nxt[q] = p
    QTOP = nxt[rungs[-1]]
    NMAX = QTOP * QTOP
    print("rungs", len(rungs), "top prime", QTOP, "NMAX", NMAX)

    isp = primes_upto(NMAX)
    KMAX = (NMAX - 1) // 6
    k = np.arange(KMAX + 1, dtype=np.int64)
    lo = 6 * k - 1
    hi = 6 * k + 1
    twin = np.zeros(KMAX + 1, dtype=bool)
    good = lo >= 2
    twin[good] = isp[lo[good]] & isp[hi[good]]
    del lo, hi, good
    print("twin columns to K =", KMAX, ":", int(twin.sum()))

    # cumulative product of (1 - 2/g) over gears
    gears = [int(g) for g in np.nonzero(sieve[:QMAX + 1])[0] if g >= 5]
    cp = {}
    v = 1.0
    prev = 3
    cp[3] = 1.0
    for g in gears:
        v *= (1.0 - 2.0 / g)
        cp[g] = v
    # cp[y] = prod_{5<=g<=y}(1-2/g); rate_y(q) = cp[q]/cp[y]

    kk = {}   # window/section endpoints
    for q in rungs:
        qp = nxt[q]
        k_lo = (q + 1) // 6 + 1
        k_hi = (qp * qp - 1) // 6 - 1
        k_s = (q * q - 1) // 6 + 1
        kk[q] = (k_lo, k_hi, k_s)

    # ---- family definitions -------------------------------------------------
    fams = []           # (name, y, modulus, residue list)
    fams.append(("win", 3, 1, [0]))
    for y in (7, 11, 13, 17, 19):
        gs = [g for g in gears if g <= y]
        P = 1
        for g in gs:
            P *= g
        fams.append(("a%d" % y, y, P, [0]))
        Mt, T = crt_sets(gs, [teeth(g) for g in gs])
        assert Mt == P
        nb = sorted({(c + 1) % P for c in T} | {(c - 1) % P for c in T})
        fams.append(("b%d" % y, y, P, nb))
        Mo, O = crt_sets(gs, [[r for r in range(g) if r not in teeth(g)] for g in gs])
        assert Mo == P
        fams.append(("o%d" % y, y, P, O))

    rows_w, rows_s = [], []
    for name, y, M, R in fams:
        if M == 1:
            mask = np.ones(KMAX + 1, dtype=bool)
        else:
            lut = np.zeros(M, dtype=bool)
            lut[np.array(R, dtype=np.int64)] = True
            mask = lut[k % M]
        cn = np.cumsum(mask, dtype=np.int64)
        cs = np.cumsum(mask & twin, dtype=np.int64)
        dens = len(R) / M
        for q in rungs:
            k_lo, k_hi, k_s = kk[q]
            rate = cp[q] / cp[y]
            for tag, a, b, rows in (("w", k_lo, k_hi, rows_w), ("s", k_s, k_hi, rows_s)):
                n = int(cn[b] - cn[a - 1])
                sv = int(cs[b] - cs[a - 1])
                rows.append((name, y, q, b - a + 1, len(R), M, n, sv, rate,
                             (sv / (n * rate)) if n * rate > 0 else float("nan")))
        print("done", name, "|R|=%d" % len(R), "M=%d" % M, "dens=%.3e" % dens, flush=True)
        del mask, cn, cs

    for fn, rows in (("families_window.tsv", rows_w), ("families_section.tsv", rows_s)):
        with open(os.path.join(OUT, fn), "w") as f:
            f.write("fam\ty\tq\tspan\tnclass\tmod\tn\tsurv\trate\texcess\n")
            for r in rows:
                f.write("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.10g\t%.6g\n" % r)

    # ---- islands (section only, position depends on q) ----------------------
    with open(os.path.join(OUT, "islands.tsv"), "w") as f:
        f.write("fam\ty\tq\tspan\tnclass\tmod\tn\tsurv\trate\texcess\tk0mod35\n")
        for q in rungs:
            k_lo, k_hi, k_s = kk[q]
            k0 = (q * q - 1) // 6
            seg = np.arange(k_s, k_hi + 1, dtype=np.int64)
            off = (seg - k0) % 35
            sel = np.isin(off, ISLAND_OFFS)
            n = int(sel.sum())
            sv = int((twin[k_s:k_hi + 1] & sel).sum())
            rate = cp[q] / cp[7]
            ex = (sv / (n * rate)) if n * rate > 0 else float("nan")
            f.write("isl\t7\t%d\t%d\t4\t35\t%d\t%d\t%.10g\t%.6g\t%d\n"
                    % (q, k_hi - k_s + 1, n, sv, rate, ex, k0 % 35))
    print("islands done")


if __name__ == "__main__":
    main()
