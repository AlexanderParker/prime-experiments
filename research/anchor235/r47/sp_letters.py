"""Item 2, following the clue: WHICH gears are the movable ones, not how many.

The move lemma (2g.i.a, cited): a gear's strikes survive being recoloured iff v = 0 (mod g)
(padded) or v = +-d_g (mod g) (a letter of g).  2g.i.a refuted |Leg(v)| -- the COUNT of letter
gears -- as the reason the real teeth glue better; it is exactly the family mean.  This script
asks instead WHICH gears they are.

For the REAL teeth d_g = 2 u_g satisfies 3 d_g = 1 (mod g), so

    v is a letter of g   <=>   v = +- d_g (mod g)   <=>   3v = +-1 (mod g)   <=>   g | 3v -+ 1,

i.e. Leg_real(v) is exactly the set of gears dividing 3v-1 or 3v+1.  For v in 6..12 those two
integers lie in 17..37, so the letter gears of a small middle are the PRIME FACTORS of two
specific numbers just above 3v -- large gears whenever 3v +- 1 is prime.  For a random
symmetric-tooth member d_g = 2 w_g is arbitrary, so v is a letter of g with probability about
2/g -- most often at the SMALL gears.  Same expected count, opposite size distribution.

Why size is what matters: moving gear g abandons the flank columns it holds, roughly 2 n / g of
them on a flank of length n, so the MOVE COST of the letter set is measured by

    W(v) := sum_{g in Leg(v)} 2/g          (small = the movable gears are cheap to move).

This script computes Leg_real, W_real and the counterfactual distribution over random
separations, and reports the real machine's percentile in W.
"""
import sys, os, random
from sp_core import gears_of, us_of, sieve, gap_stats, attaining_runs

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def d_real(g):
    return (2 * pow(6, -1, g)) % g


def leg_real(gears, v):
    d = [d_real(g) for g in gears]
    return [g for g, dd in zip(gears, d) if v % g in (dd, (-dd) % g) and v % g != 0]


def leg_div(gears, v):
    return [g for g in gears if (3 * v - 1) % g == 0 or (3 * v + 1) % g == 0]


def pad(gears, v):
    return [g for g in gears if v % g == 0]


def main(out, tops=(17, 19, 23, 29, 31), ndraw=5000, seed=470470):
    # (a) the divisor identity
    bad = 0
    G200 = [p for p in range(5, 200) if all(p % k for k in range(2, int(p ** .5) + 1))]
    for v in range(1, 400):
        a = sorted(g for g in G200 if v % g in (d_real(g), (-d_real(g)) % g) and v % g)
        b = sorted(g for g in G200 if ((3 * v - 1) % g == 0 or (3 * v + 1) % g == 0)
                   and v % g)
        if a != b:
            bad += 1
    out.write(f"IDENTITY  Leg_real(v) = {{g : g | 3v-1 or g | 3v+1}} (minus the pads): "
              f"{400-bad}/400 values of v = 1..399, gears 5..199\n")

    rng = random.Random(seed)
    for top in tops:
        gears = gears_of(top)
        us = us_of(gears)
        out.write(f"\n===== m{top}  gears {gears}  real separations "
                  f"{[min(d_real(g), g-d_real(g)) for g in gears]}\n")
        for v in range(6, 14):
            Lg = leg_real(gears, v)
            Pd = pad(gears, v)
            Wr = sum(2 / g for g in Lg)
            # counterfactual: random separations d_g = 2 w_g, w uniform in 1..(g-1)/2
            cnt = []
            ws = []
            for _ in range(ndraw):
                ds = [(2 * rng.randrange(1, (g + 1) // 2)) % g for g in gears]
                lg = [g for g, dd in zip(gears, ds)
                      if v % g in (dd, (-dd) % g) and v % g != 0]
                cnt.append(len(lg))
                ws.append(sum(2 / g for g in lg))
            pc_n = sum(1 for c in cnt if c < len(Lg)) / ndraw
            pc_w = sum(1 for w in ws if w < Wr) / ndraw
            out.write(f"  v={v:2d}: 3v-1={3*v-1} 3v+1={3*v+1}; Leg={Lg} Pad={Pd}; "
                      f"|Leg|={len(Lg)} (family mean {sum(cnt)/ndraw:.2f}, "
                      f"P(fam < real)={pc_n:.2f}); "
                      f"W={Wr:.4f} (family mean {sum(ws)/ndraw:.4f}, "
                      f"P(fam < real)={pc_w:.2f})\n")
        out.flush()


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else None
    o = open(dest, 'w') if dest else sys.stdout
    main(o)
    if dest:
        o.close()
