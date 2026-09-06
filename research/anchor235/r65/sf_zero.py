"""sf_zero.py -- is a located family's EMPTINESS decided by counting?

For each family and each rung the expected number of survivors is E(q) = n * rate * R, where
R = 0.7930 (the s = 2 handicap, e^{2 gamma}/4) on the section and the measured window value.
If survival is decided by counting alone the number of rungs with no survivor should be
sum_q exp(-E(q)) (Poisson), and the last empty rung should be where E crosses about 1.

Writes results/zeros.txt.
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
ORDER = ["win", "o7", "o11", "o13", "o17", "o19", "b7", "b11", "b13", "b17", "b19",
         "a7", "a11", "a13", "a17", "a19", "isl"]


def load(fn):
    rows = []
    with open(os.path.join(OUT, fn)) as f:
        f.readline()
        for ln in f:
            p = ln.split("\t")
            rows.append((p[0], int(p[2]), int(p[6]), int(p[7]), float(p[8])))
    return rows


def report(rows, tag, Rfun, out):
    byfam = {}
    for fam, q, n, sv, r in rows:
        byfam.setdefault(fam, []).append((q, n, sv, r))

    def w(s):
        print(s)
        out.append(s)

    w("")
    w("=== %s: observed empties against the Poisson prediction ===" % tag)
    w("%-5s %8s %8s %10s %10s %12s %12s" %
      ("fam", "obs 0s", "exp 0s", "last obs0", "last E<1", "E(4999)", "obs/exp"))
    for fam in ORDER:
        if fam not in byfam:
            continue
        rs = sorted(byfam[fam])
        obs = sum(1 for q, n, sv, r in rs if sv == 0)
        exp = sum(math.exp(-n * r * Rfun(q)) for q, n, sv, r in rs)
        lastobs = max([q for q, n, sv, r in rs if sv == 0], default=None)
        lastE = max([q for q, n, sv, r in rs if n * r * Rfun(q) < 1.0], default=None)
        E4999 = [n * r * Rfun(q) for q, n, sv, r in rs if q == 4999][0]
        w("%-5s %8d %8.1f %10s %10s %12.3g %12s" %
          (fam, obs, exp, lastobs if lastobs else "none", lastE if lastE else "none", E4999,
           ("%.2f" % (obs / exp)) if exp > 0.05 else "-"))


def main():
    out = []
    w = load("families_window.tsv")
    s = load("families_section.tsv") + load("islands.tsv")
    report(w, "WINDOW", lambda q: 0.92, out)
    report(s, "SECTION", lambda q: 0.7930, out)

    # the empty rungs of the island family and of a13/a17
    byfam = {}
    for fam, q, n, sv, r in s:
        byfam.setdefault(fam, []).append((q, n, sv, r))
    for fam in ("isl",):
        z = [q for q, n, sv, r in sorted(byfam[fam]) if sv == 0]
        out.append("")
        out.append("island family empty on the section at rungs: %s" % z)
        print(out[-1])
    byfamw = {}
    for fam, q, n, sv, r in w:
        byfamw.setdefault(fam, []).append((q, n, sv, r))
    for fam in ("a13", "a17", "b13", "b17", "b19", "a11"):
        z = [q for q, n, sv, r in sorted(byfamw[fam]) if sv == 0]
        out.append("%s empty in the WINDOW at %d rungs, last %s, first 12: %s"
                   % (fam, len(z), z[-1] if z else "-", z[:12]))
        print(out[-1])

    with open(os.path.join(OUT, "zeros.txt"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
