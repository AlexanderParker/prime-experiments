"""sr_residual.py -- how much of the measured depletion is the pair count, and how much is left.

r51 measured the depletion of the uncoupled sizes as a factor of 12-128 against the raw
multiplicities of neighbouring sizes.  Part of that is an exact identity: the number of OPEN
PAIRS at distance v is A(v) = prod_q c_q(v), and c_q(v) is q-4 exactly when q does not couple v.
So the fair comparison is rho(v) = m(v)/A(v), the fraction of open pairs at distance v that are
adjacent.  This script measures the residual depletion of rho after a local log-linear fit, and
compares it with the leave-one-out residual of the COUPLED sizes, which is the ordinary scatter.

Writes results/sr_residual.txt
"""
import os, json, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
PR = [5, 7, 11, 13, 17, 19, 23, 29, 31]
HALF = 6


def u_of(g):
    return pow(6, -1, g)


def c_local(q, v):
    u = u_of(q)
    return q - len({u % q, (-u) % q, (u - v) % q, ((-u) - v) % q})


def role(q, v):
    if v % q == 0:
        return "pad"
    d = (2 * u_of(q)) % q
    return "letter" if v % q in (d % q, (-d) % q) else None


def load(y):
    f1 = os.path.join(OUT, f"spec_m{y}.json")
    f2 = os.path.join(OUT, f"spec_rec_m{y}.json")
    if os.path.exists(f1):
        return {int(k): v for k, v in json.load(open(f1)).items()}
    return {int(k): v for k, v in json.load(open(f2))["m"].items()}


def predict(pts, v):
    """log-linear least squares on (w, log rho) excluding w = v; return predicted rho."""
    xs = np.array([w for w, _ in pts], dtype=float)
    ys = np.array([math.log(r) for _, r in pts], dtype=float)
    if xs.size < 3:
        return None
    a, b = np.polyfit(xs, ys, 1)
    return math.exp(a * v + b)


def main():
    lines = []
    W = lines.append
    for y in (19, 23, 29, 31):
        gears = [g for g in PR if g <= y]
        m = load(y)
        F = max(m)
        rho = {}
        for v in range(1, F + 1):
            A = 1
            for g in gears:
                A *= c_local(g, v)
            rho[v] = (m.get(v, 0) / A, A)
        cpl = {v: bool([g for g in gears if role(g, v)]) for v in range(1, F + 1)}
        W(f"=== m{y}   F={F}")
        W("  rho(1) = %.6f, rho(2) = %.6f  (both exactly 1 iff every open pair at distance 1 or 2 "
          "is a gap)" % (rho[1][0], rho[2][0]))
        # control: leave-one-out residual of every coupled size with rho > 0
        res = {}
        for v in range(3, F + 1):
            pts = [(w, rho[w][0]) for w in range(max(3, v - HALF), min(F, v + HALF) + 1)
                   if w != v and cpl[w] and rho[w][0] > 0]
            p = predict(pts, v)
            if p and p > 0:
                res[v] = rho[v][0] / p
        ctrl = sorted(res[v] for v in res if cpl[v])
        gm = math.exp(sum(math.log(x) for x in ctrl if x > 0) / max(1, len([x for x in ctrl if x > 0])))
        gsd = math.exp(np.std([math.log(x) for x in ctrl if x > 0]))
        W(f"  coupled control: {len(ctrl)} sizes, residual ratio geometric mean {gm:.3f}, "
          f"geometric sd {gsd:.3f}, range [{min(ctrl):.3f}, {max(ctrl):.3f}]")
        for v in sorted(res):
            if not cpl[v]:
                pc = 100.0 * sum(1 for x in ctrl if x <= res[v]) / len(ctrl)
                raw = None
                nb = [m.get(w, 0) for w in range(max(1, v - 4), min(F, v + 4) + 1)
                      if w != v and cpl[w]]
                if nb:
                    md = float(np.median(nb))
                    raw = m.get(v, 0) / md if md else None
                W(f"  UNCOUPLED v={v}: m={m.get(v,0)}  A={rho[v][1]}  rho={rho[v][0]:.4e}  "
                  f"raw depletion r={raw:.4f} (factor {1/raw:.1f}) " % () if False else
                  f"  UNCOUPLED v={v}: m={m.get(v,0)}  A={rho[v][1]}  rho={rho[v][0]:.4e}  "
                  f"raw r={raw:.4g}" + (f" (raw factor {1/raw:.1f})" if raw else "") +
                  f"  residual after A: {res[v]:.4g}" +
                  (f" (residual factor {1/res[v]:.1f})" if res[v] > 0 else " (absent)") +
                  f"  percentile among coupled {pc:.1f}")
        W("")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_residual.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
