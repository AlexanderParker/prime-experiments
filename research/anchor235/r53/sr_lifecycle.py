"""sr_lifecycle.py -- the life cycle of a size in the spectrum, m5..m31.

Reads the exact spectra produced by sr_identities.py (direct sieve, m5..m23) and by
sr_recursion.py / sr_rung31.py (the recursion, m29 and m31, both gated against the corpus).

Reports, over every rung:
  * monotonicity  m_{M+q'}(v) >= (q'-4) m_M(v)  and  Spec(M) subset Spec(M+q')
  * the survival coefficient c_{q'}(v) actually used at every size
  * the rarity r(v) = m(v) / median{m(w) : |w-v| <= 4, w coupled} at every machine
  * the first appearance of every size: the rung, and the smallest J of the merge that made it
  * the life cycle of the sizes 4, 6, 24, 36, 41 (twice the twin columns 2, 3, 12, 18; and the
    wrapped letter 41) across every machine

Writes results/sr_lifecycle.txt
"""
import os, sys, json
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
PR = [5, 7, 11, 13, 17, 19, 23, 29, 31]


def u_of(g):
    return pow(6, -1, g)


def c_local(q, v):
    u = u_of(q)
    s = {u % q, (-u) % q, (u - v) % q, ((-u) - v) % q}
    return q - len(s)


def role(q, v):
    if v % q == 0:
        return "pad"
    d = (2 * u_of(q)) % q
    if v % q in (d % q, (-d) % q):
        return "letter"
    return None


def load():
    spec, surv, mbj = {}, {}, {}
    for g in PR:
        f1 = os.path.join(OUT, f"spec_m{g}.json")
        f2 = os.path.join(OUT, f"spec_rec_m{g}.json")
        if os.path.exists(f1):
            spec[g] = {int(k): v for k, v in json.load(open(f1)).items()}
        if os.path.exists(f2):
            d = json.load(open(f2))
            if g not in spec:
                spec[g] = {int(k): v for k, v in d["m"].items()}
            surv[g] = {int(k): v for k, v in d["surv"].items()}
            mbj[g] = {int(J): {int(k): v for k, v in a.items()} for J, a in d["merge_by_J"].items()}
    return spec, surv, mbj


def main():
    spec, surv, mbj = load()
    lines = []
    W = lines.append

    # 1. monotonicity and spectrum inclusion
    W("## monotonicity and spectrum inclusion, rung by rung")
    bad_mono, bad_inc, ncell = [], [], 0
    for i in range(1, len(PR)):
        M, qp = PR[i - 1], PR[i]
        a, b = spec[M], spec[qp]
        for v, c in a.items():
            ncell += 1
            if b.get(v, 0) < (qp - 4) * c:
                bad_mono.append((M, qp, v, c, b.get(v, 0)))
            if b.get(v, 0) == 0:
                bad_inc.append((M, qp, v))
    W(f"  m_(M+q')(v) >= (q'-4) m_M(v): {ncell} cells, "
      f"{'0 exceptions' if not bad_mono else bad_mono}")
    W(f"  Spec(M) subset Spec(M+q'): {'0 exceptions' if not bad_inc else bad_inc}")

    # 2. the survival coefficient used at each rung, and its check against the split
    W("")
    W("## the survival coefficient c_q'(v) against the measured survival term")
    bad_c, nc = [], 0
    for i in range(1, len(PR)):
        M, qp = PR[i - 1], PR[i]
        if qp not in surv:
            continue
        for v, c in spec[M].items():
            nc += 1
            if surv[qp].get(v, 0) != c_local(qp, v) * c:
                bad_c.append((M, qp, v, surv[qp].get(v, 0), c_local(qp, v) * c))
        for v in surv[qp]:
            if v not in spec[M] and surv[qp][v] != 0:
                bad_c.append((M, qp, v, "phantom"))
    W(f"  survival_(M+q')(v) = c_q'(v) . m_M(v): {nc} cells, "
      f"{'0 exceptions' if not bad_c else bad_c}")

    # 3. rarity r(v) at every machine
    W("")
    W("## rarity r(v) = m(v) / median{m(w) : |w-v|<=4, w coupled}, at the uncoupled sizes")
    W("  machine | v | coupling gears | m(v) | r(v) | percentile of r among coupled sizes")
    rar = {}
    for i, y in enumerate(PR):
        gears = PR[: i + 1]
        m = spec[y]
        F = max(m)
        cpl = {v: [g for g in gears if role(g, v)] for v in range(1, F + 1)}
        rv = {}
        for v in range(2, F + 1):
            nb = [m.get(w, 0) for w in range(max(1, v - 4), min(F, v + 4) + 1)
                  if w != v and cpl[w]]
            if not nb:
                continue
            md = median(nb)
            if md > 0:
                rv[v] = m.get(v, 0) / md
        rar[y] = rv
        for v in range(2, F + 1):
            if not cpl[v] and v in rv:
                cvals = sorted(rv[w] for w in rv if cpl[w])
                pc = 100.0 * sum(1 for x in cvals if x <= rv[v]) / len(cvals)
                W(f"  m{y} | {v} | none | {m.get(v,0)} | {rv[v]:.4f} | {pc:.1f}")

    # 4. first appearance of every size
    W("")
    W("## first appearance of each size: rung, whether survival was available, smallest merge J")
    firstseen = {}
    for i, y in enumerate(PR):
        for v in spec[y]:
            firstseen.setdefault(v, y)
    rows = []
    for v in sorted(firstseen):
        y = firstseen[v]
        i = PR.index(y)
        prev = PR[i - 1] if i > 0 else None
        s = surv.get(y, {}).get(v, 0)
        Js = sorted(J for J, a in mbj.get(y, {}).items() if a.get(v, 0) > 0)
        rows.append(f"v={v} first at m{y} (survival {s}, merge J in {Js})")
    W("  " + "; ".join(rows))

    # 5. the life cycle of the marked sizes
    W("")
    W("## life cycle: m(v) at every machine, with the coefficient at each rung")
    for v in (4, 6, 24, 36, 41, 10, 20, 23, 25):
        cells = []
        for i, y in enumerate(PR):
            gears = PR[: i + 1]
            cg = [f"{g}:{role(g,v)}" for g in gears if role(g, v)]
            c = spec[y].get(v, 0)
            tag = "U" if not cg else ",".join(cg)
            F = max(spec[y])
            mark = "" if v <= F else "(>F)"
            rr = rar[y].get(v)
            cells.append(f"m{y}={c}{mark}[{tag}]" + (f"r={rr:.3g}" if rr is not None else ""))
        W(f"  v={v}: " + "  ".join(cells))
    W("")
    W("## step ratios m_(M+q')(v)/m_M(v) and the survival share, at the uncoupled sizes")
    for i in range(1, len(PR)):
        M, qp = PR[i - 1], PR[i]
        if qp not in surv:
            continue
        gears = PR[:i]
        F = max(spec[M])
        unc = [v for v in range(2, max(spec[qp]) + 1)
               if not any(role(g, v) for g in PR[: i + 1])]
        for v in unc:
            a, b = spec[M].get(v, 0), spec[qp].get(v, 0)
            if b == 0:
                continue
            sh = surv[qp].get(v, 0) / b
            W(f"  {M}->{qp} v={v} uncoupled: m {a} -> {b}  ratio "
              f"{(b/a if a else float('inf')):.1f}  c={c_local(qp,v)}  survival share {sh*100:.1f}%")
        # a coupled control at comparable size
        for v in unc:
            for w in (v - 1, v + 1):
                a, b = spec[M].get(w, 0), spec[qp].get(w, 0)
                if b and a:
                    sh = surv[qp].get(w, 0) / b
                    W(f"      control w={w} ({[f'{g}:{role(g,w)}' for g in PR[:i+1] if role(g,w)]}): "
                      f"m {a} -> {b}  ratio {b/a:.1f}  c={c_local(qp,w)}  survival share {sh*100:.1f}%")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "sr_lifecycle.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
