"""ol_witness.py -- a column that certifies a window, checked by direct arithmetic.

The pattern solver decides membership in D_k(M) by a phase assignment t_g per gear.  Because
t_g = (u_g - x) mod g, that assignment IS a column x mod P of the machine, recovered by CRT:
x = u_g - t_g (mod g) for every gear.  This module returns that column and then re-verifies the
window from scratch -- for every offset o of the span it tests o + x against every gear's two
teeth, with no reference to the solver -- so a YES verdict comes with a certificate that can be
checked by hand.

Usage: uv run python research/anchor235/r70/ol_witness.py "g1,g2,g3" [top_gear]
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")

from ol_pattern import _setup, d_of, u_of  # noqa: E402


def witness(offsets, gears):
    """A phase vector (and hence a column) realising the window, or None."""
    cmask, per, closed = _setup(offsets, gears)
    if cmask is None:
        return None
    cov = {}
    for g in gears:
        cg = {}
        for m in per[g]:
            mm = m
            while mm:
                b = mm & -mm
                cg.setdefault(b.bit_length() - 1, []).append(m)
                mm ^= b
        cov[g] = cg
    # a phase mask has to be mapped back to its phase: keep the pairing
    phase_of = {}
    for g in gears:
        d = d_of(g)
        bad = set()
        for o in offsets:
            bad.add(int(o) % g)
            bad.add((int(o) - d) % g)
        i = 0
        for t in range(g):
            if t in bad:
                continue
            phase_of[(g, per[g][i])] = t
            i += 1

    def rec(unc, rem, acc):
        if unc == 0:
            return acc
        need = bin(unc).count("1")
        tot = 0
        for g in rem:
            b = max(bin(m & unc).count("1") for m in per[g])
            tot += b
            if tot >= need:
                break
        if tot < need:
            return None
        best = None
        u = unc
        while u:
            b = u & -u
            p = b.bit_length() - 1
            u ^= b
            opts = [(g, m) for g in rem for m in cov[g].get(p, ())]
            if not opts:
                return None
            if best is None or len(opts) < len(best):
                best = opts
                if len(opts) <= 1:
                    break
        for g, m in best:
            r = rec(unc & ~m, tuple(x for x in rem if x != g), acc + [(g, phase_of[(g, m)])])
            if r is not None:
                return r
        return None

    return rec(cmask, tuple(gears), [])


def column_of(assign, gears):
    """CRT the phase vector back to a column x of the machine.  The solver's struck classes are
    {t_g, t_g + d_g} and the machine's are {-u_g - x, u_g - x}, so t_g = -u_g - x, i.e.
    x = -(u_g + t_g) mod g."""
    got = dict(assign)
    x, mod = 0, 1
    for g in gears:
        r = (-(u_of(g) + got[g])) % g
        # solve x = x0 (mod mod), x = r (mod g)
        inv = pow(mod % g, -1, g)
        k = ((r - x) * inv) % g
        x = x + mod * k
        mod *= g
    return x, mod


def verify(x, gaps, gears):
    """Check the window at column x from scratch: the openings are exactly where they should be."""
    off = list(np.concatenate([[0], np.cumsum(np.asarray(gaps, dtype=np.int64))]))
    S = int(off[-1])
    opens = set(int(o) for o in off)

    def struck(k):
        for g in gears:
            u = u_of(g)
            if k % g == u % g or k % g == (-u) % g:
                return True
        return False

    ok = True
    for o in range(S + 1):
        want_open = o in opens
        if struck(x + o) == want_open:
            ok = False
    return ok


def main():
    gaps = [int(v) for v in sys.argv[1].split(",")]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    gears = [g for g in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37] if g <= top]
    off = np.concatenate([[0], np.cumsum(gaps)])
    a = witness(off, gears)
    if a is None:
        print(f"{gaps}: NOT realised in m{top}")
        return
    # gears not needed by the cover still must not strike the openings: pick any legal phase
    used = dict(a)
    cmask, per, closed = _setup(off, gears)
    full = []
    for g in gears:
        if g in used:
            full.append((g, used[g]))
        else:
            d = d_of(g)
            bad = set()
            for o in off:
                bad.add(int(o) % g)
                bad.add((int(o) - d) % g)
            full.append((g, next(t for t in range(g) if t not in bad)))
    x, P = column_of(full, gears)
    print(f"{gaps}: realised in m{top}; phases {full}")
    print(f"  witness column x = {x} (mod P = {P}); direct check: "
          f"{'PASS' if verify(x, gaps, gears) else 'FAIL'}")


if __name__ == "__main__":
    main()
