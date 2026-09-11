"""xm_words.py -- the FIRST REALISATION COLUMN of a local pattern, by enumerating its covers and
taking the least element of each cover's residue class (no scan of the period).

Pattern: OPEN = offsets that must be open, every other offset of [0, S] must be struck.
Gear g strikes offset o at column x iff (x + o) mod g is in {u_g, g - u_g}; so the gear's
action on the pattern is a function of rho_g = x mod g alone.  rho_g is ALLOWED iff it strikes
no offset of OPEN.  A COVER assigns to every struck offset a gear and a residue (consistent per
gear); the columns realising the cover form one class mod prod(assigned gears); the gears not
assigned are free within their allowed residues.  The first realisation of the pattern is

    min over covers  [ least y >= 0 :  y = x0 (mod M),  y mod g in A_g for every free gear g ]

found by an exact-cover search (branch on the uncovered offset with the fewest options, with the
capacity bound of r70/ol_pattern.py), and at every leaf a walk y = x0, x0 + M, ... bounded by
the best value so far.  Everything is exact integer arithmetic; a result is certified by
re-sieving the pattern at the returned column against every gear.

For a run of length l flanked by openings the pattern is OPEN = {0, l + 1}: the returned x is the
opening before the run, and the run starts at x + 1.

Validation (first_realisation.md P1): m23, OPEN = {0, 34} -> 12,694,428 (run at 12,694,429);
m29, OPEN = {0, 43} -> 200,906,185; m31, OPEN = {0, 58} -> 1,468,940,242 (xm_scan.py).
"""
import argparse
import json
import os
import sys
import time

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]


def gears_of(p):
    return [g for g in PRIMES if g <= p]


def u_of(g):
    return pow(6, -1, g)


def crt(residues):
    """residues: dict g -> r.  Returns (x0, M)."""
    x, M = 0, 1
    for g, r in residues.items():
        # solve x + M t = r (mod g)
        t = ((r - x) * pow(M, -1, g)) % g
        x += M * t
        M *= g
    return x % M, M


def setup(open_offsets, S, gears):
    opens = sorted(set(int(o) for o in open_offsets))
    openset = set(opens)
    closed = [o for o in range(S + 1) if o not in openset]
    cmask = 0
    for o in closed:
        cmask |= 1 << o
    per = {}      # g -> list of (rho, mask of closed offsets struck)
    allowed = {}  # g -> sorted allowed residues
    for g in gears:
        u = u_of(g)
        teeth = {u % g, (-u) % g}
        rows = []
        al = []
        for rho in range(g):
            if any(((rho + o) % g) in teeth for o in opens):
                continue
            al.append(rho)
            m = 0
            for o in closed:
                if ((rho + o) % g) in teeth:
                    m |= 1 << o
            rows.append((rho, m))
        if not rows:
            return None
        per[g] = rows
        allowed[g] = al
    return cmask, per, allowed, closed


def verify(x, open_offsets, S, gears):
    openset = set(int(o) for o in open_offsets)
    for o in range(S + 1):
        struck = False
        for g in gears:
            u = u_of(g)
            if ((x + o) % g) in (u % g, (-u) % g):
                struck = True
                break
        if struck == (o in openset):
            return False
    return True


def first_realisation(open_offsets, S, gears, node_budget=200_000_000, time_limit=None,
                      best_init=None, verbose=False):
    st = setup(open_offsets, S, gears)
    if st is None:
        return {"realised": False, "reason": "a gear cannot avoid OPEN"}
    cmask, per, allowed, closed = st
    cov = {g: {} for g in gears}
    cap = {}
    for g in gears:
        for rho, m in per[g]:
            mm = m
            while mm:
                b = mm & -mm
                cov[g].setdefault(b.bit_length() - 1, []).append((rho, m))
                mm ^= b
        cap[g] = max(bin(m).count("1") for _, m in per[g])
    allowed_sets = {g: set(allowed[g]) for g in gears}
    best = [best_init if best_init is not None else None]
    best_leaf = [None]
    nodes = [0]
    leaves = [0]
    t0 = time.time()
    assigned = {}

    def leaf():
        leaves[0] += 1
        x0, M = crt(assigned)
        free = [g for g in gears if g not in assigned]
        y = x0
        while True:
            if best[0] is not None and y >= best[0]:
                return
            if all((y % g) in allowed_sets[g] for g in free):
                best[0] = y
                best_leaf[0] = {"x": y, "assigned": dict(assigned), "free": free, "M": M, "x0": x0}
                if verbose:
                    print(f"    leaf: x = {y:,d}  assigned {sorted(assigned)}  M = {M:,d}  "
                          f"x0 = {x0:,d}  t = {time.time() - t0:.1f}s", flush=True)
                return
            y += M

    def rec(unc, rem):
        if unc == 0:
            leaf()
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise RuntimeError("node budget exceeded")
        if time_limit is not None and time.time() - t0 > time_limit:
            raise RuntimeError("time limit exceeded")
        need = bin(unc).count("1")
        tot = 0
        for g in rem:
            b = 0
            for _, m in per[g]:
                v = bin(m & unc).count("1")
                if v > b:
                    b = v
            tot += b
            if tot >= need:
                break
        if tot < need:
            return
        bestp, bestopts = None, None
        u = unc
        while u:
            b = u & -u
            pos = b.bit_length() - 1
            u ^= b
            opts = [(g, rho, m) for g in rem for (rho, m) in cov[g].get(pos, ())]
            if not opts:
                return
            if bestopts is None or len(opts) < len(bestopts):
                bestp, bestopts = pos, opts
                if len(opts) <= 1:
                    break
        for g, rho, m in bestopts:
            assigned[g] = rho
            rec(unc & ~m, tuple(x for x in rem if x != g))
            del assigned[g]

    status = "complete"
    try:
        rec(cmask, tuple(gears))
    except RuntimeError as e:
        status = str(e)
    out = {"realised": best[0] is not None, "x": best[0], "status": status, "nodes": nodes[0],
           "leaves": leaves[0], "secs": time.time() - t0}
    if best_leaf[0] is not None:
        bl = best_leaf[0]
        out["assigned"] = {str(g): r for g, r in sorted(bl["assigned"].items())}
        out["free"] = bl["free"]
        out["M"] = bl["M"]
        out["x0"] = bl["x0"]
        out["verified"] = verify(best[0], open_offsets, S, gears)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("p", type=int)
    ap.add_argument("--open", default=None, help="comma list of open offsets, e.g. 0,34")
    ap.add_argument("--word", default=None, help="comma list of gaps, e.g. 21,14,41,15")
    ap.add_argument("--S", type=int, default=None)
    ap.add_argument("--budget", type=int, default=200_000_000)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    gears = gears_of(a.p)
    if a.word:
        gaps = [int(v) for v in a.word.split(",")]
        offs = [0]
        for v in gaps:
            offs.append(offs[-1] + v)
        S = offs[-1]
        desc = f"word {gaps}"
    else:
        offs = [int(v) for v in a.open.split(",")]
        S = a.S if a.S is not None else max(offs)
        desc = f"open {offs}"
    print(f"m{a.p} {desc} S = {S}", flush=True)
    r = first_realisation(offs, S, gears, node_budget=a.budget, time_limit=a.time_limit,
                          verbose=a.verbose)
    r.update({"p": a.p, "open": offs, "S": S, "desc": desc, "tag": a.tag})
    print(json.dumps(r), flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results", "xm_words.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
