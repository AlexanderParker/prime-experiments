"""fm_classes.py -- P6 and the division-of-work census on real sections.

For each real section (finer at p = 11..53; construction sections base 3 link 1-2, base 5 link 1-2, base 7 link 1,
base 13 link 1), with engine {5..q}:
  (i) T4 check: the columns left open by the class -1 gears alone have L in P_-^{>q} . M_+ and R in M_+;
      those left open by the class +1 gears alone have L in M_-^{odd} . ({1} u P_+^{>q}) and R in M_-^{even} . ({1} u P_+^{>q}).
  (ii) census: per side, members struck by class -1 gears only / class +1 only / both / none; per column,
      open after class -1 gears alone, open after class +1 gears alone, open after all (twins).
Output: results/classes.json, results/classes.log"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from fm_common import primes_upto, spf_table

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


def factor(n, spf):
    fs = []
    while n > 1:
        p = int(spf[n])
        fs.append(p)
        n //= p
    return fs


def sections():
    from sympy import nextprime, prevprime
    out = []
    for p in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
        pn = nextprime(p)
        out.append(dict(name=f"finer p={p}", lo=p * p, hi=pn * pn, q=p))
    for base, links in ((3, 2), (5, 2), (7, 1), (13, 1)):
        c = base
        for k in range(links):
            pk = nextprime(c - 1)
            cn = pk * pk
            pk1 = nextprime(cn - 1)
            q = prevprime(pk1)
            out.append(dict(name=f"base {base} link {k + 1}", lo=cn, hi=pk1 * pk1, q=int(q)))
            c = cn
    return out


def main():
    secs = sections()
    N = max(s["hi"] for s in secs)
    spf = spf_table(N)
    log = open(os.path.join(RES, "classes.log"), "w")
    results = []
    for s in secs:
        lo, hi, q = s["lo"], s["hi"], s["q"]
        a = (lo - 1) // 6
        b = (hi - 1) // 6
        js = list(range(a + 1, b))
        exc = []
        cnt = dict(L_minus_only=0, L_plus_only=0, L_both=0, L_none=0,
                   R_minus_only=0, R_plus_only=0, R_both=0, R_none=0,
                   col_open_after_minus=0, col_open_after_plus=0, col_open_all=0,
                   col_struck_left_only=0, col_struck_right_only=0, col_struck_both=0)
        for j in js:
            L, R = 6 * j - 1, 6 * j + 1
            fL, fR = factor(L, spf), factor(R, spf)
            # strikers by class (gears <= q)
            Lm = any(p <= q and p % 6 == 5 for p in fL)
            Lp = any(p <= q and p % 6 == 1 for p in fL)
            Rm = any(p <= q and p % 6 == 5 for p in fR)
            Rp = any(p <= q and p % 6 == 1 for p in fR)
            for side, m, pl in (("L", Lm, Lp), ("R", Rm, Rp)):
                key = f"{side}_" + ("both" if m and pl else "minus_only" if m else "plus_only" if pl else "none")
                cnt[key] += 1
            openL = not (Lm or Lp)
            openR = not (Rm or Rp)
            if openL and openR:
                cnt["col_open_all"] += 1
            elif openL:
                cnt["col_struck_right_only"] += 1
            elif openR:
                cnt["col_struck_left_only"] += 1
            else:
                cnt["col_struck_both"] += 1
            # T4 check, class -1 gears alone
            if not Lm and not Rm:
                cnt["col_open_after_minus"] += 1
                # L must be t . u with t a class -1 prime > q and u in M_+ (all class +1 factors)
                t = [p for p in fL if p % 6 == 5]
                ok_L = len(t) == 1 and t[0] > q and all(p % 6 == 1 for p in fL if p != t[0])
                ok_R = all(p % 6 == 1 for p in fR)
                if not (ok_L and ok_R):
                    exc.append(("minus", j, fL, fR))
            if not Lp and not Rp:
                cnt["col_open_after_plus"] += 1
                # L in M_-^{odd} . ({1} u P_+^{>q}); R in M_-^{even} . ({1} u P_+^{>q})
                bigp_L = [p for p in fL if p % 6 == 1]
                bigp_R = [p for p in fR if p % 6 == 1]
                ok_L = all(p > q for p in bigp_L) and len(bigp_L) <= 1
                ok_R = all(p > q for p in bigp_R) and len(bigp_R) <= 1
                if not (ok_L and ok_R):
                    exc.append(("plus", j, fL, fR))
        rec = dict(section=s["name"], lo=lo, hi=hi, q=q, columns=len(js), counts=cnt, T4_exceptions=exc[:5],
                   T4_exception_count=len(exc))
        results.append(rec)
        line = json.dumps(rec)
        print(line)
        log.write(line + "\n")
    with open(os.path.join(RES, "classes.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("total T4 exceptions:", sum(r["T4_exception_count"] for r in results))


if __name__ == "__main__":
    main()
