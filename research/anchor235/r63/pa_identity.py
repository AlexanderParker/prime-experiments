"""pa_identity.py -- the letter in tooth units: the identity a_L = 2(q' + eps) u_g (mod g), its
universal form v = 3v d_g, the tooth-unit vector mu_g(v) = v d_g^{-1}, and the divisor forms of
Pad and Leg for the three letters a_L, b_L, q'.

    eps  = +1 if q' = 5 (mod 6) else -1        6 u_{q'} = q' + eps,  a_L = 2 u_{q'} = d_{q'}
    d_g  = 2 u_g = 3^{-1} (mod g)              so  v = (3v) d_g (mod g)  for EVERY v   (I0)
    mu_g(v) = v * d_g^{-1} = 3v (mod g)        the tooth-unit multiple; constant over gears

Derived divisor forms, checked here against the direct chain-law test v = 0, +-d_g (mod g):

    Pad(a_L) n M = { g : g | q' + eps }        Leg(a_L) n M = { g : g | q' + 2 eps }
    Pad(b_L) n M = { g : g | 2q' - eps }       Leg(b_L) n M = { g : g | q' - eps }
    Pad(q')  n M = {}                          Leg(q')  n M = { g : g | 3q' - 1 or 3q' + 1 }

Outputs results/pa_identity.txt / .json.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "r56"))
from mf_core import u_of                                  # noqa: E402

# u_of(g) = 6^{-1} (mod g).  The tooth rule normalises the tooth to u_g = min(6^{-1}, g - 6^{-1})
# so that 6 u_g = g -+ 1; the TEETH SET {+-u_g} = {+-6^{-1}} is the same either way, and every
# statement below is written with w_g := 6^{-1} so that no sign bookkeeping is needed:
#     teeth of g are  +-w_g,   d_g := 2 w_g = 3^{-1} (mod g),   6 * (any distance v) = 6v.


def w_of(g):
    return u_of(g)


def unorm(g):
    w = u_of(g)
    return min(w, g - w)

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

GEARS = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# rung q' -> machine M = every gear below q'
RUNGS = [7, 11, 13, 17, 19, 23, 29, 31, 37]


def machine(q):
    return [g for g in GEARS if g < q]


def eps_of(q):
    return 1 if q % 6 == 5 else -1


def factors5(n):
    n = abs(n)
    out, d = set(), 2
    while d * d <= n:
        while n % d == 0:
            out.add(d)
            n //= d
        d += 1
    if n > 1:
        out.add(n)
    return {p for p in out if p >= 5}


def pad(v):
    return factors5(v)


def leg(v):
    return factors5(3 * v - 1) | factors5(3 * v + 1)


def chain_leg_direct(v, g):
    """the chain law test: g can strike both ends of a v-gap in one copy."""
    d = (2 * w_of(g)) % g
    return v % g in (0, d, (-d) % g)


def main():
    t0 = time.time()
    L = []
    W = L.append
    res = {}
    W("=== THE LETTER IN TOOTH UNITS: identity, universal form, divisor forms ===")

    # ---------- P1a: the identity at every rung x gear ----------
    W("\n--- (I1)  a_L = (q'+eps) d_g = 2(q'+eps) u_g  (mod g), every rung x gear ---")
    W("rung q' | eps | u_q' | a_L | 3a_L | M | checks | exceptions")
    n1 = bad1 = 0
    rows = []
    for q in RUNGS:
        e = eps_of(q)
        u = unorm(q)
        aL = 2 * u
        assert 6 * u == q + e, (q, u, e)
        assert 3 * aL == q + e, (q, aL, e)
        M = machine(q)
        exc = []
        for g in M:
            d = (2 * w_of(g)) % g
            lhs = aL % g
            r1 = ((q + e) * d) % g
            r2 = (2 * (q + e) * w_of(g)) % g
            n1 += 1
            if lhs != r1 or lhs != r2:
                bad1 += 1
                exc.append((g, lhs, r1, r2))
        W(f"{q:>7} | {e:+d} | {u:>4} | {aL:>3} | {3*aL:>4} | {M} | {len(M)} | "
          f"{exc if exc else 'none'}")
        rows.append(dict(q=q, eps=e, u=u, aL=aL, M=M, exc=exc))
    W(f"  (I1): {n1} (rung, gear) checks, {bad1} exceptions")

    # ---------- P1b: the universal form and the tooth-unit vector ----------
    n0 = bad0 = nmu = badmu = 0
    for g in GEARS:
        d = (2 * w_of(g)) % g
        dinv = pow(d, -1, g)
        for v in range(1, 121):
            n0 += 1
            if v % g != (3 * v * d) % g:
                bad0 += 1
            nmu += 1
            if (v * dinv) % g != (3 * v) % g:
                badmu += 1
    W(f"\n--- (I0)  v = (3v) d_g (mod g) for every v <= 120 and gear <= 37: "
      f"{n0} cells, {bad0} exceptions")
    W(f"--- mu_g(v) = v d_g^{{-1}} = 3v (mod g): {nmu} cells, {badmu} exceptions "
      f"-- the tooth-unit multiple of a distance is the SAME integer 3v at every gear")

    # ---------- the four forbidden classes of the near end ----------
    W("\n--- the near end's forbidden tooth-unit classes lam_g = 6x (mod g) ---")
    W("both ends of an a_L-gap open  <=>  lam_g not in {1, -1, 1-2(q'+eps), -1-2(q'+eps)} (mod g)")
    W("rung q' | gear | 2(q'+eps) mod g | forbidden classes | c_g(a_L) = |set|")
    cg = {}
    for q in RUNGS:
        e = eps_of(q)
        aL = 2 * unorm(q)
        for g in machine(q):
            sh = (2 * (q + e)) % g
            F = {1 % g, (-1) % g, (1 - sh) % g, (-1 - sh) % g}
            cg[(q, g)] = len(F)
            W(f"{q:>7} | {g:>4} | {sh:>15} | {sorted(F)} | {len(F)}")
        # cross-check against c_p(v) = |T_p u (T_p - v)|
        for g in machine(q):
            u = w_of(g)
            T = {u % g, (-u) % g}
            c = len(T | {(t - aL) % g for t in T})
            assert c == cg[(q, g)], (q, g, c, cg[(q, g)])
    W("  cross-checked against c_p(a_L) = |T_p u (T_p - a_L)|: 0 mismatches")

    # ---------- D1 / D2 / the other letters ----------
    W("\n--- the divisor forms of Pad and Leg for the three letters (D1, D2, P8) ---")
    W("rung q' | letter | value | 3*letter | Pad n M | predicted | Leg n M | predicted | ok")
    ndiv = baddiv = 0
    letrows = []
    for q in RUNGS:
        e = eps_of(q)
        M = set(machine(q))
        aL = 2 * unorm(q)
        bL = q - aL
        for name, v, padpred, legpred in (
                ("a_L", aL, factors5(q + e), factors5(q + 2 * e)),
                ("b_L", bL, factors5(2 * q - e), factors5(q - e)),
                ("q'", q, factors5(q), factors5(3 * q - 1) | factors5(3 * q + 1))):
            P = pad(v) & M
            Lg = leg(v) & M
            Pp = padpred & M
            Lp = legpred & M
            ok = (P == Pp and Lg == Lp)
            ndiv += 1
            if not ok:
                baddiv += 1
            # and the direct chain-law test
            direct = {g for g in M if chain_leg_direct(v, g)}
            assert direct == (P | Lg), (q, name, direct, P, Lg)
            W(f"{q:>7} | {name:<3} | {v:>4} | {3*v:>5} | {sorted(P) or '-'} | {sorted(Pp) or '-'}"
              f" | {sorted(Lg) or '-'} | {sorted(Lp) or '-'} | {'yes' if ok else 'NO'}")
            letrows.append(dict(q=q, name=name, v=v, pad=sorted(P), padpred=sorted(Pp),
                                leg=sorted(Lg), legpred=sorted(Lp), ok=ok))
    W(f"  divisor forms: {ndiv} (rung, letter) cells, {baddiv} exceptions; the direct chain-law "
      f"test agrees with Pad u Leg at every cell")

    # ---------- the twin rungs (P7a) ----------
    W("\n--- twin rungs: the partner p = q' - 2 in M (file 02(e): u_p = u_{q'}) ---")
    W("rung q' | p | u_p = u_q'? | d_p | a_L | 6a_L mod p | forbidden lam_p | c_p(a_L)")
    twin = []
    for q in RUNGS:
        p = q - 2
        if p not in machine(q):
            continue
        e = eps_of(q)
        aL = 2 * unorm(q)
        sh = (2 * (q + e)) % p
        Fset = {1 % p, (-1) % p, (1 - sh) % p, (-1 - sh) % p}
        W(f"{q:>7} | {p:>3} | {unorm(p) == unorm(q)} | {2*unorm(p):>3} | {aL:>3} | {sh:>10} | "
          f"{sorted(Fset)} | {len(Fset)}")
        twin.append(dict(q=q, p=p, same_u=bool(unorm(p) == unorm(q)), shift=sh, c=len(Fset)))
    W("  at every twin rung the far end sits exactly 2 tooth units beyond the near end at the")
    W("  partner, so the partner sees the letter gap as its own tooth jump and only 3 classes")
    W("  are forbidden.")

    res = dict(rungs=rows, letters=letrows, twin=twin,
               n_I1=n1, bad_I1=bad1, n_I0=n0, bad_I0=bad0, n_mu=nmu, bad_mu=badmu,
               n_div=ndiv, bad_div=baddiv)
    json.dump(res, open(os.path.join(OUT, "pa_identity.json"), "w"))
    txt = "\n".join(L)
    open(os.path.join(OUT, "pa_identity.txt"), "w").write(txt)
    print(f"wrote {OUT}/pa_identity.txt ({len(txt)} chars, {time.time()-t0:.1f}s)")
    print(f"(I1) {n1} checks {bad1} bad | (I0) {n0} checks {bad0} bad | "
          f"mu {nmu} checks {badmu} bad | divisor forms {ndiv} cells {baddiv} bad")


if __name__ == "__main__":
    main()
