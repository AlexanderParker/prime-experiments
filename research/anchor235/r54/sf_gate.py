"""sf_gate.py - the adversarial gate for the sub-machine fibre engine.

Branch R2.c.ii.  Two tests:

  (a) VALIDITY.  r50's MILP witnesses (research/anchor235/r50/results/arc_milp_K*.txt) give,
      for each K <= 12, an interval of A(K) - 1 columns that K primes with two classes each
      DO cover, with the explicit strike sets.  The engine says eta < 1 implies not covered,
      so on every witness the exact eta_SF must be >= 1, at every sub-machine cut.  Any
      witness with eta_SF < 1 refutes the engine or the implementation.

  (b) REACH.  L*_SF(K) = the shortest interval the unconditional SF-CAP envelope can rule
      out for the K smallest gears, against A(K) (consistency) and W_{K+1} = (p_{K+1}^2-1)/6
      (the open adversarial lemma A(K) < W_{K+1}, which r51's envelope reached to K = 10).

Usage: uv run python research/anchor235/r54/sf_gate.py
"""
import json
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
R50 = os.path.abspath(os.path.join(HERE, "..", "r50", "results"))
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:: i] = bytearray(len(sieve[i * i:: i]))
    return [i for i in range(n + 1) if sieve[i]]


PR = primes_upto(200000)
PSET = set(PR)
GEARS_ALL = [p for p in PR if p >= 5]

A_K = {1: 2, 2: 5, 3: 7, 4: 16, 5: 22, 6: 28, 7: 37, 8: 45, 9: 68, 10: 88, 11: 101, 12: 115}


def modulus_of(kind, key):
    if kind == "p":
        return key
    if kind == "d":
        for c in (3 * key - 1, 3 * key + 1):
            if c >= 5 and c in PSET:
                return c
    return None


def load_witnesses():
    """the maximal covered L for each K, with the explicit strike sets"""
    out = {}
    for K in range(4, 13):
        f = os.path.join(R50, "arc_milp_K%d.txt" % K)
        if not os.path.exists(f):
            continue
        best = None
        lines = open(f).read().split("\n")
        for i, ln in enumerate(lines):
            m = re.match(r"^K=%d L=(\d+): cover" % K, ln)
            if m and i + 1 < len(lines) and "witness:" in lines[i + 1]:
                L = int(m.group(1))
                w = json.loads(lines[i + 1].split("witness:", 1)[1].strip())
                if best is None or L > best[0]:
                    best = (L, w)
        if best:
            out[K] = best
    return out


def eta_exact(L, items, t):
    """exact sub-machine fibre eta on [0,L) with explicit strike sets.
    items = list of (modulus, sorted positions).  Cut t = the t smallest moduli."""
    items = sorted(items, key=lambda x: x[0])
    Qs = 1
    for g, _ in items[:t]:
        Qs *= g
    if Qs > L:
        return None
    k = np.arange(L, dtype=np.int64)
    w = np.full(L, 1.0 / L)
    Qlt = 1
    eta = 0.0
    for i, (g, pos) in enumerate(items):
        mod = Qlt if i < t else Qs
        pid = k % mod if mod > 1 else np.zeros(L, dtype=np.int64)
        npart = mod
        struck = np.zeros(L, dtype=bool)
        struck[np.array(pos, dtype=np.int64)] = True
        tot = np.bincount(pid, weights=w, minlength=npart)
        num = np.bincount(pid[struck], weights=w[struck], minlength=npart)
        nz = tot > 0
        a = np.zeros(npart)
        a[nz] = num[nz] / tot[nz]
        eta += float((tot * a * a).sum())
        af = a[pid]
        fac_on = np.where(af > 0, np.maximum(0.0, 2.0 - 1.0 / np.where(af > 0, af, 1.0)), 0.0)
        fac_off = np.minimum(1.0 / np.maximum(1e-300, 1.0 - af), 2.0)
        w = np.where(struck, w * fac_on, w * fac_off)
        if i < t:
            Qlt *= g
    return eta


def cdiv(a, b):
    return -((-a) // b)


def envelope(gs, t, L, strict=True):
    Qs = 1
    for g in gs[:t]:
        Qs *= g
    if Qs > L:
        return float("inf"), False
    eta = 0.0
    Qlt = 1
    for g in gs[:t]:
        mm, mp = L // Qlt, cdiv(L, Qlt)
        if mm < 1:
            return float("inf"), False
        a = min(1.0, 2 * cdiv(mp, g) / mm)
        if strict and a > 0.5:
            return float("inf"), False
        eta += a * a
        Qlt *= g
    mm, mp = L // Qs, cdiv(L, Qs)
    killed = 0
    for g in gs[t:]:
        s = mm - killed
        if s <= 0:
            return float("inf"), False
        a = min(1.0, 2 * cdiv(mp, g) / s)
        if strict and a > 0.5:
            return float("inf"), False
        eta += a * a
        killed += 2 * cdiv(mp, g)
        if eta >= 1.0:
            return eta, False
    return eta, eta < 1.0


def threshold(gs, strict=True, cap=10 ** 60):
    """least L with SF-CAP < 1 at the best cut (None if no cut works below cap)"""
    best = None
    for t in range(len(gs) + 1):
        lo, hi = 1, 2
        ok_hi = False
        while hi <= cap:
            _e, ok = envelope(gs, t, hi, strict)
            if ok:
                ok_hi = True
                break
            lo = hi
            hi *= 2
        if not ok_hi:
            continue
        while hi - lo > max(1, lo // 2000):
            mid = (lo + hi) // 2
            _e, ok = envelope(gs, t, mid, strict)
            if ok:
                hi = mid
            else:
                lo = mid
        if best is None or hi < best[0]:
            best = (hi, t)
    return best


def main():
    say("=" * 100)
    say("A. VALIDITY: the exact sub-machine fibre budget on r50's covered intervals")
    say("=" * 100)
    say("   each witness is an interval of A(K)-1 columns that K primes DO cover.")
    say("   the engine forbids eta_SF < 1 there, at every cut t.")
    say("")
    say("    K     L=A(K)-1   moduli                        min over cuts of eta_SF   verdict")
    wit = load_witnesses()
    for K in sorted(wit):
        L, w = wit[K]
        items = []
        okmod = True
        for kind, key, pos in w:
            g = modulus_of(kind, key)
            if g is None:
                okmod = False
                break
            items.append((g, pos))
        if not okmod:
            say("    %2d   %6d   (unresolved modulus type) - skipped" % (K, L))
            continue
        etas = []
        for t in range(0, len(items) + 1):
            e = eta_exact(L, items, t)
            if e is not None:
                etas.append((t, e))
        mn = min(etas, key=lambda x: x[1])
        mods = ",".join(str(g) for g, _ in sorted(items))
        say("    %2d   %6d   %-28s  %8.4f (at t=%d)        %s"
            % (K, L, mods[:28], mn[1], mn[0], "ok" if mn[1] >= 1.0 - 1e-9 else "REFUTED"))
    say()

    say("=" * 100)
    say("B. REACH: the unconditional SF-CAP threshold on the adversarial ladder")
    say("=" * 100)
    say("   the K smallest gears are the worst K-set for the budget.")
    say("   the open lemma is A(K) < W_{K+1} = (p_{K+1}^2 - 1)/6.")
    say("")
    say("    K   gears          A(K)   W_{K+1}   L*_SF(K)  cut t   L*/W    proves lemma?"
        "   consistent (L* >= A(K)-1)?   r51 L*max")
    r51 = {1: 2.0, 2: 5.033, 3: 10.32, 4: 16.06, 5: 24.98, 6: 40.34, 7: 76.55, 8: 106.5,
           9: 158.9, 10: 254.6, 11: 632.9, 12: 1025.0}
    for K in range(1, 13):
        gs = GEARS_ALL[:K]
        Wn = (GEARS_ALL[K] ** 2 - 1) // 6
        th = threshold(gs, strict=True)
        A = A_K[K]
        if th is None:
            say("    %2d  ..%-6d  %8d %8d       none      -       -          no"
                "                 -            %8.1f" % (K, gs[-1], A, Wn, r51[K]))
            continue
        Lst, t = th
        say("    %2d  ..%-6d  %8d %8d   %8d    %2d  %7.3f   %-10s      %-10s   %8.1f"
            % (K, gs[-1], A, Wn, Lst, t, Lst / Wn,
               "YES" if Lst < Wn else "no",
               "ok" if Lst >= A - 1 else "REFUTED", r51[K]))
    say()
    say("  (r51's L*max column is distortion_method.md R7, computed with the uniform-measure")
    say("   bound E[alpha] <= 2 ceil(L/g)/L; the SF-CAP column here assumes nothing about the")
    say("   engine's measure and self-verifies alpha <= 1/2, so it is strictly more conservative.)")

    say()
    say("=" * 100)
    say("C. The first moment under the engine's own measure, on the covered witnesses")
    say("=" * 100)
    say("   r51's envelope bounds M^(1) by its UNIFORM value 2 ceil(L/g)/L.  Under the")
    say("   engine's measure M^(1) = P_{i-1}(B_i), and P_{i-1} is uniform on the LIVE fibres")
    say("   only.  ratio = M^(1) / (2 ceil(L/g)/L), fully refining partition (r51's).")
    say("")
    say("    K      L    worst gear   M^(1)     uniform 2ceil(L/g)/L    ratio")
    for K in sorted(wit):
        L, w = wit[K]
        items = []
        bad = False
        for kind, key, pos in w:
            g = modulus_of(kind, key)
            if g is None:
                bad = True
                break
            items.append((g, pos))
        if bad:
            continue
        items = sorted(items, key=lambda x: x[0])
        k = np.arange(L, dtype=np.int64)
        wt = np.full(L, 1.0 / L)
        Qlt = 1
        worst = (0.0, None, 0.0, 0.0)
        for g, pos in items:
            mod = Qlt
            pid = k % mod if mod > 1 else np.zeros(L, dtype=np.int64)
            npart = min(mod, L) if mod <= L else L
            if mod > L:
                pid = k.copy()
                npart = L
            struck = np.zeros(L, dtype=bool)
            struck[np.array(pos, dtype=np.int64)] = True
            tot = np.bincount(pid, weights=wt, minlength=npart)
            num = np.bincount(pid[struck], weights=wt[struck], minlength=npart)
            nz = tot > 0
            a = np.zeros(npart)
            a[nz] = num[nz] / tot[nz]
            M1 = float(num.sum())
            unif = 2 * cdiv(L, g) / L
            ratio = M1 / unif if unif > 0 else 0.0
            if ratio > worst[0]:
                worst = (ratio, g, M1, unif)
            af = a[pid]
            fac_on = np.where(af > 0, np.maximum(0.0, 2.0 - 1.0 / np.where(af > 0, af, 1.0)), 0.0)
            fac_off = np.minimum(1.0 / np.maximum(1e-300, 1.0 - af), 2.0)
            wt = np.where(struck, wt * fac_on, wt * fac_off)
            Qlt *= g
        say("    %2d %6d %10s   %8.5f          %8.5f        %6.3f"
            % (K, L, str(worst[1]), worst[2], worst[3], worst[0]))

    with open(os.path.join(OUT, "sf_gate.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
