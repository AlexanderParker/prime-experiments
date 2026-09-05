"""sf_envelope.py - the UNCONDITIONAL sub-machine fibre envelope (SF-CAP).

Branch R2.c.ii.  A theorem may not use the machine's own survivor counts.  What the
sub-machine fibre partition gives for free is:

  * head gears (g in S = {5..a}, partition refining mod Q_{<i}): every part is a full class
    mod Q_{<i}, hence ENTIRELY surviving, so
        alpha_g <= min(1, 2 ceil(m_i^+/g) / m_i^-),   m_i = L/Q_{<i}
    with no hypothesis; at m_i >> g this is 2/g.
  * tail gears (g > a, partition frozen at the classes mod Q_s): a live fibre is a class mod
    Q_s that no head gear strikes, so ALL m of its columns survive the sub-machine, and each
    tail gear removes at most 2 ceil(m^+/h) of them:
        s^-(g) = m^- - sum_{a < h < g} 2 ceil(m^+/h),
        alpha_g <= min(1, 2 ceil(m^+/g) / s^-(g)).                          (SF-CAP)

This is the whole point of the sub-machine: r52's one-block CAP has to subtract the small
gears' kills from the block too (2/5 + 2/7 = 0.686 gone before the first big gear, and their
positions unknown), and is vacuous from four gears.  Here the head kills WHOLE FIBRES and the
depletion inside a live fibre starts at zero.

Validity: with delta = 1/2 the engine's measure is uniform on each part's survivors and gives
every live part its P_0 mass, PROVIDED every alpha <= 1/2.  The bound above is self-verifying:
we only accept a cut for which the SF-CAP bound itself stays <= 1/2 at every gear ("strict").
The "loose" column allows the bound above 1/2 (term = min(1,cap)^2) for comparison.

Usage: uv run python research/anchor235/r54/sf_envelope.py
"""
import os
from math import log, isfinite, log10

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
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


PR = primes_upto(60000)
GEARS_ALL = [p for p in PR if p >= 5]


def next_prime(q):
    for p in PR:
        if p > q:
            return p
    raise ValueError


def window(q):
    qp = next_prime(q)
    return (q + 1) // 6 + 1, (qp * qp - 1) // 6


def W(q):
    lo, hi = window(q)
    return hi - lo + 1


def fmt(x):
    """format a possibly astronomically large integer"""
    if x is None:
        return "none"
    if x < 10 ** 15:
        return "%.3e" % float(x)
    e = len(str(x)) - 1
    return "1e%d" % e


def cdiv(a, b):
    return -((-a) // b)


def envelope(gs, t, L, strict=True):
    """SF-CAP at cut t on an interval of L columns.  Returns (eta, ok, worst_alpha)."""
    Qs = 1
    for g in gs[:t]:
        Qs *= g
    if Qs > L:
        return float("inf"), False, 1.0
    eta = 0.0
    worst = 0.0
    Qlt = 1
    # head
    for g in gs[:t]:
        mm = L // Qlt
        mp = cdiv(L, Qlt)
        if mm < 1:
            return float("inf"), False, 1.0
        a = min(1.0, 2 * cdiv(mp, g) / mm)
        worst = max(worst, a)
        if strict and a > 0.5:
            return float("inf"), False, worst
        eta += a * a
        Qlt *= g
    # tail
    mm = L // Qs
    mp = cdiv(L, Qs)
    killed = 0
    for g in gs[t:]:
        s = mm - killed
        if s <= 0:
            return float("inf"), False, 1.0
        a = min(1.0, 2 * cdiv(mp, g) / s)
        worst = max(worst, a)
        if strict and a > 0.5:
            return float("inf"), False, worst
        eta += a * a
        killed += 2 * cdiv(mp, g)
        if eta >= 1.0:
            return eta, False, worst
    return eta, eta < 1.0, worst


def limit_envelope(gs, t, strict=True):
    """L -> infinity: head terms 4/g^2, tail terms ((2/g)/(1 - sum_{tail h<g} 2/h))^2."""
    eta = sum(4.0 / (g * g) for g in gs[:t])
    c = 0.0
    for g in gs[t:]:
        if c >= 1.0:
            return float("inf")
        a = min(1.0, (2.0 / g) / (1.0 - c))
        if strict and a > 0.5:
            return float("inf")
        eta += a * a
        c += 2.0 / g
        if eta >= 1.0:
            return float("inf")
    return eta


def threshold(gs, t, strict=True, cap=10 ** 400):
    """least L with SF-CAP < 1 at cut t (None if the limit envelope already fails)."""
    if not isfinite(limit_envelope(gs, t, strict)):
        return None
    Qs = 1
    for g in gs[:t]:
        Qs *= g
    lo = Qs
    hi = Qs
    for _ in range(400):
        eta, ok, _w = envelope(gs, t, hi, strict)
        if ok:
            break
        lo = hi
        hi *= 2
        if hi > cap:
            return None
    else:
        return None
    while hi - lo > max(1, lo // 1000):
        mid = (lo + hi) // 2
        _e, ok, _w = envelope(gs, t, mid, strict)
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    say("=" * 104)
    say("A. SF-CAP on the real window: is the unconditional budget below 1 at L = W(q)?")
    say("=" * 104)
    say("   q      W(q)   cut t  sub-machine    Q_s      m     eta strict   eta loose   verdict")
    for q in (17, 19, 23, 29, 31, 37, 41, 43, 59, 97, 199, 499, 997, 1999):
        gs = [g for g in GEARS_ALL if g <= q]
        L = W(q)
        best = None
        for t in range(len(gs) + 1):
            es, oks, _ = envelope(gs, t, L, strict=True)
            el, okl, _ = envelope(gs, t, L, strict=False)
            key = (0 if oks else 1, es if isfinite(es) else 9e9, el)
            if best is None or key < best[0]:
                best = (key, t, es, el, oks)
        _k, t, es, el, oks = best
        Qs = 1
        for g in gs[:t]:
            Qs *= g
        sm = "{5..%d}" % gs[t - 1] if t else "{}"
        say("  %5d %9d %5d  %-11s %7d %7.1f  %10s  %10s   %s"
            % (q, L, t, sm, Qs, L / Qs,
               ("%.4f" % es) if isfinite(es) else "inf",
               ("%.4f" % el) if isfinite(el) else "inf",
               "PROVES no cover" if oks else "vacuous"))
    say()

    say("=" * 104)
    say("B. The admissible cut: the limit (L -> infinity) SF-CAP budget by cut, and the")
    say("   threshold L*_SF(q) = the shortest interval the unconditional envelope can address")
    say("=" * 104)
    say("   q   gears     best cut a   Q_s(a)          ln a/ln q   limit eta   L*_SF(q)        L*/W(q)"
        "     L* at t=0 (block)   L* at t=all (r51 fibre)")
    rows = []
    for q in (17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 71, 97, 127, 199, 307, 499, 701, 997, 1499, 1999):
        gs = [g for g in GEARS_ALL if g <= q]
        best = None
        for t in range(len(gs) + 1):
            Lst = threshold(gs, t, strict=True)
            if Lst is None:
                continue
            if best is None or Lst < best[1]:
                best = (t, Lst, limit_envelope(gs, t, True))
        if best is None:
            say("  %5d %5d      -             -               -           -           none" % (q, len(gs)))
            rows.append((q, None, None))
            continue
        t, Lst, lim = best
        Qs = 1
        for g in gs[:t]:
            Qs *= g
        a = gs[t - 1] if t else 1
        L0 = threshold(gs, 0, strict=True)
        Lall = threshold(gs, len(gs), strict=True)
        say("  %5d %5d   %8d  %-22s %8.4f  %9.5f   %10s   %.3e   %14s   %14s"
            % (q, len(gs), a, ("%d" % Qs) if Qs < 10 ** 12 else "%.3e" % Qs,
               log(a) / log(q) if a > 1 else 0.0, lim, fmt(Lst),
               10 ** (log10(float(Lst)) - log10(W(q))) if Lst < 10 ** 300 else float("inf"),
               fmt(L0), fmt(Lall)))
        rows.append((q, a, Lst))
    say()
    say("  r51's fully refining fibre envelope, for comparison (distortion_method.md R5):")
    say("    q      59        97        199       499       997")
    say("    L*   1.07e4    4.08e6    1.25e14   1.93e30   9.33e52")
    say("    ln(cut)/ln q  0.629     0.644     0.702     0.703     0.713   (limit 0.728)")
    say()

    say("=" * 104)
    say("C. Machine-free: the per-gear tail term of SF-CAP against r52's block CAP")
    say("=" * 104)
    say("   the sub-machine head is {5..a}; the tail depletion inside a live fibre starts at 0,")
    say("   where a block's starts at 1 - prod_{g<=a}(1-2/g).")
    for a in (7, 13, 17):
        gs = GEARS_ALL
        t = len([g for g in gs if g <= a])
        say("  head {5..%d}:  head cost sum 4/g^2 = %.5f" % (a, sum(4.0 / (g * g) for g in gs[:t])))
        say("       g     2/g      SF-CAP alpha  term      cum eta      block-CAP alpha  term")
        c = 0.0
        eta = sum(4.0 / (g * g) for g in gs[:t])
        pi = 1.0
        for g in gs[:t]:
            pi *= (1 - 2.0 / g)
        cb = 1 - pi
        etab = sum(4.0 / (g * g) for g in gs[:t])
        for g in gs[t:t + 26]:
            aa = (2.0 / g) / (1 - c) if c < 1 else 9.9
            ab = (2.0 / g) / (1 - cb) if cb < 1 else 9.9
            aa = min(1.0, aa)
            ab = min(1.0, ab)
            eta += aa * aa
            etab += ab * ab
            say("   %5d  %.5f     %.5f   %.5f   %.5f          %.5f   %.5f"
                % (g, 2.0 / g, aa, aa * aa, eta, ab, ab * ab))
            c += 2.0 / g
            cb += 2.0 / g
            if c >= 1 and cb >= 1:
                break
        say()

    with open(os.path.join(OUT, "sf_envelope.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
