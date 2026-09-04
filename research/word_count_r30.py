"""Round 30 (constructor), item (b): THE L QUESTION FROM THE COVER HALF.

Bound the number of REALISED legal words of length m at machine M by what the
machine-free instruments can see, and test the bound against the exact counts.

THE LADDER, for a word length m:
    A_m   abstract T3-legal words over the alphabet Lambda (closed form, B4);
    S_m   those that survive PHASE SATURATION at every gear of M - i.e. whose
          prefix-sum point set X has, at every gear g, a translate inside the
          exposed set E_g.  By CRT this is exactly "some slot k has X + k all
          OPEN", i.e. the depth-0 term E_0(w) = prod_g c_g(X) of R43's pruned
          inclusion-exclusion counter is >= 1.  (The exposure half.)
    S_m^(2), S_m^(4)  those whose depth-2 / depth-4 Bonferroni upper bound
          E_s(w) = sum_{|T| <= s} (-1)^|T| N(X u T) is still >= 1;
    D_m   the REALISED words (N(w) > 0: the cover half).  D_m > 0 iff L >= m.
    EXPCAP(M) = max{m : S_m > 0};  CORRCAP = the same at gears {5,7} only (R75).

THEOREM (B0, stated and asserted here): a length-m word survives phase
saturation at M iff it survives at the sub-machine {g in M : g <= 2m+2}, since a
gear g > 2(m+1) always has a free translate (|X| = m+1 points forbid at most
2(m+1) of its g residues).  So S_m and EXPCAP depend on M only through the
SMALL gears and the alphabet.

Usage:  uv run python research/word_count_r30.py [--upto 53] [--mcap 30]
        [--nodes 3000000]
"""
import json
import os
import sys
import time
from itertools import combinations
from math import comb, prod

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DDIR = os.path.join(HERE, "data")
R30 = os.path.join(DDIR, "r30")
from qualrun_zerocert import pattern_count           # noqa: E402

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145, 59: 161}
KNOWN_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2,
           43: 2, 47: 4, 53: 3}
# R75's CORRCAP row (gears 5,7 only; values <= F; T3), the gate
CORRCAP_R75 = {19: 4, 23: 2, 29: 3, 31: 5, 37: 25, 41: 25, 43: 11, 47: 5,
               53: None}          # None = INFINITE
# exact realised single-gap value sets where on record beyond the scans
HOLES_41 = {84, 87, 89}


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def next_prime(y):
    p = y + 1
    while not is_prime(p):
        p += 1
    return p


def gears_of(y):
    return [p for p in range(5, y + 1) if is_prime(p)]


def alphabet(y, vals=None):
    q1 = next_prime(y)
    u1 = round(q1 / 6)
    a, b = 2 * u1, q1 - 2 * u1
    F = KNOWN_F[y]
    out = []
    for v in range(1, F + 1):
        if vals is not None and v not in vals:
            continue
        r = v % q1
        if r == 0:
            out.append((v, 0))
        elif r == a % q1:
            out.append((v, 1))
        elif r == b % q1:
            out.append((v, -1))
    return q1, a, b, out


def A_closed(p, la, lb, m):
    """abstract T3 words of length m: choose which k positions are padded,
    the other n = m-k positions carry an alternating class word."""
    tot = 0
    for k in range(m + 1):
        n = m - k
        if n == 0:
            T = 1
        else:
            h, l = (n + 1) // 2, n // 2
            T = la ** h * lb ** l + lb ** h * la ** l
        tot += comb(m, k) * p ** k * T
    return tot


def enum_T3(letters, m):
    words = [((), 0)]
    for _ in range(m):
        nxt = []
        for w, last in words:
            for v, c in letters:
                if c and c == last:
                    continue
                nxt.append((w + (v,), c if c else last))
        words = nxt
    return [w for w, _ in words]


def exposed_mask(g):
    u = pow(6, -1, g)
    m = (1 << g) - 1
    m &= ~(1 << (u % g))
    m &= ~(1 << ((-u) % g))
    return m


def rot(mask, t, g):
    t %= g
    return ((mask >> t) | (mask << (g - t))) & ((1 << g) - 1)


def ps_survivors(letters, gears, mcap, node_cap):
    """DFS over T3 words with per-gear admissible-translate bitmasks.
    Returns (S list indexed by length, longest word found, nodes, capped)."""
    E = [(g, exposed_mask(g)) for g in gears]
    S = [0] * (mcap + 1)
    longest = [()]
    nodes = [0]
    capped = [False]

    def rec(word, last, x, masks, depth):
        if depth == mcap:
            return
        for v, c in letters:
            if c and c == last:
                continue
            nx = x + v
            nm = []
            dead = False
            for (g, em), mk in zip(E, masks):
                m2 = mk & rot(em, nx, g)
                if m2 == 0:
                    dead = True
                    break
                nm.append(m2)
            if dead:
                continue
            nodes[0] += 1
            if nodes[0] > node_cap:
                capped[0] = True
                return
            S[depth + 1] += 1
            w2 = word + (v,)
            if len(w2) > len(longest[0]):
                longest[0] = w2
            rec(w2, c if c else last, nx, nm, depth + 1)
            if capped[0]:
                return

    rec((), 0, 0, [em for _, em in E], 0)
    return S, longest[0], nodes[0], capped[0]


def corrcap_automaton(letters):
    """Longest T3 word whose prefix-sum walk stays in E mod 35, or None if a
    cycle exists (R75's CORRCAP) - exact on the 35 x 3 state automaton."""
    E35 = {r for r in range(35) if r % 5 not in (1, 4) and r % 7 not in (6, 1)}
    # gear 5: u = 1, teeth {1,4}; gear 7: u = 6, teeth {6,1}
    states = [(r, c) for r in E35 for c in (0, 1, -1)]
    adj = {s: [] for s in states}
    for (r, last) in states:
        for v, c in letters:
            if c and c == last:
                continue
            r2 = (r + v) % 35
            if r2 in E35:
                adj[(r, last)].append((r2, c if c else last))
    # longest path with cycle detection (DFS, memo)
    colour, best = {}, {}

    def dfs(s):
        colour[s] = 1
        b = 0
        for t in adj[s]:
            if colour.get(t) == 1:
                raise ValueError("cycle")
            if colour.get(t) is None:
                dfs(t)
            b = max(b, 1 + best[t])
        colour[s] = 2
        best[s] = b

    try:
        for s in states:
            if colour.get(s) is None:
                dfs(s)
    except ValueError:
        return None
    return max(best[(r, 0)] for r in E35)


def bonferroni(gears, X, Y, depth):
    """partial sums S_0..S_depth of the pruned IE, S_k = sum_{|T|=k} N(X u T)."""
    base = []
    for g in gears:
        em = exposed_mask(g)
        mm = (1 << g) - 1
        for x in X:
            mm &= rot(em, x, g)
        if mm == 0:
            return [0] * (depth + 1)
        base.append((g, em, mm))
    S = [prod(mm.bit_count() for _, _, mm in base)] + [0] * depth
    Ys = list(Y)
    rotc = [[rot(em, t, g) for (g, em, _) in base] for t in Ys]
    masks0 = [mm for _, _, mm in base]

    def rec(start, masks, k):
        if k == depth:
            return
        for i in range(start, len(Ys)):
            nm, n = [], 1
            for m, r in zip(masks, rotc[i]):
                mr = m & r
                if mr == 0:
                    n = 0
                    break
                n *= mr.bit_count()
                nm.append(mr)
            if n == 0:
                continue
            S[k + 1] += n
            rec(i + 1, nm, k + 1)

    rec(0, masks0, 0)
    return S


def prefix_X(w):
    X, acc = [0], 0
    for v in w:
        acc += v
        X.append(acc)
    return X


def realised_words(y):
    """exact D_m from the counted census (m <= 4) where it exists."""
    p = os.path.join(R30, "occ_%d_words.json" % y)
    if not os.path.exists(p):
        return None, None
    with open(p) as fh:
        raw = json.load(fh)
    words = {tuple(int(t) for t in k.split()): v for k, v in raw.items()}
    q1, a, b, _ = alphabet(y)
    cls = {v: c for v, c in alphabet(y)[3]}

    def t3(w):
        last = 0
        for v in w:
            c = cls.get(v)
            if c is None:
                return False
            if c:
                if c == last:
                    return False
                last = c
        return True
    D = {}
    for w in words:
        if t3(w):
            D[len(w)] = D.get(len(w), 0) + 1
    vals = None
    hp = os.path.join(R30, "occ_%d.npz" % y)
    if os.path.exists(hp):
        h = np.load(hp)["hist"]
        vals = set(int(v) for v in np.flatnonzero(h))
    return D, vals


def main():
    args = sys.argv[1:]

    def opt(nm, d):
        return type(d)(args[args.index(nm) + 1]) if nm in args else d

    upto = opt("--upto", 53)
    mcap = opt("--mcap", 30)
    node_cap = opt("--nodes", 3_000_000)
    lines = []

    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    out("=" * 78)
    out("ITEM (b): REALISED LEGAL WORDS BOUNDED FROM THE EXPOSURE HALF")
    out("=" * 78)
    summary = {}
    for y in (11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53):
        if y > upto:
            break
        gears = gears_of(y)
        F = KNOWN_F[y]
        D, vals = realised_words(y)
        if vals is None:
            gp = os.path.join(DDIR, "r26", "ghist_%d.csv" % y)
            if os.path.exists(gp):          # Mechanic's exact cyclic histogram
                vals = set()
                with open(gp) as fh:
                    for line in fh:
                        parts = line.strip().split(",")
                        try:
                            if int(parts[0]) == y and int(parts[2]) > 0:
                                vals.add(int(parts[1]))
                        except (ValueError, IndexError):
                            continue
                out("   (realised value set read from Mechanic's r26 ghist_%d.csv)"
                    % y)
        if y == 41:
            vals = set(range(1, 92)) - HOLES_41
        q1, a, b, lam_all = alphabet(y)
        _, _, _, lam_real = alphabet(y, vals) if vals is not None else (None,) * 4
        p = sum(1 for _, c in lam_all if c == 0)
        la = sum(1 for _, c in lam_all if c == 1)
        lb = sum(1 for _, c in lam_all if c == -1)
        out("")
        out("machine %d  q'=%d  a=%d b=%d  F=%d  gears %s" % (y, q1, a, b, F, gears))
        out("   Lambda_all (residue-legal <= F) = %s   [padded %d, class a %d, "
            "class b %d]" % ([v for v, _ in lam_all], p, la, lb))
        if lam_real is not None:
            out("   Lambda(M) (realised values only) = %s   [exact value set]"
                % [v for v, _ in lam_real])
        else:
            out("   realised value set NOT on record - the abstract alphabet is used")
        # B4 closed form vs enumeration
        for m in range(1, 7):
            assert A_closed(p, la, lb, m) == len(enum_T3(lam_all, m)), ("B4", y, m)
        out("   B4 closed form A_m == enumeration at m = 1..6: OK")
        # CORRCAP gate
        cc = corrcap_automaton(lam_all)
        ref = CORRCAP_R75.get(y, "n/a")
        out("   CORRCAP (gears 5,7 only, automaton, exact): %s   R75: %s   %s"
            % ("INFINITE" if cc is None else cc,
               "n/a" if ref == "n/a" else ("INFINITE" if ref is None else ref),
               "GATE OK" if ref == "n/a" or ref == cc else "** MISMATCH"))
        if ref != "n/a":
            assert ref == cc, ("CORRCAP gate", y, cc, ref)
        # S_m: all gears, abstract alphabet and realised alphabet
        t0 = time.time()
        S_all, longest, nodes, capped = ps_survivors(lam_all, gears, mcap, node_cap)
        if lam_real is not None:
            S_real, longest_r, nodes_r, capped_r = ps_survivors(lam_real, gears,
                                                                 mcap, node_cap)
        else:
            S_real, longest_r, capped_r = None, None, False
        expcap = max((m for m in range(1, mcap + 1) if S_all[m] > 0), default=0)
        expcap_r = (max((m for m in range(1, mcap + 1) if S_real[m] > 0), default=0)
                    if S_real else None)
        # B0: sub-machine check at every m <= expcap (and one beyond)
        b0_ok = True
        for m in range(1, min(expcap + 1, mcap) + 1):
            sub = [g for g in gears if g <= 2 * m + 2]
            S_sub, _, _, cap2 = ps_survivors(lam_all, sub, m, node_cap)
            if cap2 or capped:
                continue
            if S_sub[m] != S_all[m]:
                b0_ok = False
                out("   B0 VIOLATION at m=%d: sub-machine %s gives %d, all gears %d"
                    % (m, sub, S_sub[m], S_all[m]))
        assert b0_ok, ("B0", y)
        out("   B0 lemma (S_m decided by gears <= 2m+2): asserted at m = 1..%d"
            % min(expcap + 1, mcap))
        # Bonferroni depth 2 / 4 on the survivors (abstract alphabet), m19..m31
        bon = {}
        if 19 <= y <= 37 and not capped:
            for m in range(1, (expcap if y <= 31 else min(expcap, 3)) + 1):
                words = [w for w in enum_T3(lam_all, m)]
                kill2 = kill4 = nsurv = 0
                nexact = nreal = 0
                ratio_min = None
                for w in words:
                    X = prefix_X(w)
                    Y = [t for t in range(1, X[-1]) if t not in set(X)]
                    S = bonferroni(gears, X, Y, 2)
                    if S[0] == 0:
                        continue
                    nsurv += 1
                    E2 = S[0] - S[1] + S[2]
                    if E2 < 1:
                        kill2 += 1
                    if len(Y) <= 45:
                        S4 = bonferroni(gears, X, Y, 4)
                        E4 = S4[0] - S4[1] + S4[2] - S4[3] + S4[4]
                        if E4 < 1:
                            kill4 += 1
                    if X[-1] <= 70:
                        cnt, _ = pattern_count(gears, X, Y)
                        if cnt is not None:
                            nexact += 1
                            if cnt > 0:
                                nreal += 1
                                r = S[0] // cnt
                                ratio_min = r if ratio_min is None else min(ratio_min, r)
                bon[m] = (nsurv, kill2, kill4, nexact, nreal, ratio_min)
        # table
        out("   %-3s %10s %10s %10s %8s %8s %8s" % ("m", "A_m", "S_m(all)",
                                                    "S_m(real)", "E2<1", "E4<1", "D_m"))
        for m in range(1, mcap + 1):
            if S_all[m] == 0 and (S_real is None or S_real[m] == 0) and m > 1 \
                    and S_all[m - 1] == 0 and (S_real is None or S_real[m - 1] == 0):
                break
            Dm = "-" if D is None else (str(D.get(m, 0)) if m <= 4 else
                                        ("0 (L cert.)" if KNOWN_L.get(y, 99) < m
                                         else "n/a"))
            if y == 41 and m <= 3:
                Dm = {1: "6", 2: "5", 3: "0"}[m] + " (r28)"
            k2 = k4 = ""
            if m in bon:
                k2 = str(bon[m][1])
                k4 = str(bon[m][2]) if bon[m][2] or bon[m][0] else ""
            out("   %-3d %10d %10s %10s %8s %8s %8s"
                % (m, A_closed(p, la, lb, m),
                   "%d%s" % (S_all[m], "+" if capped else ""),
                   "-" if S_real is None else "%d%s" % (S_real[m], "+" if capped_r else ""),
                   k2, k4, Dm))
        out("   EXPCAP(all gears, abstract alphabet) = %s%s   longest survivor %s"
            % (expcap, " (node cap hit - lower bound)" if capped else "",
               longest))
        if S_real is not None:
            out("   EXPCAP(all gears, realised alphabet) = %s%s   longest %s"
                % (expcap_r, " (node cap hit - lower bound)" if capped_r else "",
                   longest_r))
        out("   L(M) on record = %s ;  DFS nodes %d  [%.0f s]"
            % (KNOWN_L.get(y, "?"), nodes, time.time() - t0))
        for m, (nsurv, k2, k4, nex, nreal, rmin) in bon.items():
            out("   Bonferroni m=%d: %d survivors, depth-2 kills %d, depth-4 kills %d"
                " (|Y|<=45 only); exact N(w) on %d words (span<=70), %d realised,"
                " min E_0/N over realised = %s"
                % (m, nsurv, k2, k4, nex, nreal, rmin))
        # first-moment threshold (observation)
        fm = None
        hp = os.path.join(R30, "occ_%d.npz" % y)
        if os.path.exists(hp):
            h = np.load(hp)["hist"]
            N = int(h.sum())
            fl = sum(int(h[v]) for v, _ in lam_all if v < len(h))
            if fl:
                m1 = 1
                while N * fl ** m1 >= N ** m1 and m1 < 60:   # N f^m >= 1
                    m1 += 1
                fm = (fl, N, m1)
                out("   first-moment threshold (observation): f_legal = %d/%d = "
                    "%.3e, smallest m with N f^m < 1 is m = %d  (L = %d)"
                    % (fl, N, fl / N, m1, KNOWN_L[y]))
        summary[y] = dict(expcap=expcap, capped=capped, expcap_real=expcap_r,
                          corrcap=cc, L=KNOWN_L.get(y), S=S_all[:mcap + 1],
                          fm=fm, bon={str(k): v for k, v in bon.items()})
    out("")
    out("=" * 78)
    out("SUMMARY   M : L  CORRCAP(5,7)  EXPCAP(all gears)  EXPCAP - L")
    for y, s in summary.items():
        out("   m%-3d : %s   %-8s  %-8s  %s"
            % (y, s["L"], "INF" if s["corrcap"] is None else s["corrcap"],
               "%s%s" % (s["expcap"], "+" if s["capped"] else ""),
               "%s%s" % (s["expcap"] - s["L"], "+" if s["capped"] else "")
               if s["L"] is not None else "?"))
    out("=" * 78)
    out("all assertions passed")
    with open(os.path.join(R30, "word_count_r30.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(R30, "word_count_r30.json"), "w") as fh:
        json.dump(summary, fh, indent=0, default=str)


if __name__ == "__main__":
    main()
