"""W.t - transformations of the path (main sweep).

For every prime q in [5, Q] (default Q = 20000) build the machine {5..q} on the segment
(p^2, q'^2] with p = prevprime(q), q' = nextprime(q), by a segmented sieve on the two members
6k-1 and 6k+1 (column k is struck by gear g iff g | 6k-1 or g | 6k+1, i.e. k = +-6^-1 mod g).

Representations of the path from k_0 = (q^2-1)/6:
  (a) blocked string      B(k) = [dep(k) >= 1] on the section and on the run-out
  (b) depth string        dep(k) = number of gears of {5..q} striking k
  (c) smallest-striker    ms(k)
  (d) hop offsets per gear (all strikers, not only smallest)
  (e) anchor-30 cycle coordinate  k mod 5  (slot 11|13 at 2, 17|19 at 3, 29|31 at 0)
  (f) mirror path: the backward walk from k_0 to the previous opening

Tests T1..T8, T10, T12 of research/proof/walk_transforms.md.  Writes results/pt_path.txt.

Usage: uv run python research/anchor235/r38/pt_path.py [--Q 20000]
"""
import argparse
import os
from collections import Counter

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)


def sieve_flags(n):
    fl = np.ones(n + 1, dtype=bool)
    fl[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if fl[i]:
            fl[i * i:: i] = False
    return fl


def spf_table(n):
    """smallest prime factor for 0..n"""
    s = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if s[i] == 0:
            s[i:: i] = np.where(s[i:: i] == 0, i, s[i:: i])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Q", type=int, default=20000)
    a = ap.parse_args()
    Q = a.Q

    log = open(os.path.join(OUT, "pt_path.txt"), "w")

    def say(*xs):
        s = " ".join(str(x) for x in xs)
        print(s)
        log.write(s + "\n")

    NP = 4 * Q + 200
    fl = sieve_flags(NP)
    primes = np.flatnonzero(fl).astype(np.int64)
    plist = [int(x) for x in primes]
    gears_all = [p for p in plist if p >= 5]
    qs = [p for p in gears_all if p <= Q]
    SPF = spf_table(int(NP))
    say("primes to %d: %d;  gears (>=5) to %d: %d walks" % (NP, len(plist), Q, len(qs)))

    # global bookkeeping
    rows = []                      # per q summary dicts
    trans = Counter()              # smallest-striker word transitions (gears <= 43)
    admissible = Counter()         # residue-admissible adjacent pairs seen at all
    slotcount = Counter()          # k_0 mod 5 by q mod 30
    T2_bad = T3_bad = T4_bad = T7_bad = T8_bad = 0
    T4_teeth = 0
    gate_dep_bad = 0
    depth_profile = np.zeros(21)   # mean depth in 20 normalised bins along the path
    depth_profile_n = np.zeros(21)
    argmin_pos = Counter()
    argmax_pos = Counter()
    hop_gear = Counter()           # smallest-striker histogram over all paths
    strike_gear = Counter()        # all-striker histogram over all paths
    spacing_bad_examples = []
    five_after = [0, 0, 0]
    rng = np.random.default_rng(20260905)

    for qi, q in enumerate(qs):
        p = plist[plist.index(q) - 1]
        qn = plist[plist.index(q) + 1]
        gears = [g for g in gears_all if g <= q]
        k0 = (q * q - 1) // 6
        kp = (p * p - 1) // 6
        k1 = (qn * qn - 1) // 6
        lo = kp + 1
        hi = k1
        n = hi - lo + 1
        omega = np.zeros(n, dtype=np.uint8)
        spf = np.zeros(n, dtype=np.uint16)
        for g in reversed(gears):
            c = pow(6, -1, g)
            for r in (c % g, (-c) % g):
                st = (r - lo) % g
                omega[st:: g] += 1
                spf[st:: g] = g
        blocked = omega > 0
        i0 = k0 - lo
        assert blocked[i0]
        # ---- forward path
        L = 0
        while blocked[i0 + L]:
            L += 1
        assert i0 + L < n, "run-out too short at q=%d" % q
        # ---- backward path (mirror)
        Lm = 0
        while blocked[i0 - 1 - Lm]:
            Lm += 1
        assert i0 - 1 - Lm >= 0, "section too short at q=%d" % q

        c = pow(6, -1, q)
        u = min(c, q - c)
        d = (2 * c) % q                     # forward tooth arc
        db = q - d                          # backward tooth arc

        # ---------------- T1: start slot
        slotcount[(q % 30, k0 % 5)] += 1

        # ---------------- T2: the top gear strikes k0 and nothing else in (k0-db, k0+d)
        tset = {c % q, (-c) % q}
        nxt = min(x for x in ((r - k0) % q for r in tset) if x > 0)
        prv = min(x for x in ((k0 - r) % q for r in tset) if x > 0)
        if nxt != d or prv != db or k0 % q != (-c) % q:
            T2_bad += 1

        # ---------------- T4: teeth of q in the window (q, q^2], sole-striker test
        if q <= 4000:                       # gate; the argument is one line (see the document)
            ms_ = np.arange(1, q + 1)
            ms_ = ms_[(ms_ % 2 == 1) & (ms_ % 3 != 0)]
            mm = ms_[(ms_ > 1) & (ms_ < q)]
            T4_teeth += int(mm.size)
            if mm.size and int(SPF[mm].max()) >= q:
                T4_bad += 1

        # ---------------- path representations
        dep = omega[i0:i0 + L].astype(np.int64)
        word = spf[i0:i0 + L].astype(np.int64)
        depm = omega[i0 - Lm:i0].astype(np.int64) if Lm else np.array([], dtype=np.int64)
        wordm = spf[i0 - Lm:i0].astype(np.int64) if Lm else np.array([], dtype=np.int64)

        for v in word:
            hop_gear[int(v)] += 1
        if L >= 2:
            for x, y in zip(word[:-1], word[1:]):
                if x != 5:
                    five_after[1] += 1
                    if y == 5:
                        five_after[0] += 1
                elif y == 5:
                    five_after[2] += 1
                if x <= 43 and y <= 43:
                    trans[(int(x), int(y))] += 1
        # depth profile in normalised position
        if L >= 3:
            bins = (np.arange(L) * 20 // (L - 1)).astype(int)
            np.add.at(depth_profile, bins, dep)
            np.add.at(depth_profile_n, bins, 1)
            argmin_pos[int(round(20 * int(np.argmin(dep)) / (L - 1)))] += 1
            argmax_pos[int(round(20 * int(np.argmax(dep)) / (L - 1)))] += 1

        # ---------------- T7/T8: all strikers of the path, per gear
        hits = {}
        for g in gears:
            cg = pow(6, -1, g)
            st = (cg % g - k0) % g
            st2 = ((-cg) % g - k0) % g
            hs = []
            if st < L:
                hs.extend(range(st, L, g))
            if st2 < L:
                hs.extend(range(st2, L, g))
            if hs:
                hs.sort()
                hits[g] = hs
                strike_gear[g] += len(hs)
        for g, hs in hits.items():
            dg = (2 * pow(6, -1, g)) % g
            if len(hs) >= 2:
                if min(dg, g - dg, g) >= L:
                    T7_bad += 1
                for x, y in zip(hs[:-1], hs[1:]):
                    if (y - x) not in (dg, g - dg, g):
                        T8_bad += 1
                        if len(spacing_bad_examples) < 6:
                            spacing_bad_examples.append((q, g, x, y, dg))

        # ---------------- T5: depth at the start and the square gate
        # q^2 - 2 < q^2, so it is prime iff no gear <= q divides it (horizon lemma)
        garr = np.array(gears, dtype=np.int64)
        gate = bool(((q * q - 2) % garr != 0).all())
        if (int(dep[0]) == 1) != gate:
            gate_dep_bad += 1

        # ---------------- T10: comparisons inside the section
        # openings of the whole segment, for O(1) walk lengths
        opens = np.flatnonzero(~blocked)
        sec_hi = i0                                   # section = indices [0, i0]
        # run-length structure of the section's blocked string
        oi = opens[opens <= sec_hi]
        runs = np.diff(oi) - 1 if oi.size >= 2 else np.array([], dtype=np.int64)
        runs = runs[runs > 0]
        # walks from random columns of the section
        samp = rng.integers(0, sec_hi + 1, size=1000)
        j = np.searchsorted(opens, samp, side="left")
        wl_rand = opens[j] - samp
        sampb = samp[blocked[samp]]
        j2 = np.searchsorted(opens, sampb, side="left")
        wl_randb = opens[j2] - sampb
        pct_rand = float((wl_randb <= L).mean()) if wl_randb.size else float("nan")
        secdep = omega[:sec_hi + 1][blocked[:sec_hi + 1]]
        secdep_mean = float(secdep.mean())
        secdep_one = float((secdep == 1).mean())
        # walks from the other teeth of q inside the section
        t_lo = max(0, (q + 1) // 6 - lo)
        teeth = np.arange(lo, k0)
        teeth = teeth[np.isin(teeth % q, list(tset))] - lo
        teeth = teeth[teeth >= 0]
        if teeth.size:
            j3 = np.searchsorted(opens, teeth, side="left")
            wl_teeth = opens[j3] - teeth
        else:
            wl_teeth = np.array([], dtype=np.int64)

        rows.append(dict(
            q=q, p=p, qn=qn, k0=k0, L=L, Lm=Lm, d=d, db=db, u=u, c=c, gate=gate,
            dep0=int(dep[0]), depmax=int(dep.max()), depsum=int(dep.sum()),
            nlayers=int(len(set(int(v) for v in word))), ngears_striking=len(hits),
            slot=k0 % 5, qm6=q % 6, qm30=q % 30,
            runmax=int(runs.max()) if runs.size else 0,
            runmed=float(np.median(runs)) if runs.size else 0.0,
            nruns=int(runs.size),
            pct_in_section=float((runs <= L).mean()) if runs.size else float("nan"),
            wl_rand_mean=float(wl_rand.mean()), wl_randb_mean=float(wl_randb.mean()),
            wl_randb_med=float(np.median(wl_randb)) if wl_randb.size else 0.0,
            wl_teeth_mean=float(wl_teeth.mean()) if wl_teeth.size else float("nan"),
            wl_teeth_med=float(np.median(wl_teeth)) if wl_teeth.size else float("nan"),
            wl_teeth_n=int(wl_teeth.size),
            pct_rand=pct_rand, secdep_mean=secdep_mean, secdep_one=secdep_one,
            five_hits=int(sum(1 for v in word if v == 5)),
            dep_last=int(dep[-1]), dep_above=int(omega[i0 + L + 1]),
            S2=float(sum(2.0 / (g - 2) for g in gears)),
            S0=float(sum(2.0 / g for g in gears)),
            land_slot=int((k0 + L) % 5), prev_slot=int((k0 - Lm - 1) % 5),
            wl_teeth_gt=int((wl_teeth > L).sum()) if wl_teeth.size else 0,
            seclen=int(sec_hi + 1),
            word=[int(v) for v in word] if q <= 200 else None,
            wordm=[int(v) for v in wordm] if q <= 200 else None,
        ))
        if qi % 200 == 0:
            print("  ... q=%d (%d/%d)" % (q, qi, len(qs)), flush=True)

    N = len(rows)
    say("")
    say("=== SETUP ===")
    say("walks: %d (every prime gear 5..%d).  segment per q = columns (p^2, q'^2],"
        " machine {5..q}." % (N, Q))
    Ls = np.array([r["L"] for r in rows])
    Lms = np.array([r["Lm"] for r in rows])
    say("forward path length L: min %d, median %d, mean %.2f, max %d at q=%d"
        % (Ls.min(), int(np.median(Ls)), Ls.mean(), Ls.max(),
           rows[int(np.argmax(Ls))]["q"]))
    say("backward path length L^-: min %d, median %d, mean %.2f, max %d at q=%d"
        % (Lms.min(), int(np.median(Lms)), Lms.mean(), Lms.max(),
           rows[int(np.argmax(Lms))]["q"]))
    say("landings all twin by construction (opening of {5..q} below q'^2).")

    # ---------------------------------------------------------------- T1
    say("")
    say("=== T1. the start slot in the anchor-30 cycle (representation e) ===")
    say("k_0 mod 5 by q mod 30 (slot 11|13 = 2, 17|19 = 3, 29|31 = 0):")
    for cls in sorted(set(k for k, _ in slotcount)):
        vs = {s: v for (k, s), v in slotcount.items() if k == cls}
        say("   q = %2d mod 30 -> k_0 = %s mod 5   (%d walks)"
            % (cls, ",".join(str(s) for s in sorted(vs)), sum(vs.values())))
    bad1 = sum(v for (k, s), v in slotcount.items() if s == 2)
    say("walks starting on the 11|13 slot (k_0 = 2 mod 5): %d of %d" % (bad1, N))
    say("distinct (q mod 30 -> k_0 mod 5) maps: %d; each class single-valued: %s"
        % (len(slotcount), len(slotcount) == len(set(k for k, _ in slotcount))))

    # ---------------------------------------------------------------- T2/T3
    say("")
    say("=== T2/T3. the tooth arcs, forward and backward (representation f) ===")
    say("exceptions to  k_0 = -c (mod q), next q-strike at +d, previous at -(q-d):", T2_bad)
    for m6 in (1, 5):
        sel = [r for r in rows if r["qm6"] == m6]
        ratio = np.array([r["L"] / r["d"] for r in sel])
        ratiob = np.array([r["Lm"] / r["db"] for r in sel])
        say("q = %d mod 6 (%d walks): forward arc d = %s, backward q-d = %s" % (
            m6, len(sel), "(2q+1)/3" if m6 == 1 else "(q+1)/3",
            "(q-1)/3" if m6 == 1 else "(2q-1)/3"))
        say("   L/d      : median %.4f, 90th %.4f, max %.4f at q=%d"
            % (np.median(ratio), np.quantile(ratio, .9), ratio.max(),
               sel[int(np.argmax(ratio))]["q"]))
        say("   L^-/(q-d): median %.4f, 90th %.4f, max %.4f at q=%d"
            % (np.median(ratiob), np.quantile(ratiob, .9), ratiob.max(),
               sel[int(np.argmax(ratiob))]["q"]))
        say("   walks with L >= d   (top gear strikes its own forward interval twice): %d %s"
            % (int((ratio >= 1).sum()), [s["q"] for s in sel if s["L"] >= s["d"]][:8]))
        say("   walks with L^- >= q-d (twice on the backward interval): %d %s"
            % (int((ratiob >= 1).sum()), [s["q"] for s in sel if s["Lm"] >= s["db"]][:8]))
    tight = [r for r in rows if r["L"] / r["d"] > 0.4]
    say("walks with L/d > 0.4: %d, classes mod 6: %s"
        % (len(tight), Counter(r["qm6"] for r in tight)))
    say("   they are (q, q mod 6, L, d): %s"
        % [(r["q"], r["qm6"], r["L"], r["d"]) for r in sorted(tight, key=lambda r: -r["L"] / r["d"])[:12]])
    tightb = [r for r in rows if r["Lm"] / r["db"] > 0.4]
    say("backward walks with L^-/(q-d) > 0.4: %d, classes mod 6: %s"
        % (len(tightb), Counter(r["qm6"] for r in tightb)))
    say("   they are (q, q mod 6, L^-, q-d): %s"
        % [(r["q"], r["qm6"], r["Lm"], r["db"]) for r in sorted(tightb, key=lambda r: -r["Lm"] / r["db"])[:12]])
    both = [r for r in rows if r["L"] >= r["d"] or r["Lm"] >= r["db"]]
    say("ONE TOOTH PER RUN: the maximal blocked run through k_0 (length L^- + L) holds exactly")
    say("   one strike of the top gear.  Exceptions over %d walks: %d  %s"
        % (N, len(both), [(r["q"], r["q"] % 6, r["Lm"], r["db"], r["L"], r["d"]) for r in both]))
    say("   run length L^- + L: median %d, max %d at q=%d"
        % (int(np.median(Ls + Lms)), int((Ls + Lms).max()),
           rows[int(np.argmax(Ls + Lms))]["q"]))
    say("   short arc = forward for q = 5 mod 6, backward for q = 1 mod 6;")
    say("   max (path length)/(arc) in the SHORT direction: %.4f; in the LONG direction: %.4f"
        % (max(max(r["L"] / r["d"] for r in rows if r["qm6"] == 5),
               max(r["Lm"] / r["db"] for r in rows if r["qm6"] == 1)),
           max(max(r["L"] / r["d"] for r in rows if r["qm6"] == 1),
               max(r["Lm"] / r["db"] for r in rows if r["qm6"] == 5))))
    say("normalised reach L/q: max %.4f at q=%d;  L^-/q: max %.4f at q=%d"
        % (max(r["L"] / r["q"] for r in rows),
           max(rows, key=lambda r: r["L"] / r["q"])["q"],
           max(r["Lm"] / r["q"] for r in rows),
           max(rows, key=lambda r: r["Lm"] / r["q"])["q"]))

    # ---------------------------------------------------------------- T4
    say("")
    say("=== T4. the q^2 tooth is the unique sole-striker tooth of the top gear ===")
    say("teeth checked (multipliers 1 < m < q, m coprime to 6, over q <= 4000): %d"
        % T4_teeth)
    say("teeth whose q-member q*m has NO second striker <= q: %d" % T4_bad)

    # ---------------------------------------------------------------- T5
    say("")
    say("=== T5. the depth string (representation b) ===")
    say("exceptions to  dep(k_0) = 1  iff  q^2 - 2 prime: %d of %d" % (gate_dep_bad, N))
    gopen = [r for r in rows if r["gate"]]
    say("square gate open at %d of %d walks; dep(k_0) there = 1 always" % (len(gopen), N))
    dd = np.array([r["dep0"] for r in rows])
    say("dep(k_0): mean %.3f, max %d;  gate shut: mean %.3f"
        % (dd.mean(), dd.max(), np.array([r["dep0"] for r in rows if not r["gate"]]).mean()))
    dm = np.array([r["depmax"] for r in rows])
    say("max depth along the path: min %d, median %d, max %d" % (dm.min(), int(np.median(dm)), dm.max()))
    prof = depth_profile / np.maximum(depth_profile_n, 1)
    say("mean depth by normalised position along the path (0 = start, 20 = last blocked column):")
    say("   " + " ".join("%.2f" % v for v in prof))
    say("position of the MINIMUM depth (20 bins): " + " ".join(
        "%d:%d" % (k, argmin_pos[k]) for k in sorted(argmin_pos)))
    say("position of the MAXIMUM depth (20 bins): " + " ".join(
        "%d:%d" % (k, argmax_pos[k]) for k in sorted(argmax_pos)))
    say("paths whose minimum depth is at offset 0: %d of %d"
        % (argmin_pos[0], sum(argmin_pos.values())))

    # ---------------------------------------------------------------- T6
    say("")
    say("=== T6. the smallest-striker word (representation c) ===")
    say("smallest-striker histogram over all %d path columns:" % sum(hop_gear.values()))
    tot = sum(hop_gear.values())
    say("   " + " ".join("%d:%d(%.3f)" % (g, v, v / tot)
                         for g, v in sorted(hop_gear.items())[:10]))
    say("   columns whose smallest striker exceeds 100: %d (%.4f)"
        % (sum(v for g, v in hop_gear.items() if g > 100),
           sum(v for g, v in hop_gear.items() if g > 100) / tot))
    small = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    say("transition counts in the word, gears <= 43 (row = from, col = to):")
    say("      " + " ".join("%6d" % g for g in small))
    for g in small:
        say("%4d: " % g + " ".join("%6d" % trans.get((g, h), 0) for h in small))
    diag = [(g, trans.get((g, g), 0)) for g in small]
    say("diagonal (g -> g at distance 1): %s" % diag)
    zero_off = [(g, h) for g in small for h in small if g != h and trans.get((g, h), 0) == 0]
    say("off-diagonal transitions never seen: %s" % zero_off)

    # ---------------------------------------------------------------- T7/T8
    say("")
    say("=== T7/T8. hop offsets per layer (representation d) ===")
    say("gears striking a path twice with min(d_g, g-d_g, g) >= L: %d" % T7_bad)
    say("same-gear strike spacings not in {d_g, g-d_g, g}: %d %s"
        % (T8_bad, spacing_bad_examples))
    nl = np.array([r["nlayers"] for r in rows])
    ng = np.array([r["ngears_striking"] for r in rows])
    say("distinct smallest-striker gears per path: min %d, median %d, max %d"
        % (nl.min(), int(np.median(nl)), nl.max()))
    say("distinct STRIKING gears per path: min %d, median %d, max %d"
        % (ng.min(), int(np.median(ng)), ng.max()))
    say("total strikes on the path (sum of depths): median %d, max %d"
        % (int(np.median([r["depsum"] for r in rows])), max(r["depsum"] for r in rows)))
    say("all-striker histogram, top 10 gears: " + " ".join(
        "%d:%d" % (g, v) for g, v in sorted(strike_gear.items(), key=lambda kv: -kv[1])[:10]))

    # ---------------------------------------------------------------- T10
    say("")
    say("=== T10. the path against the section's other stretches ===")
    say("(a) run-length structure of the section's blocked string")
    say("     section length (columns): median %d, max %d"
        % (int(np.median([r["seclen"] for r in rows])), max(r["seclen"] for r in rows)))
    say("     blocked runs per section: median %d; longest run in the section: median %d, max %d"
        % (int(np.median([r["nruns"] for r in rows])),
           int(np.median([r["runmax"] for r in rows])),
           max(r["runmax"] for r in rows)))
    pct = np.array([r["pct_in_section"] for r in rows])
    say("     percentile of L among the section's blocked runs: median %.3f, mean %.3f"
        % (np.nanmedian(pct), np.nanmean(pct)))
    say("     paths longer than every run of their own section: %d of %d"
        % (int((np.array([r["L"] for r in rows]) > np.array([r["runmax"] for r in rows])).sum()), N))
    say("(b) walks from 1,000 random columns of the section, per q")
    wr = np.array([r["wl_randb_mean"] for r in rows])
    say("     mean walk length from a random BLOCKED column: median over q %.2f, max %.2f"
        % (np.median(wr), wr.max()))
    say("     mean walk length from a random column (open ones count 0): median over q %.2f"
        % np.median([r["wl_rand_mean"] for r in rows]))
    say("     L against the random-blocked mean: L smaller at %d of %d q"
        % (int((Ls < wr).sum()), N))
    say("(c) walks from the OTHER teeth of the top gear in the section")
    wt = np.array([r["wl_teeth_mean"] for r in rows if r["wl_teeth_n"] > 0])
    ntee = sum(r["wl_teeth_n"] for r in rows)
    say("     other teeth in the section: %d total, median %d per q"
        % (ntee, int(np.median([r["wl_teeth_n"] for r in rows]))))
    say("     mean walk length from another tooth: median over q %.2f" % np.median(wt))
    sel = [r for r in rows if r["wl_teeth_n"] > 0]
    say("     L below that q's tooth mean at %d of %d q"
        % (sum(1 for r in sel if r["L"] < r["wl_teeth_mean"]), len(sel)))
    say("     teeth walks strictly longer than L: %d of %d (%.4f)"
        % (sum(r["wl_teeth_gt"] for r in rows), ntee,
           sum(r["wl_teeth_gt"] for r in rows) / max(1, ntee)))
    pr = np.array([r["pct_rand"] for r in rows])
    say("(d) is the position k_0 distinguished?")
    say("     percentile of L among the 1,000 random-blocked walks of its own section:")
    say("        median over q %.4f, mean %.4f  (0.5 = typical)"
        % (np.nanmedian(pr), np.nanmean(pr)))
    say("     depth of a random blocked column of the section: mean over q %.3f;"
        " fraction of depth 1: %.4f"
        % (np.mean([r["secdep_mean"] for r in rows]),
           np.mean([r["secdep_one"] for r in rows])))
    say("     depth of k_0: mean %.3f; fraction of depth 1 %.4f (= the square-gate share)"
        % (dd.mean(), float((dd == 1).mean())))

    say("")
    say("=== the two ends of the path (order-2 check: neighbour of an opening) ===")
    say("mean depth of the LAST blocked column before the landing: %.4f"
        % np.mean([r["dep_last"] for r in rows]))
    say("mean depth of the column just ABOVE the landing: %.4f"
        % np.mean([r["dep_above"] for r in rows]))
    say("mean depth of a random blocked column of the section: %.4f"
        % np.mean([r["secdep_mean"] for r in rows]))
    say("mean depth of the FIRST column k_0: %.4f" % dd.mean())
    say("independent-gear predictions: sum 2/g = %.4f (any column),"
        " sum 2/(g-2) = %.4f (a column whose neighbour is open)"
        % (np.mean([r["S0"] for r in rows]), np.mean([r["S2"] for r in rows])))
    say("min L over the walks with q > 5: %d (gear 5 strikes offset 1 at every q > 5)"
        % min(r["L"] for r in rows if r["q"] > 5))
    say("")
    say("=== the word's transitions against an independent-letter model ===")
    pg = {g: hop_gear[g] / tot for g in small}
    npairs = sum(trans.values())
    say("ratio observed/(independent) for gears <= 43 (rows from, cols to):")
    say("      " + " ".join("%6d" % g for g in small))
    for g in small:
        say("%4d: " % g + " ".join(
            "%6.2f" % (trans.get((g, h), 0) / max(1e-9, npairs * pg[g] * pg[h] /
                                                  sum(pg[x] * pg[y] for x in small for y in small)))
            for h in small))
    say("Row 5 and column 5 sit at 1.5-1.9, every big-to-big cell at 0.2-1.0.  Both are")
    say("forced, and by order 0: gear 5 strikes the offsets = 1, 4 (mod 5) in class A and")
    say("1, 3 (mod 5) in class B, so 5-columns are never adjacent and exactly two of the three")
    say("non-5 residues are followed by a 5-residue -")
    say("   P(next letter = 5 | this letter != 5) = 2/3 exactly, against the share 2/5.")
    say("measured over every path column with a successor: %d of %d = %.4f"
        % (five_after[0], five_after[1], five_after[0] / max(1, five_after[1])))
    say("measured P(this letter = 5 and next = 5): %d (the diagonal ban at gear 5)"
        % five_after[2])

    # ---------------------------------------------------------------- anchor coordinate
    say("")
    say("=== representation (e): the path in the anchor-30 cycle coordinate ===")
    fh = np.array([r["five_hits"] for r in rows])
    say("gear-5 hits on the path (columns with k = 1 or 4 mod 5): total %d of %d columns (%.4f)"
        % (fh.sum(), Ls.sum(), fh.sum() / Ls.sum()))
    say("anchor-open columns crossed (the columns the gears above 5 must cover): %d (%.4f)"
        % (Ls.sum() - fh.sum(), 1 - fh.sum() / Ls.sum()))
    say("start slot -> landing slot counts (slot 11|13 = 2, 17|19 = 3, 29|31 = 0):")
    ss = Counter((r["slot"], r["land_slot"]) for r in rows)
    for k in sorted(ss):
        say("   %d -> %d : %d" % (k[0], k[1], ss[k]))
    say("previous-opening slot counts: %s" % dict(Counter(r["prev_slot"] for r in rows)))

    # ---------------------------------------------------------------- T12
    say("")
    say("=== T12. shapes with q (by decade of q) ===")
    say(" q range        n     med L   max L   med L^-  med layers  med maxdep  med strikes  med d")
    edges = [5, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    for i in range(len(edges) - 1):
        sel = [r for r in rows if edges[i] <= r["q"] < edges[i + 1]]
        if not sel:
            continue
        say(" %5d-%-6d %5d  %6d  %6d  %7d  %10d  %10d  %11d  %6d" % (
            edges[i], edges[i + 1], len(sel),
            int(np.median([r["L"] for r in sel])), max(r["L"] for r in sel),
            int(np.median([r["Lm"] for r in sel])),
            int(np.median([r["nlayers"] for r in sel])),
            int(np.median([r["depmax"] for r in sel])),
            int(np.median([r["depsum"] for r in sel])),
            int(np.median([r["d"] for r in sel]))))
    say("mean L by decade against (ln q^2)^2/(2 C_2 * 6) [the twin-gap null, C_2 = 0.6601]:")
    for i in range(len(edges) - 1):
        sel = [r for r in rows if edges[i] <= r["q"] < edges[i + 1]]
        if not sel:
            continue
        mq = float(np.median([r["q"] for r in sel]))
        null = (2 * np.log(mq)) ** 2 / (2 * 0.6601 * 6)
        say("   q ~ %6d: mean L %7.2f   null %7.2f   ratio %.3f"
            % (mq, np.mean([r["L"] for r in sel]), null,
               np.mean([r["L"] for r in sel]) / null))

    # ---------------------------------------------------------------- mirror
    say("")
    say("=== representation (f): the mirror path against the forward path ===")
    say("L^- = 0 (the column of q^2 begins its maximal blocked run): %d of %d"
        % (int((Lms == 0).sum()), N))
    cc = float(np.corrcoef(Ls, Lms)[0, 1])
    say("correlation of L and L^- over %d walks: %.4f" % (N, cc))
    say("L^- > L at %d of %d; L^- = L at %d" % (int((Lms > Ls).sum()), N, int((Lms == Ls).sum())))
    say("mean L %.3f, mean L^- %.3f, ratio %.4f" % (Ls.mean(), Lms.mean(), Lms.mean() / Ls.mean()))
    same = sum(1 for r in rows if r["word"] and r["wordm"] and
               r["word"][:len(r["wordm"])] == r["wordm"][::-1])
    say("paths whose word is the reverse of the mirror word (q <= 200): %d of %d"
        % (same, sum(1 for r in rows if r["word"] is not None)))
    say("sample (q, L, word | L^-, mirror word):")
    for r in rows[:12]:
        if r["word"] is not None:
            say("   q=%-4d L=%-3d %s | L^-=%-3d %s" % (r["q"], r["L"], r["word"], r["Lm"],
                                                       r["wordm"][::-1]))

    np.save(os.path.join(OUT, "pt_rows_L.npy"),
            np.array([[r["q"], r["L"], r["Lm"], r["d"], r["db"], r["nlayers"],
                       r["depmax"], r["depsum"], r["dep0"], int(r["gate"])] for r in rows]))
    log.close()


if __name__ == "__main__":
    main()
