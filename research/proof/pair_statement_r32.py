"""Pair statement F_2(M) <= F(M) + q' : exact gates (Prover A, round 32).

Entry points (all exact integer arithmetic, numpy sieves over full periods):
  real   -- machines {5..p}, p = 11..23: F, F_2, d_0, L2 discharge, re-phasing certificate,
            sole-coverer counts, lag-1 conditional means and the shuffle null.
  family -- tooth-counterfactual family V(p), p = 11,13,17[,19]: wrap instance, non-wrap
            slack, F_2 = 2 d_0 frequency, tightest non-wrap pair residue structure,
            single-gear certificate coverage, lag-1 effect.
  oneclass -- ordinary Jacobsthal on P_5..P_9: j, one-hole record, sole-coverer counts,
            single-prime certificate.
  d0     -- real machine d_0 (first twin prime above p) against q' for all p <= 10^6.
"""
import sys, json, time, itertools
from math import prod
import numpy as np

OUT = "C:/dev/primes/research/proof/"

PRIMES = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
NEXT = {PRIMES[i]: PRIMES[i + 1] for i in range(len(PRIMES) - 1)}


def u_of(g):
    return pow(6, -1, g)


def sieve_openings(gears, teeth):
    """teeth: dict gear -> set of struck residues.  Returns (P, openings ascending)."""
    P = prod(gears)
    w = np.ones(P, dtype=bool)
    for g in gears:
        for v in teeth[g]:
            w[v % g::g] = False
    return P, np.flatnonzero(w)


def cyc_gaps(ops, P):
    return np.diff(np.concatenate([ops, [ops[0] + P]]))


def struck_by(c, gears, teeth, exclude=()):
    return [g for g in gears if g not in exclude and (c % g) in teeth[g]]


def rephase_gap(x, gears, teeth, S_new):
    """S_new: dict gear -> new tooth set for the re-phased gears (a translate of the old set).
    Returns (L, R): the nearest open columns left/right of x in the re-phased machine, where
    blocked(c) = struck by a gear not in S_new (old teeth) or by a gear in S_new (new teeth).
    Every column of the re-phased machine is a translate of a column of M, so R - L is a gap
    of M.  x itself must be struck by some gear of S_new (asserted)."""
    others = [g for g in gears if g not in S_new]

    def blocked(c):
        for g in others:
            if (c % g) in teeth[g]:
                return True
        for g, T in S_new.items():
            if (c % g) in T:
                return True
        return False

    assert blocked(x)
    R = x + 1
    while blocked(R):
        R += 1
    L = x - 1
    while blocked(L):
        L -= 1
    return L, R


def single_gear_cert(x, gears, teeth, per_gear=None):
    """max over gears q0 and over translates of q0's tooth set containing x of the re-phased gap.
    If per_gear is a dict it receives gear -> best gap via that gear alone."""
    best = (0, None, None)
    for q0 in gears:
        old = sorted(teeth[q0])
        bg = 0
        for v0 in old:
            t = (x - v0) % q0
            T = {(v + t) % q0 for v in old}
            L, R = rephase_gap(x, gears, teeth, {q0: T})
            bg = max(bg, R - L)
            if R - L > best[0]:
                best = (R - L, q0, tuple(sorted(T)))
        if per_gear is not None:
            per_gear[q0] = bg
    return best


def two_gear_cert(x, gears, teeth, need):
    """re-phase q0 to strike x, q1 to any translate; stop as soon as a gap >= need is found."""
    best = 0
    for q0 in gears:
        old0 = sorted(teeth[q0])
        for v0 in old0:
            t0 = (x - v0) % q0
            T0 = {(v + t0) % q0 for v in old0}
            for q1 in gears:
                if q1 == q0:
                    continue
                old1 = sorted(teeth[q1])
                for t1 in range(q1):
                    T1 = {(v + t1) % q1 for v in old1}
                    L, R = rephase_gap(x, gears, teeth, {q0: T0, q1: T1})
                    if R - L > best:
                        best = R - L
                        if best >= need:
                            return best
    return best


def sole_counts(x, gL, gR, gears, teeth):
    """per-gear count of columns in the one-hole stretch (x-gL, x+gR) \\ {x} struck by that gear alone."""
    cnt = {g: 0 for g in gears}
    for c in range(x - gL + 1, x + gR):
        if c == x:
            continue
        s = struck_by(c, gears, teeth)
        if len(s) == 1:
            cnt[s[0]] += 1
    return cnt


def pair_analysis(P, ops, gears, teeth, qn, label, do_two=True, max_pairs=4000, verbose=True):
    gaps = cyc_gaps(ops, P)
    n = len(ops)
    F = int(gaps.max())
    prev = np.roll(gaps, 1)                # left gap of opening i is gaps[i-1]
    sums = prev + gaps
    F2 = int(sums.max())
    d0 = int(ops[1] - ops[0])
    both_big = int(np.count_nonzero((prev > qn) & (gaps > qn)))
    res = dict(label=label, P=int(P), n_open=int(n), F=F, F2=F2, d0=d0, qn=qn,
               L2_free=(F2 <= 2 * qn + 1), both_flanks_gt_qn=both_big,
               wrap_ok=(2 * d0 <= F + qn), F2_is_wrap=(F2 == 2 * d0))
    # non-wrap maximum
    mask = np.ones(n, dtype=bool)
    mask[0] = False                        # ops[0] == 0 is the wrap pair
    nw = int(sums[mask].max())
    res["nonwrap_max"] = nw
    res["nonwrap_slack"] = F + qn - nw
    # pairs exceeding the record: the only ones with content
    idx = np.flatnonzero(sums > F)
    res["pairs_above_F"] = int(len(idx))
    idx = idx[:max_pairs]
    cert_ok = cert_fail = two_ok = 0
    fails = []
    argmax_info = []
    top_ok = 0; which = {}; maxloss = 0; loss_top_max = 0
    for i in idx:
        x = int(ops[i]); gL = int(prev[i]); gR = int(gaps[i]); s = gL + gR
        need = s - qn
        pg = {}
        c, q0, T = single_gear_cert(x, gears, teeth, pg)
        top_ok += (pg[gears[-1]] >= need)
        which[q0] = which.get(q0, 0) + 1
        maxloss = max(maxloss, s - c)
        loss_top_max = max(loss_top_max, s - pg[gears[-1]])
        if c >= need:
            cert_ok += 1
        else:
            cert_fail += 1
            c2 = two_gear_cert(x, gears, teeth, need) if do_two else 0
            if c2 >= need:
                two_ok += 1
            fails.append(dict(x=x, gL=gL, gR=gR, cert1=c, cert2=c2, need=need))
        if s == F2:
            sc = sole_counts(x, gL, gR, gears, teeth)
            argmax_info.append(dict(x=x, gL=gL, gR=gR, cert1=c, cert_gear=q0, sole=sc))
    res.update(cert1_ok=cert_ok, cert1_fail=cert_fail, cert2_ok=two_ok,
               cert_fails=fails[:20], F2_argmax=argmax_info[:8],
               top_gear_ok=top_ok, best_gear_hist=which, max_min_loss=maxloss, max_loss_via_top=loss_top_max)
    # lag-1 conditional means (manager branch 5b) and shuffle null
    thr = 0.7 * F
    big = gaps >= thr
    nxt = np.roll(gaps, -1)
    res["mean_gap"] = float(gaps.mean())
    res["mean_after_big"] = float(nxt[big].mean())
    res["mean_after_record"] = float(nxt[gaps == F].mean())
    rng = np.random.default_rng(1)
    sh = []
    for _ in range(10):
        gsh = rng.permutation(gaps)
        sh.append(int((gsh + np.roll(gsh, 1)).max()))
    res["F2_shuffled_min_max"] = (min(sh), max(sh))
    if verbose:
        print(f"[{label}] P={P} open={n} F={F} F2={F2} d0={d0} q'={qn} L2free={res['L2_free']} "
              f"bothflanks>q'={both_big} wrap_ok={res['wrap_ok']} F2=2d0:{res['F2_is_wrap']} "
              f"nonwrap_max={nw} slack={res['nonwrap_slack']} pairs>F={res['pairs_above_F']} "
              f"cert1 ok/fail={cert_ok}/{cert_fail} cert2_ok={two_ok} top-alone ok={top_ok} best-gear={which} "
              f"max min-loss={maxloss} max loss via top={loss_top_max} "
              f"E[gap]={res['mean_gap']:.2f} E[after>=0.7F]={res['mean_after_big']:.2f} "
              f"E[after F]={res['mean_after_record']:.2f} shuffledF2={res['F2_shuffled_min_max']}")
        for a in argmax_info[:4]:
            print(f"    F2 argmax x={a['x']} ({a['gL']},{a['gR']}) cert1={a['cert1']} via gear {a['cert_gear']} sole={a['sole']}")
        for f in fails[:6]:
            print(f"    uncertified x={f['x']} ({f['gL']},{f['gR']}) cert1={f['cert1']} cert2={f['cert2']} need={f['need']}")
    return res


def real_teeth(gears):
    return {g: {u_of(g), g - u_of(g)} for g in gears}


def run_real():
    out = {}
    for p in [11, 13, 17, 19, 23]:
        gears = [g for g in PRIMES if g <= p]
        teeth = real_teeth(gears)
        t0 = time.time()
        P, ops = sieve_openings(gears, teeth)
        r = pair_analysis(P, ops, gears, teeth, NEXT[p], f"m{p}")
        # sole-coverer counts in the record (zero-hole) gaps as well
        gaps = cyc_gaps(ops, P)
        F = int(gaps.max())
        recs = np.flatnonzero(gaps == F)[:4]
        r["F_record_sole"] = []
        for i in recs:
            x0 = int(ops[i]); cnt = {g: 0 for g in gears}
            for c in range(x0 + 1, x0 + F):
                s = struck_by(c, gears, teeth)
                if len(s) == 1:
                    cnt[s[0]] += 1
            r["F_record_sole"].append(dict(start=x0, sole=cnt))
            print(f"    F record gap at {x0}: sole counts {cnt}")
        r["seconds"] = round(time.time() - t0, 1)
        out[f"m{p}"] = r
        sys.stdout.flush()
    json.dump(out, open(OUT + "pair_real_r32.json", "w"), indent=1)


def run_family(pmax=19, limit=None, cap=200):
    gears = [g for g in PRIMES if g <= pmax]
    qn = NEXT[pmax]
    ranges = [range(1, (g - 1) // 2 + 1) for g in gears]
    t0 = time.time()
    members = 0
    wrap_viol = []
    nonwrap_viol = []
    slack_min = None
    F2wrap = 0
    tight = []            # tightest non-wrap pairs: residue structure
    cert_pairs = cert_ok = 0
    cert_fail_members = 0
    lag_present = 0
    shuffle_below = 0
    Lcap = 0
    for vs in itertools.product(*ranges):
        if limit and members >= limit:
            break
        teeth = {g: {v, g - v} for g, v in zip(gears, vs)}
        P, ops = sieve_openings(gears, teeth)
        gaps = cyc_gaps(ops, P)
        F = int(gaps.max())
        prev = np.roll(gaps, 1)
        sums = prev + gaps
        F2 = int(sums.max())
        d0 = int(ops[1])
        members += 1
        if 2 * d0 > F + qn:
            wrap_viol.append(dict(v=vs, F=F, F2=F2, d0=d0))
        if F2 == 2 * d0:
            F2wrap += 1
        mask = np.ones(len(ops), dtype=bool); mask[0] = False
        i_nw = int(np.flatnonzero(mask)[np.argmax(sums[mask])])
        nw = int(sums[i_nw])
        slack = F + qn - nw
        if slack < 0:
            nonwrap_viol.append(dict(v=vs, F=F, F2=F2, nw=nw, x=int(ops[i_nw])))
        if slack_min is None or slack < slack_min[0]:
            slack_min = (slack, vs, F, nw, int(ops[i_nw]))
        if slack <= 3:
            x = int(ops[i_nw])
            zeros = sum(1 for g in gears if x % g == 0)
            tight.append(dict(v=vs, slack=slack, x=x, gL=int(prev[i_nw]), gR=int(gaps[i_nw]),
                              zero_gears=zeros, res=[x % g for g in gears], F=F))
        # single-gear certificate on pairs above F
        idx = np.flatnonzero(sums > F)
        member_fail = False
        for i in idx[:cap]:
            x = int(ops[i]); need = int(sums[i]) - qn
            c, _, _ = single_gear_cert(x, gears, teeth)
            cert_pairs += 1
            if c >= need:
                cert_ok += 1
            else:
                member_fail = True
        cert_fail_members += member_fail
        # lag-1 effect
        big = gaps >= 0.7 * F
        nxt = np.roll(gaps, -1)
        if nxt[big].mean() < gaps.mean():
            lag_present += 1
        rng = np.random.default_rng(members)
        gsh = rng.permutation(gaps)
        if F2 < int((gsh + np.roll(gsh, 1)).max()):
            shuffle_below += 1
        if members % 500 == 0:
            print(f"  V({pmax}) {members} members, {time.time()-t0:.0f}s", flush=True)
    tight.sort(key=lambda d: d["slack"])
    res = dict(pmax=pmax, qn=qn, members=members, wrap_violations=wrap_viol,
               nonwrap_violations=nonwrap_viol, nonwrap_slack_min=slack_min,
               F2_equals_2d0=F2wrap, tight_nonwrap=tight[:40],
               cert_pairs=cert_pairs, cert1_ok=cert_ok, members_with_uncertified_pair=cert_fail_members,
               lag1_effect_present=lag_present, F2_below_shuffle=shuffle_below,
               seconds=round(time.time() - t0, 1))
    print(f"V({pmax}) q'={qn}: members={members} wrap_viol={len(wrap_viol)} nonwrap_viol={len(nonwrap_viol)} "
          f"min nonwrap slack={slack_min} F2=2d0 at {F2wrap} members; cert1 {cert_ok}/{cert_pairs} pairs, "
          f"members with an uncertified pair {cert_fail_members}; lag1 effect present {lag_present}/{members}; "
          f"F2 below one shuffle {shuffle_below}/{members}; {res['seconds']}s")
    for w in wrap_viol[:5]:
        print("   wrap violation:", w)
    zg = [t["zero_gears"] for t in tight]
    print(f"   tight non-wrap pairs (slack<=3): {len(tight)}; zero-gear counts histogram "
          f"{ {k: zg.count(k) for k in sorted(set(zg))} }")
    for t in tight[:8]:
        print("   ", t)
    json.dump(res, open(OUT + f"pair_family_V{pmax}_r32.json", "w"), indent=1)


def run_oneclass():
    ps = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    out = {}
    for k in range(5, 10):
        gears = ps[:k]
        qn = ps[k]
        teeth = {g: {0} for g in gears}
        t0 = time.time()
        P, ops = sieve_openings(gears, teeth)
        gaps = cyc_gaps(ops, P)
        j = int(gaps.max())
        prev = np.roll(gaps, 1)
        sums = prev + gaps
        oh = int(sums.max())
        idx = np.flatnonzero(sums > j)
        cert_ok = 0
        arg = []
        top_ok = 0; which = {}; maxloss = 0; loss_top_max = 0
        for i in idx:
            x = int(ops[i]); gL = int(prev[i]); gR = int(gaps[i]); need = gL + gR - qn
            pg = {}
            c, q0, T = single_gear_cert(x, gears, teeth, pg)
            cert_ok += (c >= need)
            top_ok += (pg[gears[-1]] >= need)
            which[q0] = which.get(q0, 0) + 1
            maxloss = max(maxloss, gL + gR - c)
            loss_top_max = max(loss_top_max, gL + gR - pg[gears[-1]])
            if gL + gR == oh:
                arg.append(dict(x=x, gL=gL, gR=gR, cert1=c, gear=q0, sole=sole_counts(x, gL, gR, gears, teeth)))
        # pair at x = 1: (2, q'-1)
        i1 = int(np.searchsorted(ops, 1))
        pair1 = (int(prev[i1]), int(gaps[i1]))
        r = dict(k=k, P=int(P), j=j, onehole=oh, qn=qn, pairs_above_j=int(len(idx)), cert1_ok=cert_ok,
                 argmax=arg[:6], pair_at_1=pair1, seconds=round(time.time() - t0, 1))
        out[k] = r
        r.update(top_gear_ok=top_ok, best_gear_hist=which, max_min_loss=maxloss, max_loss_via_top=loss_top_max)
        print(f"P_{k}={P}: j={j} one-hole={oh} q'={qn} one-hole<=j+q':{oh <= j + qn} pairs>j={len(idx)} "
              f"cert1 ok={cert_ok} top-prime-alone ok={top_ok} best-gear hist={which} max min-loss={maxloss} "
              f"max loss via top={loss_top_max} pair at 1={pair1}")
        for a in arg[:4]:
            print(f"    one-hole argmax x={a['x']} ({a['gL']},{a['gR']}) cert1={a['cert1']} via {a['gear']} sole={a['sole']}")
        sys.stdout.flush()
    json.dump(out, open(OUT + "pair_oneclass_r32.json", "w"), indent=1)


def run_d0(N=10 ** 6, LIM=3 * 10 ** 7):
    s = np.ones(LIM + 1, dtype=bool); s[:2] = False
    for i in range(2, int(LIM ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    pr = np.flatnonzero(s)
    twins_lo = pr[s[pr + 2 <= LIM] if False else (pr + 2 <= LIM)]
    twins_lo = twins_lo[s[np.minimum(twins_lo + 2, LIM)]]
    ps = pr[(pr >= 5) & (pr <= N)]
    nxt = pr[np.searchsorted(pr, ps, side="right")]
    first_t = twins_lo[np.searchsorted(twins_lo, ps, side="right")]   # first twin lower member > p
    d0 = (first_t + 1) // 6
    ratio = d0 / nxt
    worst = int(np.argmax(ratio))
    ok = int(np.count_nonzero(d0 <= nxt))
    print(f"real machine d_0 vs q' for {len(ps)} primes 5..{N}: d0<=q' at {ok}/{len(ps)}; "
          f"max d0/q' = {ratio[worst]:.4f} at p={ps[worst]} (q'={nxt[worst]}, first twin {first_t[worst]}, d0={d0[worst]}); "
          f"max 6d0/p = {(6*d0/ps).max():.4f}; d0 at m7..m41: {[int(d0[np.searchsorted(ps, p)]) for p in [7,11,13,17,19,23,29,31,37,41]]}")
    json.dump(dict(N=N, count=len(ps), d0_le_qn=ok, max_ratio=float(ratio[worst]), p_worst=int(ps[worst])),
              open(OUT + "pair_d0_r32.json", "w"))


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "real":
        run_real()
    elif what == "family":
        run_family(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None,
                   int(sys.argv[4]) if len(sys.argv) > 4 else 200)
    elif what == "oneclass":
        run_oneclass()
    elif what == "d0":
        run_d0()


def run_famfail(pmax, cap=200, limit=None):
    """family members with a pair above F that the single-gear certificate misses: try two gears;
    plus the exhibit of the wrap-failure member on record."""
    gears = [g for g in PRIMES if g <= pmax]
    qn = NEXT[pmax]
    ranges = [range(1, (g - 1) // 2 + 1) for g in gears]
    t0 = time.time(); members = 0; fails = []; two_ok = two_fail = 0
    for vs in itertools.product(*ranges):
        if limit and members >= limit:
            break
        members += 1
        teeth = {g: {v, g - v} for g, v in zip(gears, vs)}
        P, ops = sieve_openings(gears, teeth)
        gaps = cyc_gaps(ops, P); F = int(gaps.max()); prev = np.roll(gaps, 1); sums = prev + gaps
        for i in np.flatnonzero(sums > F)[:cap]:
            x = int(ops[i]); gL = int(prev[i]); gR = int(gaps[i]); need = gL + gR - qn
            c, _, _ = single_gear_cert(x, gears, teeth)
            if c < need:
                c2 = two_gear_cert(x, gears, teeth, need)
                two_ok += (c2 >= need); two_fail += (c2 < need)
                fails.append(dict(v=vs, x=x, gL=gL, gR=gR, F=F, cert1=c, cert2=c2, need=need,
                                  wrap=(x == 0), res=[x % g for g in gears]))
    print(f"V({pmax}) famfail: members={members} uncertified-by-one-gear pairs={len(fails)} "
          f"two-gear ok={two_ok} two-gear fail={two_fail} {time.time()-t0:.0f}s")
    for f in fails[:12]:
        print("   ", f)
    json.dump(dict(pmax=pmax, members=members, fails=fails), open(OUT + f"pair_famfail_V{pmax}_r32.json", "w"), indent=1)


def run_exhibit():
    gears = [5, 7, 11, 13, 17, 19]; vs = (1, 1, 4, 3, 5, 2); qn = 23
    teeth = {g: {v, g - v} for g, v in zip(gears, vs)}
    P, ops = sieve_openings(gears, teeth)
    gaps = cyc_gaps(ops, P); F = int(gaps.max()); prev = np.roll(gaps, 1); sums = prev + gaps
    d0 = int(ops[1]); F2 = int(sums.max())
    pg = {}
    c, q0, T = single_gear_cert(0, gears, teeth, pg)
    c2 = two_gear_cert(0, gears, teeth, 2 * d0 - qn)
    print(f"EXHIBIT V(19) member v={vs}: P={P} F={F} F2={F2} d0={d0} 2d0-q'={2*d0-qn} > F: {2*d0-qn > F}; "
          f"F2 == 2 d0: {F2 == 2*d0}; single-gear cert at x=0: {c} (per gear {pg}); two-gear cert at 0: {c2}")
    cov = {t: struck_by(t, gears, teeth) for t in range(1, d0)}
    print("   coverers of columns 1..d0-1:", cov)
    nonwrap = int(sums[1:].max())
    print(f"   non-wrap max pair {nonwrap} <= F + q' = {F+qn}: {nonwrap <= F+qn}")


if __name__ == "__main__" and sys.argv[1] in ("famfail", "exhibit"):
    if sys.argv[1] == "famfail":
        run_famfail(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 200,
                    int(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        run_exhibit()
