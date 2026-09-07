"""The stack by squares: machines built to the square of their first gear, run over [1, q#].

Anchor 2, 3, 5 (period 30).  Cycle j = the integers 30j .. 30j+29 with its three twin slots
(30j+11, 30j+13), (30j+17, 30j+19), (30j+29, 30j+31).  Machine 1 = the engine = primes 7..q.
Machine 2 = primes in [q', q'^2], q' the next prime after q.  Machine k+1 = the primes from the
next prime after machine k's largest gear (the first prime above g_k^2) up to the square of that
first gear.  A machine strikes the multiples of its gears; a slot (a, a+2) is struck if a or a+2
is.  A cycle is OPEN for a machine if none of its three slots is struck, CLOSED if all three are,
MIXED otherwise.  Square part of machine k = numbers below g_k^2 (half-open at the square, see
the document); band = [g_k^2, g_{k+1}^2) capped at q#.

Everything is exact: one full sieve of [1, q#+31], then per machine a boolean strike array over
the six slot numbers of every cycle 0 .. q#/30 (the extra cycle q#/30 is the cycle AT q#, the
numbers q#+11 .. q#+31).  Usage: uv run python research/stack/r1/stacked_squares.py [q ...]
(default 5 7 11 13 17 19 23).  Writes results/stack_<q>.json, results/stack_<q>.txt and
results/positions_<q>.npz.  Memory at q = 23: about 1.2 GB; time about two minutes.
"""
import sys, os, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

E = (11, 13, 17, 19, 29, 31)          # in-cycle offsets; slot t = offset index // 2
SLOT_OF = (0, 0, 1, 1, 2, 2)
HOME_CLASSES = {1, 11, 13, 17, 19, 29}  # residues mod 30 that lie in a twin slot


def sieve(n):
    """Boolean primality array for 0..n."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def runs_of(mask):
    """(start, length) of every maximal run of True in a 1-d boolean array."""
    if mask.size == 0:
        return np.zeros((0, 2), np.int64)
    d = np.diff(np.concatenate((np.zeros(1, np.int8), mask.view(np.int8), np.zeros(1, np.int8))))
    starts = np.flatnonzero(d == 1).astype(np.int64)
    ends = np.flatnonzero(d == -1).astype(np.int64)
    return np.stack((starts, ends - starts), axis=1)


def longest_run(mask, offset=0):
    r = runs_of(mask)
    if len(r) == 0:
        return {"length": 0, "start": None, "count_at_max": 0}
    L = int(r[:, 1].max())
    at = r[r[:, 1] == L]
    return {"length": L, "start": int(at[0, 0] + offset), "count_at_max": int(len(at)),
            "starts": [int(s + offset) for s in at[:8, 0]]}


def residue_sets(g):
    """Per slot t, the residues j (mod g) at which gear g strikes a number of slot t."""
    inv30 = pow(30, -1, g)
    sets = [set(), set(), set()]
    for i, e in enumerate(E):
        sets[SLOT_OF[i]].add((-e * inv30) % g)
    return sets


def crt_densities(gears):
    """Exact CRT (full-period) densities for the gear set: per-slot open, cycle open/closed/mixed.
    P(all slots in T open) = prod_g (1 - |union of residue sets of T| / g)."""
    subsets = {(0,): 1.0, (1,): 1.0, (2,): 1.0, (0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0, (0, 1, 2): 1.0}
    for g in gears:
        g = int(g)
        rs = residue_sets(g)
        for T in subsets:
            u = set()
            for t in T:
                u |= rs[t]
            subsets[T] *= (1.0 - len(u) / g)
    p1 = [subsets[(0,)], subsets[(1,)], subsets[(2,)]]
    p2 = subsets[(0, 1)] + subsets[(0, 2)] + subsets[(1, 2)]
    p3 = subsets[(0, 1, 2)]
    closed = 1.0 - sum(p1) + p2 - p3
    return {"slot_open": p1, "cycle_open": p3, "cycle_closed": closed,
            "cycle_mixed": 1.0 - p3 - closed,
            "mertens_1": float(np.prod([1.0 - 1.0 / g for g in gears])) if len(gears) else 1.0}


def classify(S, lo, hi):
    """S: (6, ncycles) bool strike array; cycles lo..hi-1.  Returns counts and positions."""
    if hi <= lo:
        return {"cycles": 0}, None, None
    s = S[:, lo:hi]
    slot_struck = np.stack((s[0] | s[1], s[2] | s[3], s[4] | s[5]))   # (3, n)
    nstruck = slot_struck.sum(axis=0)
    openm = nstruck == 0
    closedm = nstruck == 3
    out = {"cycles": int(hi - lo),
           "open": int(openm.sum()), "closed": int(closedm.sum()),
           "mixed": int(((nstruck > 0) & (nstruck < 3)).sum()),
           "one_open": int((nstruck == 2).sum()), "two_open": int((nstruck == 1).sum()),
           "slot_strikes": [int(x) for x in slot_struck.sum(axis=1)],
           "number_strikes": [int(x) for x in s.sum(axis=1)]}
    oj = np.flatnonzero(openm)
    cj = np.flatnonzero(closedm)
    out["first_open"] = int(oj[0] + lo) if len(oj) else None
    out["last_open"] = int(oj[-1] + lo) if len(oj) else None
    out["first_closed"] = int(cj[0] + lo) if len(cj) else None
    out["last_closed"] = int(cj[-1] + lo) if len(cj) else None
    out["open_head"] = [int(x + lo) for x in oj[:12]]
    out["closed_head"] = [int(x + lo) for x in cj[:12]]
    out["longest_closed_run"] = longest_run(closedm, lo)
    out["longest_open_run"] = longest_run(openm, lo)
    out["longest_nonopen_run"] = longest_run(~openm, lo)
    return out, oj + lo, cj + lo


def dickman_table(umax=12.0, h=1e-3):
    """Dickman's rho on a grid of step h: rho = 1 on [0, 1], u rho'(u) = -rho(u - 1)."""
    n = int(umax / h) + 1
    r = np.ones(n)
    for i in range(int(1 / h) + 1, n):
        u = i * h
        # trapezoid step of u rho'(u) = -rho(u-1): rho(u) = rho(u-h) - (h/u) * rho(u - 1) (midpoint)
        um = u - h / 2
        r[i] = r[i - 1] - (h / um) * r[max(0, int(round((um - 1) / h)))]
    return r


def rho_at(rho, u, h=1e-3):
    if u <= 1:
        return 1.0
    i = int(round(u / h))
    return float(rho[min(i, len(rho) - 1)])


def smooth_model(u, g, lng, rho, xa, xb, isp):
    """Model of the open-number density of a machine with gears [g, g^2] near x = g^u, per slot
    number (numbers coprime to 30): smooth part rho(u); prime part (30/8) / ln x; smooth x prime
    part = sum over g-smooth m coprime to 30, 7 <= m <= x/g^2, of (30/8) / (m ln(x/m))."""
    x = math.sqrt(xa * xb)
    smooth = rho_at(rho, u)
    prime = (30 / 8) / math.log(x)
    mixed = 0.0
    M = int(x / (g * g))
    for mm in range(7, M + 1):
        if math.gcd(mm, 30) != 1:
            continue
        # g-smooth test by trial division
        t = mm
        f = 7
        ok = True
        while f * f <= t:
            if t % f == 0:
                if f >= g:
                    ok = False
                    break
                while t % f == 0:
                    t //= f
            f += 1
        if ok and t >= g:
            ok = False
        if ok:
            mixed += (30 / 8) / (mm * math.log(x / mm))
    return {"u": u, "smooth": smooth, "prime": prime, "mixed": mixed, "total": smooth + prime + mixed}


def main(qs):
    for q in qs:
        run_q(q)


def run_q(q):
    t0 = time.time()
    small = sieve(200)
    plist_small = [int(p) for p in np.flatnonzero(small)]
    assert q in plist_small and q >= 5
    qsharp = 1
    for p in plist_small:
        if p <= q:
            qsharp *= p
    J = qsharp // 30                      # cycles 0..J-1 inside [1, q#]; cycle J is the cycle at q#
    top = qsharp + 32
    isp = sieve(top)
    ar = np.arange(J + 1, dtype=np.int64)
    N = np.stack([30 * ar + e for e in E])          # (6, J+1) slot numbers
    isp6 = np.stack([isp[30 * ar + e] for e in E])  # primality of the slot numbers
    del ar

    def nextprime(n):
        m = n + 1
        while not isp[m]:
            m += 1
        return int(m)

    # ---- the machines (claim B) ----
    machines = []
    g = 7
    k = 1
    while True:
        if k == 1:
            hi = q                       # the engine: primes 7..q
        else:
            hi = g * g
        hi_eff = min(hi, top)
        gears = np.flatnonzero(isp[g:hi_eff + 1]).astype(np.int64) + g
        m = {"k": k, "g": int(g), "square": int(g * g), "gear_lo": int(g), "gear_hi": int(hi),
             "first_gear": int(gears[0]) if len(gears) else None,
             "last_gear": int(gears[-1]) if len(gears) else None,
             "n_gears": int(len(gears)), "n_gears_below_qsharp": int((gears <= qsharp).sum()),
             "gears": gears}
        machines.append(m)
        if k >= 2 and g * g > qsharp:    # the final machine: its square part covers [1, q#]
            m["final"] = True
            break
        m["final"] = False
        if k == 1:
            g = nextprime(q)
        else:
            g = nextprime(g * g)         # next prime after the largest gear = first prime above g^2
        k += 1
    K = len(machines)
    for i, m in enumerate(machines):
        m["band_lo"] = m["square"]
        m["band_hi"] = min(machines[i + 1]["square"], qsharp + 1) if i + 1 < K else None
    print(f"q={q} q#={qsharp} J={J} machines={K}", flush=True)
    for m in machines:
        print(f"  machine {m['k']}: g={m['g']} square={m['square']} gears {m['first_gear']}..{m['last_gear']} "
              f"({m['n_gears']} gears, {m['n_gears_below_qsharp']} below q#) band=[{m['band_lo']}, {m['band_hi']})", flush=True)

    # ---- strike arrays ----
    S = []
    mf = machines[-1]
    final_by_marking = mf["g"] * mf["g"] <= top or mf["n_gears"] <= 200000
    for m in machines:
        s = np.zeros((6, J + 1), dtype=bool)
        if not m["final"] or final_by_marking:
            for p in m["gears"].tolist():
                inv30 = pow(30, -1, p)
                for i, e in enumerate(E):
                    r = (-e * inv30) % p
                    s[i, r::p] = True
        S.append(s)
    # R = the cofactor of every slot number after the gears of the machines already processed are
    # divided out; at machine k it holds R_k = N with every prime below g_k removed.
    R = N.copy()
    final_home = None
    final_echo = None
    print(f"  strike arrays done {time.time() - t0:.1f}s", flush=True)

    out = {"q": q, "qsharp": qsharp, "J": J, "machines": []}
    positions = {}
    rho = dickman_table()

    lower = np.zeros((6, J + 1), dtype=bool)
    for idx, m in enumerate(machines):
        k = m["k"]
        if m["final"] and not final_by_marking:
            # the final machine through the residual cofactor: every residual above 1 must be a
            # prime >= g_final (no slot number below q#+31 has two factors >= g_final)
            S[idx] = R > 1
            resid = R[R > 1]
            assert bool(np.all(isp[resid])) and int(resid.min()) >= m["g"], "final machine residuals not prime"
            final_home = int((R == N).sum())
            final_echo = int(((R > 1) & (R < N)).sum())
            out["final_home"] = final_home
            out["final_echo"] = final_echo
            del resid
        s = S[idx]
        # consistency of the cofactor with the strike array: a number is open under machine k iff its
        # cofactor R_k is 1 or a prime above g_k^2 (the only ways to have no factor in [g_k, g_k^2])
        if not m["final"] and k >= 2:
            jchk = min(J, max(0, (machines[idx + 1]["square"] - 31) // 30))   # numbers below the next square
            Rp = R[:, :jchk]
            openk = ~s[:, :jchk]
            expl = (Rp == 1) | ((Rp > m["square"]) & isp[Rp])
            assert bool(np.array_equal(openk, expl)), "cofactor and strike array disagree"
            del Rp, openk, expl
        rec = {kk: (v if kk != "gears" else None) for kk, v in m.items()}
        del rec["gears"]
        gears = m["gears"]
        gsq = m["square"]
        T = s & ~lower                                  # the machine's genuine (new) strikes
        # home strikes: slot numbers that are gears of this machine
        is_gear = isp6 & (N >= m["gear_lo"]) & (N <= m["gear_hi"])
        home_classes = int(np.isin(gears % 30, sorted(HOME_CLASSES)).sum()) if len(gears) else 0
        rec["home_strikes_total"] = int(is_gear.sum())
        rec["gears_in_slot_classes"] = home_classes
        # ---- claim A on the square part (numbers below the square, half-open) ----
        sq = N < gsq
        new_on_square = T & sq
        rec["claimA_square_new_strikes"] = int(new_on_square.sum())
        rec["claimA_square_new_not_home"] = int((new_on_square & ~is_gear).sum())
        rec["claimA_square_home"] = int((new_on_square & is_gear).sum())
        # the square itself
        at_sq = N == gsq
        rec["square_is_slot_number"] = bool(at_sq.any())
        rec["square_struck_new"] = bool((T & at_sq).any())
        rec["square_mod_30"] = int(gsq % 30)
        rec["square_cycle"] = int(np.flatnonzero(at_sq.any(axis=0))[0]) if at_sq.any() else None
        # ---- claim A on the band: open under 1..k  <=>  twin ----
        joint = ~(lower | s)
        if m["band_hi"] is not None:
            blo, bhi = m["band_lo"], m["band_hi"]
            for t in range(3):
                a = N[2 * t]
                inband = (a >= blo) & (a + 2 < bhi)
                op = joint[2 * t] & joint[2 * t + 1] & inband
                tw = isp6[2 * t] & isp6[2 * t + 1] & inband
                rec.setdefault("claimA_band_mismatch", []).append(int((op != tw).sum()))
                rec.setdefault("band_twins_by_slot", []).append(int(tw.sum()))
                rec.setdefault("band_slots_by_slot", []).append(int(inband.sum()))
            # boundary cycles at the square: does a twin pair below the square sit in a band cycle?
        # ---- cycle-level parts ----
        J_sq = -(-(gsq - 31) // 30)             # first cycle whose top number reaches the square
        J_sq = max(J_sq, 0)
        boundary = J_sq if (30 * J_sq + 11 < gsq) else None
        J_band0 = J_sq + (1 if boundary is not None else 0)
        if m["band_hi"] is not None:
            nxt_sq = machines[idx + 1]["square"]
            J_top = min(-(-(nxt_sq - 31) // 30), J)   # cycles below this have all numbers < next square
            if J_top > J:
                J_top = J
        else:
            J_top = J
        J_top = min(J_top, J)                          # cycle J (at q#) is never in a band
        rec["cycles_square"] = [0, int(min(J_sq, J))]
        if boundary is not None and boundary >= J:
            boundary = None
        rec["boundary_cycle"] = int(boundary) if boundary is not None else None
        rec["cycles_band"] = [int(J_band0), int(J_top)] if m["band_hi"] is not None else None
        # classification of the machine's own strikes
        c_sq, oj, cj = classify(s, 0, min(J_sq, J)) if min(J_sq, J) > 0 else ({"cycles": 0}, None, None)
        rec["square_class"] = c_sq
        if oj is not None:
            positions[f"m{k}_square_open"] = oj
            positions[f"m{k}_square_closed"] = cj
        if boundary is not None and boundary <= J:
            sb = s[:, boundary]
            rec["boundary_pattern"] = [int(x) for x in sb]
            rec["boundary_numbers"] = [int(x) for x in N[:, boundary]]
            rec["boundary_new_pattern"] = [int(x) for x in T[:, boundary]]
        if m["band_hi"] is not None and J_top > J_band0:
            c_b, oj, cj = classify(s, J_band0, J_top)
            rec["band_class"] = c_b
            positions[f"m{k}_band_open"] = oj
            positions[f"m{k}_band_closed"] = cj
            c_t, _, _ = classify(T, J_band0, J_top)
            rec["band_class_new"] = c_t
            # joint classification on the band (open under 1..k)
            c_j, ojj, cjj = classify(~joint, J_band0, J_top)
            rec["band_class_joint"] = c_j
            positions[f"m{k}_band_joint_open"] = ojj
            # per-machine slot-open fractions on the band and the ratio
            sl = slice(J_band0, J_top)
            nslots = 3 * (J_top - J_band0)
            fr = []
            for i2 in range(idx + 1):
                s2 = S[i2][:, sl]
                so = ~(s2[0] | s2[1]) , ~(s2[2] | s2[3]), ~(s2[4] | s2[5])
                fr.append(float(sum(int(x.sum()) for x in so) / nslots))
            jj = joint[:, sl]
            jo = int((jj[0] & jj[1]).sum() + (jj[2] & jj[3]).sum() + (jj[4] & jj[5]).sum())
            prod = float(np.prod(fr))
            rec["band_slot_open_fraction_by_machine"] = fr
            rec["band_joint_open_slots"] = jo
            rec["band_joint_open_fraction"] = jo / nslots
            rec["band_independence_ratio"] = (jo / nslots) / prod if prod > 0 else None
            # cycle-level joint: cycles open under every machine separately vs jointly
            cyc_open = []
            for i2 in range(idx + 1):
                s2 = S[i2][:, sl]
                cyc_open.append(int((~s2.any(axis=0)).sum()))
            rec["band_cycle_open_by_machine"] = cyc_open
            rec["band_cycle_open_joint"] = int((jj.all(axis=0)).sum())
            # ---- the profile along the band in u = ln x / ln g ----
            lng = math.log(m["g"])
            x0 = 30 * J_band0 + 11
            x1 = 30 * (J_top - 1) + 31
            u0, u1 = math.log(x0) / lng, math.log(x1) / lng
            edges = [u0] + [u for u in np.arange(math.ceil(u0 * 4) / 4, u1, 0.25)] + [u1]
            prof = []
            crt_slot = crt_densities(gears)["slot_open"] if len(gears) else [1.0, 1.0, 1.0]
            crt_slot_mean = float(np.mean(crt_slot))
            for a_u, b_u in zip(edges[:-1], edges[1:]):
                ja = max(J_band0, int(math.ceil((math.exp(a_u * lng) - 11) / 30)))
                jb = min(J_top, int(math.ceil((math.exp(b_u * lng) - 11) / 30)))
                if jb <= ja:
                    continue
                s2 = s[:, ja:jb]
                n = 3 * (jb - ja)
                so = int((~(s2[0] | s2[1])).sum() + (~(s2[2] | s2[3])).sum() + (~(s2[4] | s2[5])).sum())
                nst = (s2[0] | s2[1]).astype(np.int8) + (s2[2] | s2[3]) + (s2[4] | s2[5])
                t2 = T[:, ja:jb]
                tstruck = int((t2[0] | t2[1]).sum() + (t2[2] | t2[3]).sum() + (t2[4] | t2[5]).sum())
                j2 = joint[:, ja:jb]
                jo2 = int((j2[0] & j2[1]).sum() + (j2[2] & j2[3]).sum() + (j2[4] & j2[5]).sum())
                fr2 = []
                for i2 in range(idx + 1):
                    s3 = S[i2][:, ja:jb]
                    fr2.append(float(((~(s3[0] | s3[1])).sum() + (~(s3[2] | s3[3])).sum() + (~(s3[4] | s3[5])).sum()) / n))
                prod2 = float(np.prod(fr2))
                # the open NUMBERS of machine k in this bin, decomposed by the cofactor R_k:
                # smooth (R = 1), prime (R = N), smooth x one prime above g_k^2 (1 < R < N)
                Rb = R[:, ja:jb]
                Nb = N[:, ja:jb]
                ob = ~s2
                nn = 6 * (jb - ja)
                n_smooth = int((ob & (Rb == 1)).sum())
                n_prime = int((ob & (Rb == Nb)).sum())
                n_mixed = int((ob & (Rb > 1) & (Rb < Nb)).sum())
                xa, xb = 30 * ja + 11, 30 * jb + 11
                um = math.log(math.sqrt(xa * xb)) / lng
                model = smooth_model(um, m["g"], lng, rho, xa, xb, isp) if (not m["final"] and k >= 2) else None
                prof.append({"u": [round(a_u, 3), round(b_u, 3)], "cycles": int(jb - ja),
                             "j": [int(ja), int(jb)],
                             "own_slot_open": so / n, "own_over_crt": (so / n) / crt_slot_mean,
                             "new_slot_struck": tstruck / n,
                             "cycle_open": float((nst == 0).mean()), "cycle_closed": float((nst == 3).mean()),
                             "joint_open": jo2 / n, "twins": jo2,
                             "fractions": fr2, "ratio": (jo2 / n) / prod2 if prod2 > 0 else None,
                             "number_open": (n_smooth + n_prime + n_mixed) / nn,
                             "number_smooth": n_smooth / nn, "number_prime": n_prime / nn,
                             "number_mixed": n_mixed / nn, "model": model})
            rec["band_profile"] = prof
        # CRT expectations for the machine's gear set
        if len(gears) and not m["final"]:
            rec["crt"] = crt_densities(gears)
        else:
            rec["crt"] = None
        # whole-range classification (the final machine's square part is everything)
        if m["final"]:
            c_all, oj, cj = classify(s, 0, J)
            rec["all_class"] = c_all
            c_new, _, _ = classify(T, 0, J)
            rec["all_class_new"] = c_new
        # the two top cycles: j = J-1 (inside) and j = J (at q#)
        rec["top_cycle_pattern"] = {"j": int(J - 1), "own": [int(x) for x in s[:, J - 1]],
                                    "numbers": [int(x) for x in N[:, J - 1]]}
        rec["cycle_at_qsharp"] = {"j": int(J), "own": [int(x) for x in s[:, J]],
                                  "numbers": [int(x) for x in N[:, J]],
                                  "new": [int(x) for x in T[:, J]]}
        # per-gear three-gear check on closed cycles of the band: count distinct gears striking
        # (for machines with gears >= 11 a closed cycle needs three distinct gears; verified by the
        # one-number-per-cycle rule: number_strikes per cycle <= 3 when every gear >= 11 strikes one number)
        if m["band_hi"] is not None and J_top > J_band0 and not m["final"]:
            s2 = s[:, J_band0:J_top]
            per_cycle_numbers = s2.sum(axis=0)
            rec["band_max_numbers_struck_per_cycle"] = int(per_cycle_numbers.max())
            # exact three-gear check for closed cycles: each struck slot needs a gear; count gears
            # striking the cycle by direct division on a sample of up to 2000 closed cycles
            cjs = positions.get(f"m{k}_band_closed")
            if cjs is not None and len(cjs):
                samp = cjs if len(cjs) <= 500 else cjs[np.linspace(0, len(cjs) - 1, 500).astype(int)]
                min_gears = 10 ** 9
                hist = {}
                for j0 in samp.tolist():
                    gs = set()
                    for e in E:
                        n = 30 * j0 + e
                        gs |= set(gears[(n % gears) == 0].tolist())
                    hist[len(gs)] = hist.get(len(gs), 0) + 1
                    min_gears = min(min_gears, len(gs))
                rec["band_closed_gear_count_hist"] = {str(a): b for a, b in sorted(hist.items())}
                rec["band_closed_min_gears"] = min_gears
                rec["band_closed_sampled"] = int(len(samp))
        out["machines"].append(rec)
        lower |= s
        # divide this machine's gears (with their powers) out of the cofactor for the next machine
        if not m["final"]:
            for p in gears.tolist():
                pa = p
                while pa <= top:
                    inv = pow(30, -1, pa)
                    for i, e in enumerate(E):
                        r = (-e * inv) % pa
                        R[i, r::pa] //= p
                    pa *= p
        print(f"  machine {k} analysed {time.time() - t0:.1f}s", flush=True)

    # every slot number inside [1, q#] is struck by some machine (the machines partition the primes 7..q#)
    union = np.zeros((6, J), dtype=bool)
    for s in S:
        union |= s[:, :J]
    assert bool(union.all()), "some slot number inside [1, q#] struck by no machine"
    del union, R

    # ---- cross-machine: gears of k+1 = primes of band k; twins of band k = double-home slots of k+1 ----
    cross = []
    for idx in range(K - 1):
        m, m2 = machines[idx], machines[idx + 1]
        blo, bhi = m["band_lo"], m["band_hi"]
        primes_band = np.flatnonzero(isp[blo:bhi]).astype(np.int64) + blo
        g2 = m2["gears"][m2["gears"] <= qsharp]
        same = bool(np.array_equal(primes_band, g2))
        # twins of band k (by number) versus slots with both members gears of k+1
        both_gear = isp6 & (N >= m2["gear_lo"]) & (N <= min(m2["gear_hi"], qsharp))
        tw = 0
        dh = 0
        mism = 0
        for t in range(3):
            a = N[2 * t]
            inband = (a >= blo) & (a + 2 < bhi)
            twin = isp6[2 * t] & isp6[2 * t + 1] & inband
            dbl = both_gear[2 * t] & both_gear[2 * t + 1] & inband
            tw += int(twin.sum()); dh += int(dbl.sum()); mism += int((twin != dbl).sum())
        c = {"k": m["k"], "band": [int(blo), int(bhi)], "primes_in_band": int(len(primes_band)),
             "gears_of_next_below_qsharp": int(len(g2)), "gear_sets_equal": bool(same),
             "twins_in_band": tw, "double_home_slots_of_next": dh, "mismatch": mism}
        # closed positions of k+1 against the squares of closed positions of k
        cj = positions.get(f"m{m['k']}_band_closed")
        cj2 = positions.get(f"m{m2['k']}_band_closed")
        if cj is not None and cj2 is not None and len(cj) and len(cj2):
            closed2 = np.zeros(J + 1, dtype=bool)
            closed2[cj2] = True
            x = (30 * cj.astype(np.int64) + 11)
            jsq = (x * x) // 30
            ok = jsq <= J
            hit = int(closed2[jsq[ok]].sum())
            rate = hit / max(1, int(ok.sum()))
            base = len(cj2) / max(1, (out["machines"][idx + 1]["cycles_band"][1] - out["machines"][idx + 1]["cycles_band"][0]))
            c["square_map"] = {"closed_of_k_mapped": int(ok.sum()), "landing_on_closed_of_next": hit,
                               "rate": rate, "closed_density_of_next": base}
        cross.append(c)
    out["cross"] = cross
    out["seconds"] = time.time() - t0

    with open(os.path.join(RES, f"stack_{q}.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    np.savez_compressed(os.path.join(RES, f"positions_{q}.npz"), **{a: b for a, b in positions.items() if b is not None})
    with open(os.path.join(RES, f"stack_{q}.txt"), "w") as f:
        f.write(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"q={q} done in {out['seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    qs = [int(a) for a in sys.argv[1:]] or [5, 7, 11, 13, 17, 19, 23]
    main(qs)
