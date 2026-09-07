"""The step's object: how the twin gears of machines 1..k act on band k (q = 7..23).

Machines as in research/stack/r1/stacked_squares.py (engine 7..q; machine 2 = [q', q'^2]; machine
k + 1 = [nextprime(g_k^2), .^2]); the anchor 2, 3, 5 is the clock and strikes no slot number.  A
gear is a TWIN GEAR if p - 2 or p + 2 is prime (the partner may sit in a neighbouring machine or,
for 7, in the anchor).  For each banded machine the gears are split three ways for the control:
twin gears / the other gears, and, independently, a size-matched CONTROL set (for each twin gear
the nearest non-twin prime of the same machine not yet taken) / the rest.

Per band k: among the blocked slots (struck by some machine <= k), the share struck only by twin
gears, only by non-twin gears, or both; the same with the control set in the twin gears' place;
per machine the share of the band's slots its twin gears strike.  The collision cycles of every
twin gear pair (p, p + 2) of machines 1..k: the slots whose lower member n is 0, -2, p or -(p + 2)
mod p(p + 2) (both gears strike the same slot: the +4 law on the raw line); the count against
12 per p(p + 2) cycles; and the twin rate in the OTHER two slots of those cycles against the
band's local twin rate, with the neighbour-of-a-hit prediction 1 / ((1 - 2/p)(1 - 2/(p + 2))).

Usage: uv run python research/stack/r2/bandstep.py [q ...]   (default 7 11 13 17 19 23; ~1 GB at q = 23)
"""
import sys, os, json, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
E = (11, 13, 17, 19, 29, 31)


def sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


def strike_array(gears, J):
    s = np.zeros((6, J + 1), dtype=bool)
    for p in gears:
        inv30 = pow(30, -1, p)
        for i, e in enumerate(E):
            s[i, (-e * inv30) % p::p] = True
    return s


def slot_struck(s):
    return np.stack((s[0] | s[1], s[2] | s[3], s[4] | s[5]))


def run_q(q):
    t0 = time.time()
    small = sieve(10**6)
    qsharp = 1
    for p in np.flatnonzero(small[:q + 1]).tolist():
        qsharp *= p
    J = qsharp // 30
    top = qsharp + 32
    isp = sieve(top)

    def nextprime(n):
        m = n + 1
        while not isp[m]:
            m += 1
        return int(m)

    machines = [(7, q)]
    g = nextprime(q)
    while g * g <= qsharp:
        machines.append((g, g * g))
        g = nextprime(g * g)
    gks = [7] + [m[0] for m in machines[1:]] + [g]
    ar = np.arange(J + 1, dtype=np.int64)
    isp6 = np.stack([isp[30 * ar + e] for e in E])
    twin_slot = np.stack((isp6[0] & isp6[1], isp6[2] & isp6[3], isp6[4] & isp6[5]))   # (3, J+1)
    del ar
    # gear classes per machine
    per_machine = []
    for (lo, hi) in machines:
        gears = np.flatnonzero(isp[lo:hi + 1]).astype(np.int64) + lo
        gl = gears.tolist()
        is_twin = np.array([bool(isp[p - 2] or isp[p + 2]) for p in gl])
        twin_g = gears[is_twin].tolist()
        non_g = gears[~is_twin].tolist()
        # control: nearest non-twin prime of the same machine, without replacement
        taken = set()
        ctl = []
        non_arr = np.array(non_g)
        for p in twin_g:
            order = np.argsort(np.abs(non_arr - p))
            for o in order.tolist():
                if o not in taken:
                    taken.add(o)
                    ctl.append(int(non_arr[o]))
                    break
        ctl_set = set(ctl)
        notctl = [p for p in gl if p not in ctl_set]
        pairs = [(p, p + 2) for p in twin_g if p + 2 in set(twin_g)]
        per_machine.append({"range": (lo, hi), "gears": gl, "twin": twin_g, "non": non_g, "ctl": ctl,
                            "notctl": notctl, "pairs_inside": pairs,
                            "S_twin": strike_array(twin_g, J), "S_non": strike_array(non_g, J),
                            "S_ctl": strike_array(ctl, J), "S_notctl": strike_array(notctl, J)})
        print(f"  q={q} machine {len(per_machine)} [{lo}, {hi}] gears {len(gl)} twin {len(twin_g)} pairs inside {len(pairs)} ({time.time() - t0:.0f}s)", flush=True)
    # all twin gear pairs among the gears of machines 1..K (partners may straddle machines)
    all_gears = sorted(set(p for m in per_machine for p in m["gears"]))
    gear_set = set(all_gears)
    out = {"q": q, "qsharp": qsharp, "machines": [{"range": m["range"], "gears": len(m["gears"]),
                                                    "twin_gears": len(m["twin"]), "pairs_inside": len(m["pairs_inside"]),
                                                    "twin_head": m["twin"][:12], "ctl_head": m["ctl"][:12]} for m in per_machine],
           "bands": []}
    for k in range(1, len(machines) + 1):
        gk = gks[k - 1]
        A = gk * gk
        B = min(gks[k] * gks[k], qsharp + 1)
        if B <= A:
            continue
        # band cycles: first cycle with 30j + 11 >= A, last with 30j + 31 < B
        j0 = -(-(A - 11) // 30)
        j1 = (B - 32) // 30 + 1
        if j1 <= j0:
            continue
        sl = slice(j0, j1)
        ncyc = j1 - j0
        nslots = 3 * ncyc
        tw = twin_slot[:, sl]
        twins = int(tw.sum())
        # blocked-slot decomposition
        by_twin = np.zeros((3, ncyc), dtype=bool)
        by_non = np.zeros((3, ncyc), dtype=bool)
        by_ctl = np.zeros((3, ncyc), dtype=bool)
        by_notctl = np.zeros((3, ncyc), dtype=bool)
        per_m = []
        for i in range(k):
            m = per_machine[i]
            st = slot_struck(m["S_twin"][:, sl]); sn = slot_struck(m["S_non"][:, sl])
            sc = slot_struck(m["S_ctl"][:, sl]); snc = slot_struck(m["S_notctl"][:, sl])
            per_m.append({"machine": i + 1, "twin_gears": len(m["twin"]), "gears": len(m["gears"]),
                          "slots_struck_by_twin_gears": float(st.mean()), "slots_struck_by_non_twin": float(sn.mean()),
                          "slots_struck_by_control": float(sc.mean()),
                          "crt_twin": 1 - float(np.prod([1 - 2 / p for p in m["twin"]])) if m["twin"] else 0.0,
                          "crt_ctl": 1 - float(np.prod([1 - 2 / p for p in m["ctl"]])) if m["ctl"] else 0.0})
            by_twin |= st; by_non |= sn; by_ctl |= sc; by_notctl |= snc
        blocked = by_twin | by_non
        nb = int(blocked.sum())
        assert nb == nslots - twins, "blocked + twins != slots (claim A)"
        assert bool(np.array_equal(blocked, by_ctl | by_notctl))
        rec = {"k": k, "gk": gk, "band": [A, B], "cycles": ncyc, "slots": nslots, "twins": twins, "blocked": nb,
               "only_twin": float((by_twin & ~by_non).sum() / nb), "only_non": float((by_non & ~by_twin).sum() / nb),
               "both": float((by_twin & by_non).sum() / nb),
               "only_ctl": float((by_ctl & ~by_notctl).sum() / nb), "only_notctl": float((by_notctl & ~by_ctl).sum() / nb),
               "both_ctl": float((by_ctl & by_notctl).sum() / nb), "per_machine": per_m}
        # collision cycles of every twin gear pair with both gears among machines 1..k
        gears_k = sorted(set(p for m in per_machine[:k] for p in m["gears"]))
        gset_k = set(gears_k)
        pairs = [(p, p + 2) for p in gears_k if p + 2 in gset_k]
        # local twin rate: 20 equal blocks of cycles
        nblk = 20
        edges = np.linspace(j0, j1, nblk + 1).astype(np.int64)
        local = np.zeros(nblk)
        for b in range(nblk):
            seg = twin_slot[:, edges[b]:edges[b + 1]]
            local[b] = seg.mean() if seg.size else 0.0
        coll = []
        tot_obs = 0.0
        tot_exp = 0.0
        tot_pred = 0.0
        for (p, p2) in pairs:
            M = p * p2
            res = [0 % M, (-2) % M, p % M, (-p2) % M]
            n_list = []
            for r in res:
                start = r + M * (-(-(A - r) // M))
                if start < A:
                    start += M
                ns = np.arange(start, B - 2, M, dtype=np.int64)
                r30 = ns % 30
                ns = ns[(r30 == 11) | (r30 == 17) | (r30 == 29)]
                n_list.append(ns)
            ns = np.concatenate(n_list) if n_list else np.zeros(0, np.int64)
            ns = ns[(ns // 30 >= j0) & (ns // 30 < j1)]
            jc = ns // 30
            tslot = np.select([ns % 30 == 11, ns % 30 == 17], [0, 1], 2)
            # sanity: the collided slot is struck by both gears
            lower_ok = ((ns % p == 0) | ((ns + 2) % p == 0)) & ((ns % p2 == 0) | ((ns + 2) % p2 == 0))
            assert bool(lower_ok.all())
            # the other two slots of each collision cycle
            obs = 0
            exp = 0.0
            for d in (1, 2):
                ts = (tslot + d) % 3
                obs += int(twin_slot[ts, jc].sum())
                blk = np.minimum(((jc - j0) * nblk) // ncyc, nblk - 1)
                exp += float(local[blk].sum())
            pred = 1 / ((1 - 2 / p) * (1 - 2 / p2))
            tot_obs += obs; tot_exp += exp; tot_pred += exp * pred
            coll.append({"pair": [p, p2], "collision_slots": int(len(ns)), "expected_12_per_period": 12 * ncyc / M,
                         "other_slot_twins": obs, "expected_at_local_rate": exp, "ratio": obs / exp if exp else None,
                         "predicted_ratio": pred})
        rec["pairs"] = len(pairs)
        rec["collision_total"] = {"observed": tot_obs, "expected": tot_exp, "ratio": tot_obs / tot_exp if tot_exp else None,
                                  "predicted_ratio_weighted": tot_pred / tot_exp if tot_exp else None,
                                  "pairs_with_ge_50": sum(1 for c in coll if c["collision_slots"] >= 50)}
        rec["collisions"] = [c for c in coll if c["collision_slots"] >= 50][:60]
        rec["collisions_head"] = coll[:8]
        out["bands"].append(rec)
        print(f"  q={q} band {k}: slots {nslots} twins {twins} blocked {nb} only-twin {rec['only_twin']:.4f} only-ctl {rec['only_ctl']:.4f} "
              f"both {rec['both']:.4f}; pairs {len(pairs)} collision ratio {rec['collision_total']['ratio']} pred {rec['collision_total']['predicted_ratio_weighted']} ({time.time() - t0:.0f}s)", flush=True)
    out["seconds"] = time.time() - t0
    with open(os.path.join(RES, f"bandstep_{q}.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    qs = [int(a) for a in sys.argv[1:]] or [7, 11, 13, 17, 19, 23]
    for q in qs:
        run_q(q)
