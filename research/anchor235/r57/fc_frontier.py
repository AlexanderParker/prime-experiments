"""fc_frontier.py -- the frontier a -> Rest(a), its slack against the budget line, the attaining
fusion words, and the three mechanism quantities (suppression / chain law / rarity), at rungs
5->7 .. 19->23 on full periods and at 23->29 on the streamed m29 period.

Definitions (frontier_collapse.md 0.1):
    every gap of M + q' is a run (p_1..p_J) of consecutive old gaps (merge law);
    mx = max piece, rest = size - mx, Rest(a) = max rest over gaps with mx = a;
    budget line B(a) = F_old + q' - a, slack s(a) = B(a) - Rest(a).
Old-machine quantities: m(v) multiplicity, n1(v) = max single neighbour, N(v) = max neighbour sum,
and the rarity null n1_0(a) = largest r with 2 m(a) * P(neighbour >= r) >= 1.

Writes results/fc_frontier.txt  and  results/fc_frontier.json (machine-readable).
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r56"))
from mf_core import build_levels, u_of  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def letters(q):
    u = u_of(q)
    a = (2 * u) % q
    return (min(a, q - a), max(a, q - a))


def old_stats(size, q=None, aL=None, bL=None):
    """m(v), n1(v), N(v), and (if q given) letter-neighbour availability, cyclic gap sizes."""
    N = size.size
    mx = int(size.max())
    m = np.bincount(size, minlength=mx + 1)
    left = np.roll(size, 1)
    right = np.roll(size, -1)
    n1 = np.zeros(mx + 1, dtype=np.int64)
    Nsum = np.zeros(mx + 1, dtype=np.int64)
    np.maximum.at(n1, size, np.maximum(left, right))
    np.maximum.at(Nsum, size, left + right)
    if q is None:
        return m, n1, Nsum, N
    legal = lambda arr: ((arr % q == 0) | (arr % q == aL) | (arr % q == bL)) & (arr >= aL)
    lb = np.where(legal(left), left, 0)
    rb = np.where(legal(right), right, 0)
    hl = (lb > 0) | (rb > 0)
    has = np.bincount(size[hl], minlength=mx + 1)
    nl = np.zeros(mx + 1, dtype=np.int64)
    np.maximum.at(nl, size, np.maximum(lb, rb))
    return m, n1, Nsum, N, has, nl


def rarity_null(m, Ntot):
    """n1_0(a) = largest r with 2 m(a) * (#gaps >= r)/Ntot >= 1  (0 if none)."""
    mx = m.size - 1
    tail = np.cumsum(m[::-1])[::-1]          # tail[r] = # gaps of size >= r
    out = np.zeros(mx + 1, dtype=np.int64)
    for a in range(mx + 1):
        if m[a] == 0:
            out[a] = -1
            continue
        ok = np.flatnonzero(2 * m[a] * tail >= Ntot)
        out[a] = int(ok.max()) if ok.size else 0
    return out


def word_of(oldsize, t0, J, Nold):
    return [int(oldsize[(t0 + k) % Nold]) for k in range(J)]


def classify(word):
    i = int(np.argmax(word))
    l = sum(word[:i]); r = sum(word[i + 1:])
    side = "left-heavy" if l > r else ("right-heavy" if r > l else "balanced")
    pos = "end" if (i == 0 or i == len(word) - 1) else "interior"
    return i, l, r, side, pos


def frontier_from_arrays(oldsize, Nold, newpos, order, size, Onew):
    """returns Rest array, witness dict a -> (rest, word, x), fusion counts."""
    n = size.size
    Jm = int(order.max())
    mx = np.zeros(n, dtype=np.int64)
    for k in range(Jm):
        act = order > k
        idx = np.flatnonzero(act)
        psz = oldsize[(newpos[idx] + k) % Nold]
        mx[idx] = np.maximum(mx[idx], psz)
    rest = size - mx
    A = int(mx.max())
    Rest = -np.ones(A + 1, dtype=np.int64)
    np.maximum.at(Rest, mx, rest)
    RestJ = {}
    for jj in range(1, Jm + 1):
        r = -np.ones(A + 1, dtype=np.int64)
        selj = order == jj
        if selj.any():
            np.maximum.at(r, mx[selj], rest[selj])
        RestJ[jj] = r
    wit = {}
    for a in np.flatnonzero(Rest >= 0).tolist():
        cand = np.flatnonzero((mx == a) & (rest == Rest[a]))
        i = int(cand[0])
        # prefer the smallest order among the attaining gaps (the simplest fusion)
        Js = order[cand]
        i = int(cand[int(np.argmin(Js))])
        wit[a] = (int(Rest[a]), word_of(oldsize, int(newpos[i]), int(order[i]), Nold), int(Onew[i]))
    return mx, rest, Rest, wit, RestJ


def piece_census_full(oldsize, Nold, newpos, order, A):
    """occ / fused / interior counts per old size, over one new period, vectorised."""
    Jm = int(order.max())
    occ = np.zeros(A + 1, dtype=np.int64)
    fus = np.zeros(A + 1, dtype=np.int64)
    inte = np.zeros(A + 1, dtype=np.int64)
    for k in range(Jm):
        act = np.flatnonzero(order > k)
        psz = oldsize[(newpos[act] + k) % Nold]
        J = order[act]
        occ += np.bincount(psz, minlength=A + 1)
        fus += np.bincount(psz, weights=(J > 1).astype(np.int64), minlength=A + 1).astype(np.int64)
        inte += np.bincount(psz, weights=((k > 0) & (k < J - 1)).astype(np.int64),
                            minlength=A + 1).astype(np.int64)
    return occ, fus, inte


def report(W, J, q, Fold, F, aL, bL, m, n1, Nsum, null, Rest, wit, Nold, census,
           RestJ=None, has=None, nlet=None):
    B = lambda a: Fold + q - a
    realised = [a for a in range(len(Rest)) if Rest[a] >= 0]
    s = {a: int(B(a) - Rest[a]) for a in realised}
    amin = min(realised, key=lambda a: (s[a], -a))
    mins = int(s[amin])
    amin = int(amin)
    peaks = [int(a) for a in realised if a + Rest[a] == F]
    W(f"\n=== rung {Fold}-machine -> +{q}:  F_old = {Fold}, q' = {q}, F = {F}, "
      f"letters (a_L, b_L) = ({aL}, {bL}), F_old mod q' = {Fold % q}, "
      f"F_old interior-legal? {'YES' if (Fold % q) in (0, aL, bL) else 'no'} ===")
    W(f"  budget slack min s = {mins} at a = {amin} (a/F_old = {amin/Fold:.3f}); "
      f"record attained at a = {peaks} (a/F_old = {[round(a/Fold,3) for a in peaks]}); "
      f"top slack s(F_old) = {s[Fold]} = q' - Rest(F_old) = {q} - {Rest[Fold]}")
    legal_a = lambda a: (a % q in (0, aL, bL)) and a >= aL
    W("  a | a/F_old | Rest(a) | a+Rest | B(a) | s(a) | m_old(a) | n1(a) | N(a) | n1_0(a) | "
      "Rest_2 | Rest_3 | Rest_4+ | occ_with_letter_nbr | max_letter_nbr | a_legal | "
      "J | word | max at | left | right | class")
    for a in realised:
        r, word, x = wit[a]
        i, l, rr, side, pos = classify(word)
        r2 = int(RestJ[2][a]) if (RestJ and 2 in RestJ) else -1
        r3 = int(RestJ[3][a]) if (RestJ and 3 in RestJ) else -1
        r4 = max([int(RestJ[j][a]) for j in RestJ if j >= 4], default=-1)
        hh = int(has[a]) if has is not None else -1
        nl = int(nlet[a]) if nlet is not None else -1
        W(f"  {a} | {a/Fold:.3f} | {Rest[a]} | {a+Rest[a]} | {B(a)} | {s[a]} | {int(m[a])} | "
          f"{int(n1[a])} | {int(Nsum[a])} | {int(null[a])} | {r2} | {r3} | {r4} | {hh} | {nl} | "
          f"{'YES' if legal_a(a) else 'no'} | {len(word)} | "
          f"{' '.join(map(str, word))} | {i} | {l} | {rr} | {side}/{pos}")
    # mechanism block
    tg = sorted({Fold, int(round(0.65 * Fold)), amin} | set(peaks))
    tg = [v for v in tg if v < len(m) and m[v] > 0]
    occ_a, fus_a, int_a = census
    W("  MECHANISM at selected a:")
    W("   a | a/F_old | m_old(a) | n1(a) | N(a) | n1_0(a) | Rest(a) | occ in new period | "
      "fused (J>=2) | frac fused | as interior piece | neighbours >= a_L? ")
    for v in tg:
        occ, fu, inte = int(occ_a[v]), int(fus_a[v]), int(int_a[v])
        W(f"   {v} | {v/Fold:.3f} | {int(m[v])} | {int(n1[v])} | {int(Nsum[v])} | {int(null[v])} | "
          f"{int(Rest[v]) if v < len(Rest) else '-'} | {occ} | {fu} | {fu/max(1,occ):.4f} | "
          f"{inte} | n1 {'>=' if n1[v] >= aL else '<'} a_L={aL}")
    return dict(q=int(q), Fold=int(Fold), F=int(F), aL=int(aL), bL=int(bL), amin=amin, mins=mins, peaks=peaks,
                Rest={a: int(Rest[a]) for a in realised},
                s={a: int(s[a]) for a in realised},
                m={a: int(m[a]) for a in realised},
                n1={a: int(n1[a]) for a in realised},
                N={a: int(Nsum[a]) for a in realised},
                null={a: int(null[a]) for a in realised},
                R2={a: (int(RestJ[2][a]) if (RestJ and 2 in RestJ) else -1) for a in realised},
                R3={a: (int(RestJ[3][a]) if (RestJ and 3 in RestJ) else -1) for a in realised},
                R4={a: (max([int(RestJ[j][a]) for j in RestJ if j >= 4], default=-1)
                        if RestJ else -1) for a in realised},
                has={a: (int(has[a]) if has is not None else -1) for a in realised},
                nlet={a: (int(nlet[a]) if nlet is not None else -1) for a in realised},
                occ={a: int(occ_a[a]) for a in realised},
                fus={a: int(fus_a[a]) for a in realised},
                inte={a: int(int_a[a]) for a in realised},
                wit={a: [wit[a][0], wit[a][1], wit[a][2]] for a in realised})


# ---------------------------------------------------------------- rung 23 -> 29 (streamed)

def copy_openings(lv, j, q):
    u = u_of(q)
    r = (lv.O + j * lv.P) % q
    msk = (r != u % q) & (r != (-u) % q)
    idx = np.flatnonzero(msk).astype(np.int64)
    return lv.O[idx] + j * lv.P, j * lv.N + idx


def rung29(L, W):
    q = 29
    lv23 = L[6]
    oldsize, Nold, Fold = lv23.size, lv23.N, lv23.F
    aL, bL = letters(q)
    m23, n1_23, N_23, Ntot23, has23, nlet23 = old_stats(oldsize, q, aL, bL)
    null23 = rarity_null(m23, Ntot23)
    P29 = lv23.P * q
    A = int(Fold)
    Rest = -np.ones(A + 1, dtype=np.int64)
    RestJ = {j: -np.ones(A + 1, dtype=np.int64) for j in (1, 2, 3, 4, 5)}
    wit = {}
    # m29 gap sizes for the next rung's old stats
    spec29 = np.zeros(64, dtype=np.int64)
    n1_29 = np.zeros(64, dtype=np.int64)
    N_29 = np.zeros(64, dtype=np.int64)
    has_29 = np.zeros(64, dtype=np.int64)
    nlet_29 = np.zeros(64, dtype=np.int64)
    aL31, bL31 = letters(31)
    lg = lambda arr: (((arr % 31 == 0) | (arr % 31 == aL31) | (arr % 31 == bL31))
                      & (arr >= aL31))
    cen_occ = np.zeros(Fold + 1, dtype=np.int64)
    cen_fus = np.zeros(Fold + 1, dtype=np.int64)
    cen_int = np.zeros(Fold + 1, dtype=np.int64)
    carry_pos = carry_t = None
    first_pos = first_t = None
    prev_tail = None   # (sizes of last two gaps) for neighbour stats across chunk borders
    buf_sz = np.empty(0, dtype=np.int64)
    for j in range(q + 1):
        if j < q:
            pos, t = copy_openings(lv23, j, q)
            if first_pos is None:
                first_pos, first_t = pos[0], t[0]
        else:
            pos = np.array([first_pos + P29], dtype=np.int64)
            t = np.array([first_t + q * Nold], dtype=np.int64)
        if carry_pos is not None:
            pos = np.concatenate([carry_pos, pos]); t = np.concatenate([carry_t, t])
        sz = pos[1:] - pos[:-1]
        od = t[1:] - t[:-1]
        t0 = t[:-1]
        spec29 += np.bincount(sz, minlength=64)
        # frontier
        Jm = int(od.max())
        mxp = np.zeros(sz.size, dtype=np.int64)
        for k in range(Jm):
            act = np.flatnonzero(od > k)
            psz = oldsize[(t0[act] + k) % Nold]
            mxp[act] = np.maximum(mxp[act], psz)
        rst = sz - mxp
        loc = -np.ones(A + 1, dtype=np.int64)
        np.maximum.at(loc, mxp, rst)
        for jj in RestJ:
            sj = od == jj
            if sj.any():
                np.maximum.at(RestJ[jj], mxp[sj], rst[sj])
        for a in np.flatnonzero(loc >= 0).tolist():
            if loc[a] > Rest[a]:
                Rest[a] = loc[a]
                cand = np.flatnonzero((mxp == a) & (rst == loc[a]))
                i = int(cand[int(np.argmin(od[cand]))])
                wit[a] = (int(loc[a]),
                          word_of(oldsize, int(t0[i]), int(od[i]), Nold), int(pos[i]))
        # piece census, all sizes
        for k in range(Jm):
            act = np.flatnonzero(od > k)
            psz = oldsize[(t0[act] + k) % Nold]
            JJ = od[act]
            cen_occ += np.bincount(psz, minlength=Fold + 1)
            cen_fus += np.bincount(psz, weights=(JJ > 1).astype(np.int64),
                                   minlength=Fold + 1).astype(np.int64)
            cen_int += np.bincount(psz, weights=((k > 0) & (k < JJ - 1)).astype(np.int64),
                                   minlength=Fold + 1).astype(np.int64)
        # m29 neighbour stats: need the gap before and after each gap
        buf_sz = np.concatenate([buf_sz, sz]) if buf_sz.size else sz
        if buf_sz.size > 4:
            core = buf_sz[1:-1]
            np.maximum.at(n1_29, core, np.maximum(buf_sz[:-2], buf_sz[2:]))
            np.maximum.at(N_29, core, buf_sz[:-2] + buf_sz[2:])
            lb = np.where(lg(buf_sz[:-2]), buf_sz[:-2], 0)
            rb = np.where(lg(buf_sz[2:]), buf_sz[2:], 0)
            np.maximum.at(nlet_29, core, np.maximum(lb, rb))
            hl = (lb > 0) | (rb > 0)
            has_29 += np.bincount(core[hl], minlength=64)
            buf_sz = buf_sz[-2:]
        carry_pos, carry_t = pos[-1:], t[-1:]
    F29 = int(np.flatnonzero(spec29).max())
    W("")
    d = report(W, None, q, Fold, F29, aL, bL, m23, n1_23, N_23, null23, Rest, wit, Ntot23,
               (cen_occ, cen_fus, cen_int), RestJ, has23, nlet23)
    return d, spec29, n1_29, N_29, has_29, nlet_29


def main():
    t0 = time.time()
    lines = []
    W = lines.append
    W("=== THE FRONTIER a -> Rest(a) AND ITS SLACK AGAINST THE BUDGET LINE ===")
    W("Rest(a) = max over gaps of M+q' whose largest piece is a, of (size - a).")
    W("Budget line B(a) = F_old + q' - a; slack s(a) = B(a) - Rest(a); budget inequality is s >= 0.")
    L = build_levels()
    data = {}
    for n in range(1, 7):
        data[str(L[n].q)] = do_rung_capture(L, n, W)
    W(f"\n[full-period rungs done {time.time()-t0:.1f}s]")
    d29, spec29, n1_29, N_29, has_29, nlet_29 = rung29(L, W)
    data["29"] = d29
    np.save(os.path.join(OUT, "spec29_fc.npy"), spec29)
    np.save(os.path.join(OUT, "n1_29.npy"), n1_29)
    np.save(os.path.join(OUT, "N_29.npy"), N_29)
    np.save(os.path.join(OUT, "has_29.npy"), has_29)
    np.save(os.path.join(OUT, "nlet_29.npy"), nlet_29)
    W("")
    W("--- m29 old-machine profile (for the 29->31 rung), sizes with m(v) > 0 ---")
    W("  v | m29(v) | n1(v) | N(v) | n1_0(v) | occ_with_31letter_nbr | max_letter_nbr")
    Ntot29 = int(spec29.sum())
    null29 = rarity_null(spec29, Ntot29)
    for v in np.flatnonzero(spec29).tolist():
        W(f"  {v} | {int(spec29[v])} | {int(n1_29[v])} | {int(N_29[v])} | {int(null29[v])} | "
          f"{int(has_29[v])} | {int(nlet_29[v])}")
    json.dump(data, open(os.path.join(OUT, "fc_frontier.json"), "w"))
    W(f"\n[total {time.time()-t0:.1f}s]")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "fc_frontier.txt"), "w").write(txt)
    print(txt)


def do_rung_capture(L, n, W):
    lv, old = L[n], L[n - 1]
    q, Fold = lv.q, old.F
    aL, bL = letters(q)
    m, n1, Nsum, Nold, has, nlet = old_stats(old.size, q, aL, bL)
    null = rarity_null(m, Nold)
    mx, rest, Rest, wit, RestJ = frontier_from_arrays(old.size, old.N, lv.newpos, lv.order,
                                                      lv.size, lv.O)
    cen = piece_census_full(old.size, old.N, lv.newpos, lv.order, int(old.size.max()))
    return report(W, None, q, Fold, lv.F, aL, bL, m, n1, Nsum, null, Rest, wit, Nold, cen,
                  RestJ, has, nlet)


if __name__ == "__main__":
    main()
