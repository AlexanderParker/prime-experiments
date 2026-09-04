"""Round 30 (constructor), items (a) and (c).

(a) THE IMPLICATION CHAIN WITH CONSTANTS.  With Delta_J = Q*_J - F_2(M),
    Delta_2 = 0 and Delta_J = Delta_{J-1} - eps_J along the maximising chain
    (R91), and L(M) the longest realised legal word (R89: J_max = L + 2):

      THEOREM.  If (A) |eps_J| <= c_A for every 3 <= J <= J_max along the
      maximising chain and (B) L(M) <= c_B, then
          max_J Delta_J <= c_A c_B,   hence   F(M+q') <= F_2(M) + c_A c_B,
      and (D) at the step, F(M+q') <= F(M) + q', follows IF ALSO
          (D2)  F(M) + q' - F_2(M) >= c_A c_B       (the depth-2 half).
      Sharper: max_J Delta_J = max_J sum_{3<=j<=J} (-eps_j) <= sum_j max(0,-eps_j).

    This file tabulates, at every machine where F_2 is exact, the three
    numbers the theorem consumes: max |eps| along the maximising chain (LITERAL
    chain as in R91, and the OVERALL chain, padded letters included), L, and
    the depth-2 slack S_2 = F + q' - F_2.  Values are LOOKED UP (KNOWN_*) or
    COMPUTED from exact sources (evenj_r29.analyse); nothing is filled in.

(c) THE eps MECHANISM.  eps(v) = Phi(u) - Phi(v) - x for v = u.x / x.u.
    LEMMA C0: at a Phi(v)-maximising occurrence (g_kept, u, x, g_out) [kept flank
    on the non-extension side, outer flank on the extension side],
        eps(v) = d - g_out,   d = Phi(u) - x - g_kept >= 0,
    because (g_kept, u, x) is an occurrence of u with flank sum g_kept + x <=
    Phi(u).  MECH-A: Phi is an extreme value over occ(w) draws, so
    Phi(u) - Phi(v) ~ lambda ln(occ(u)/occ(v)); the counted census
    (research/occ_census_r30.py) supplies occ(w) and the association ratio
    r(x|u) = occ(u.x) N / (occ(u) occ(x)).

Usage:  uv run python research/eps_chain_r30.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
R30 = os.path.join(HERE, "data", "r30")
import evenj_r29                                         # noqa: E402

KNOWN_F = {11: 7, 13: 11, 17: 18, 19: 25, 23: 34, 29: 43, 31: 58, 37: 88,
           41: 91, 43: 103, 47: 118, 53: 145, 59: 161, 61: None}
KNOWN_F2 = {11: 11, 13: 16, 17: 25, 19: 31, 23: 39, 29: 55, 31: 68, 37: 90,
            41: 103, 43: 116, 47: 134, 53: 159, 59: 173}
F2_NOTE = {59: "F_2(59) = 173: >= 173 unconditional, <= 173 conditional on the "
               "r28 span cap (no 2-window of m59 in (173, 220])"}
KNOWN_L = {11: 1, 13: 1, 17: 1, 19: 2, 23: 1, 29: 3, 31: 3, 37: 2, 41: 2,
           43: 2, 47: 4, 53: 3}          # m59: A_kill(59 -> 61) not on record
NEXTP = {11: 13, 13: 17, 17: 19, 19: 23, 23: 29, 29: 31, 31: 37, 37: 41,
         41: 43, 43: 47, 47: 53, 53: 59, 59: 61}
# exact per-J Q*_J beyond the evenj sources, on record (R80/R81/R68):
#   m41: Q*_3 <= 116, Q*_4 <= 100 (bounds, not values) - not usable as exact
CENSUS_OCC = {}


def load_occ(y):
    p = os.path.join(R30, "occ_%d_words.json" % y)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        raw = json.load(fh)
    return {tuple(int(t) for t in k.split()): tuple(v) for k, v in raw.items()}


def s_min_of(q1):
    u1 = round(q1 / 6)
    return min(2 * u1, q1 - 2 * u1)


def chain_from(qstar_by_J, F2, jmax):
    """Delta_J and eps_J = Delta_{J-1} - Delta_J along a maximising chain."""
    delta = {2: 0}
    eps = {}
    prev = 0
    for J in range(3, jmax + 1):
        if J not in qstar_by_J:
            break
        delta[J] = qstar_by_J[J] - F2
        eps[J] = prev - delta[J]
        prev = delta[J]
    return delta, eps


def main():
    lines = []

    def out(s=""):
        lines.append(s)
        print(s, flush=True)

    out("=" * 78)
    out("ITEM (a): THE IMPLICATION CHAIN WITH CONSTANTS")
    out("=" * 78)
    out(__doc__.split("(a) ")[1].split("(c) ")[0])
    res = {}
    for y in (11, 13, 17, 19, 23, 29, 31, 37):
        res[y] = evenj_r29.analyse(y)
    out("%-4s %3s %5s %4s %4s %3s | %4s %8s %8s %6s | %6s %6s %7s %7s"
        % ("M", "q'", "s_min", "F", "F_2", "L", "S_2", "eps_lit", "eps_all",
           "maxDel", "cA_lit", "cA*L", "cAall*L", "verdict"))
    table = {}
    for y in sorted(NEXTP):
        q1 = NEXTP[y]
        F, F2, L = KNOWN_F[y], KNOWN_F2.get(y), KNOWN_L.get(y)
        if F2 is None:
            continue
        S2 = F + q1 - F2
        Fn = KNOWN_F.get(q1)
        maxdel = (Fn - F2) if Fn is not None else None     # = max_J Delta_J (R68)
        eps_lit = eps_all = None
        if y in res:
            r = res[y]
            jmax = L + 2
            # overall chain (all word-legal windows)
            qs_all = {J: r["qstar"][J][0] for J in r["qstar"] if J >= 3}
            # literal chain
            lit = {}
            for v, (fs, gL, gR) in r["words"].items():
                if all(x % q1 for x in v):
                    J = len(v) + 2
                    lit[J] = max(lit.get(J, -1), sum(v) + fs)
            d_all, e_all = chain_from(qs_all, F2, jmax)
            d_lit, e_lit = chain_from(lit, F2, jmax)
            # gate: R91 identity and the attainment theorem
            for J in d_all:
                if J >= 3:
                    assert d_all[J] == d_all[J - 1] - e_all[J]
            if all(J in qs_all for J in range(3, jmax + 1)):
                assert max(d_all.values()) == maxdel, ("attainment", y, d_all, maxdel)
                covered = "all J"
            else:
                covered = "J<=%d (source arity)" % max(d_all)
            eps_lit = max((abs(e) for e in e_lit.values()), default=0)
            eps_all = max((abs(e) for e in e_all.values()), default=0)
            table[y] = dict(d_all=d_all, e_all=e_all, d_lit=d_lit, e_lit=e_lit,
                            covered=covered)
        cA_L = "%d" % (eps_lit * L) if eps_lit is not None else "n/r"
        cAa_L = "%d" % (eps_all * L) if eps_all is not None else "n/r"
        if eps_all is not None:
            verdict = ("S_2 >= cA_all*L" if eps_all * L <= S2 else
                       "S_2 < cA_all*L (bound lossy; (D) true by record)")
        else:
            verdict = "eps per letter not on record"
        out("%-4s %3d %5d %4d %4d %3s | %4d %8s %8s %6s | %6s %6s %7s %s"
            % ("m%d" % y, q1, s_min_of(q1), F, F2, L if L is not None else "?",
               S2, eps_lit if eps_lit is not None else "n/r",
               eps_all if eps_all is not None else "n/r",
               maxdel if maxdel is not None else "?",
               eps_lit if eps_lit is not None else "n/r", cA_L, cAa_L, verdict))
    out("")
    out("(n/r = not on record; maxDel = F(M+q') - F_2(M) = max_J Delta_J by the")
    out(" attainment theorem R68; eps_lit = max |eps| along the LITERAL chain,")
    out(" eps_all = along the OVERALL chain, padded letters included)")
    for y, n in F2_NOTE.items():
        out(" note m%d: %s" % (y, n))
    out("")
    out("THE CHAINS THEMSELVES (Delta_J, eps_J), overall then literal:")
    for y, t in table.items():
        out("   m%-3d overall: Delta %s  eps %s   [%s]"
            % (y, {J: v for J, v in t["d_all"].items()}, t["e_all"], t["covered"]))
        out("        literal: Delta %s  eps %s"
            % ({J: v for J, v in t["d_lit"].items()}, t["e_lit"]))
    out("")
    out("TOTALS AT THE MACHINES WITHOUT PER-LETTER DATA (sum of eps over the")
    out("chain to the attaining depth = F_2 - F(M+q'), exact from the record):")
    for y in (41, 43, 47, 53):
        out("   m%d: sum eps = %d - %d = %+d over at most L = %d letters, so some "
            "|eps| >= %d"
            % (y, KNOWN_F2[y], KNOWN_F[NEXTP[y]], KNOWN_F2[y] - KNOWN_F[NEXTP[y]],
               KNOWN_L[y],
               -(-abs(KNOWN_F2[y] - KNOWN_F[NEXTP[y]]) // KNOWN_L[y])))

    # ------------------------------------------------------------ item (c)
    out("")
    out("=" * 78)
    out("ITEM (c): THE eps MECHANISM - decomposition and the counted census")
    out("=" * 78)
    out("%-4s %-14s %-6s %-12s %4s %4s %4s %4s | %5s %5s | %10s %10s %7s %8s"
        % ("M", "v", "side", "u", "x", "Phi_u", "Phi_v", "eps", "d", "g_out",
           "occ_u", "occ_v", "ratio", "assoc"))
    cells = []
    for y in sorted(res):
        r = res[y]
        q1, s_min = r["q1"], r["s_min"]
        W = r["words"]
        occ = load_occ(y)
        N = None
        if occ is not None:
            hp = os.path.join(R30, "occ_%d.npz" % y)
            if os.path.exists(hp):
                import numpy as np
                N = int(np.load(hp)["hist"].sum())
        for v in sorted(W):
            if len(v) < 2:
                continue
            for tag, u, x in (("suffix", v[:-1], v[-1]), ("prefix", v[1:], v[0])):
                if u not in W:
                    continue
                Phi_u, Phi_v = W[u][0], W[v][0]
                eps = Phi_u - Phi_v - x
                # decomposition from the COUNTED census argmax (occ json), falling
                # back to evenj's argmax pair
                if occ is not None and v in occ:
                    _, fs, gL, gR = occ[v]
                    assert fs == Phi_v, ("Phi mismatch", y, v)
                else:
                    fs, gL, gR = W[v]
                g_kept, g_out = (gL, gR) if tag == "suffix" else (gR, gL)
                d = Phi_u - x - g_kept
                assert d >= 0, ("C0 lemma", y, v, tag, d)
                assert d - g_out == eps
                lit = all(t % q1 for t in v)
                ratio = assoc = None
                if occ is not None and u in occ and v in occ and N:
                    ou, ov, ox = occ[u][0], occ[v][0], occ[(x,)][0]
                    if ou > ov:
                        ratio = (Phi_u - Phi_v) / math.log(ou / ov)
                    assoc = ov * N / (ou * ox)
                cells.append(dict(y=y, v=v, tag=tag, u=u, x=x, eps=eps, d=d,
                                  g_out=g_out, lit=lit, ratio=ratio, assoc=assoc,
                                  s_min=s_min, Phi_u=Phi_u, Phi_v=Phi_v,
                                  occ_u=occ[u][0] if occ and u in occ else None,
                                  occ_v=occ[v][0] if occ and v in occ else None))
                out("m%-3d %-14s %-6s %-12s %4d %4d %4d %+4d | %5d %5d | %10s %10s %7s %8s %s"
                    % (y, str(v), tag, str(u), x, Phi_u, Phi_v, eps, d, g_out,
                       occ[u][0] if occ and u in occ else "-",
                       occ[v][0] if occ and v in occ else "-",
                       "%.2f" % ratio if ratio is not None else "-",
                       "%.3g" % assoc if assoc is not None else "-",
                       "literal" if lit else "PADDED"))
    out("")
    out("SCORING (pre-registered in research/data/r30/constructor_prereg_r30.txt)")
    n = len(cells)
    lit_cells = [c for c in cells if c["lit"]]
    pad_cells = [c for c in cells if not c["lit"]]
    out("   cells: %d (literal %d, padded %d)" % (n, len(lit_cells), len(pad_cells)))
    # C1
    big = [c for c in lit_cells if c["d"] > c["s_min"] and c["g_out"] > c["s_min"]]
    out("   C1  literal cells with d > s_min AND g_out > s_min (cancellation): %d  %s"
        % (len(big), [(c["y"], c["v"], c["tag"], c["d"], c["g_out"]) for c in big]))
    out("       -> %s" % ("CONFIRMED" if big else "REFUTED"))
    # C2a
    lr = [c for c in lit_cells if c["ratio"] is not None]
    inb = [c for c in lr if 1.5 <= c["ratio"] <= 4.5]
    out("   C2a literal cells with occ data: %d ; (Phi_u-Phi_v)/ln(occ_u/occ_v) in "
        "[1.5,4.5]: %d ; values %s"
        % (len(lr), len(inb), sorted(round(c["ratio"], 2) for c in lr)))
    out("       -> %s" % ("CONFIRMED" if len(inb) == len(lr) and lr else
                          "REFUTED (%d outside)" % (len(lr) - len(inb))))
    # C2b
    la = [c for c in lit_cells if c["assoc"] is not None]
    inb = [c for c in la if 0.1 <= c["assoc"] <= 10]
    out("   C2b literal association ratios in [1/10, 10]: %d of %d ; values %s"
        % (len(inb), len(la), sorted(round(c["assoc"], 3) for c in la)))
    out("       -> %s" % ("CONFIRMED" if len(inb) == len(la) and la else
                          "REFUTED (%d outside)" % (len(la) - len(inb))))
    # C2c
    fails = [c for c in pad_cells if abs(c["eps"]) > c["s_min"]]
    ok = bad = nodata = 0
    for c in fails:
        if c["assoc"] is None:
            nodata += 1
            continue
        want = (c["assoc"] > 10) if c["eps"] < 0 else (c["assoc"] < 0.1)
        ok += want
        bad += not want
        out("   C2c padded failure m%d %s %s eps %+d : assoc %.3g  %s"
            % (c["y"], c["v"], c["tag"], c["eps"], c["assoc"],
               "as predicted" if want else "AGAINST prediction"))
    out("   C2c -> %d as predicted, %d against, %d without occ data: %s"
        % (ok, bad, nodata, "CONFIRMED" if bad == 0 and ok else
           ("REFUTED" if bad else "NOT DECIDED")))
    # also the sign/association picture on every padded cell
    out("   padded cells, all: (eps, assoc):")
    for c in pad_cells:
        out("      m%d %-12s %-6s eps %+4d  assoc %s  ratio %s"
            % (c["y"], str(c["v"]), c["tag"], c["eps"],
               "%.3g" % c["assoc"] if c["assoc"] is not None else "-",
               "%.2f" % c["ratio"] if c["ratio"] is not None else "-"))
    # C2d
    occ31 = load_occ(31)
    occ37 = load_occ(37)
    out("   C2d occ(25,37; m31) = %s (predicted <= 4) ; occ(27,41; m37) = %s "
        "(predicted <= 4)"
        % (occ31[(25, 37)][0] if occ31 and (25, 37) in occ31 else "n/r",
           occ37[(27, 41)][0] if occ37 and (27, 41) in occ37 else "n/r"))
    # C3
    mx = max(abs(c["eps"]) for c in lit_cells)
    out("   C3  max |eps| over literal cells = %d (predicted <= 6): %s"
        % (mx, "CONFIRMED" if mx <= 6 else "REFUTED"))
    # C4
    if occ31:
        o37 = occ31[(37,)][0]
        N31 = 6226553025
        out("   C4  occ(37; m31) = %d (predicted < 1e5): %s ; per gap %.3e "
            "(predicted in [1e-6, 3e-5]): %s ; Phi(37)/ln occ = %.2f (predicted > 4): %s"
            % (o37, "CONFIRMED" if o37 < 1e5 else "REFUTED", o37 / N31,
               "CONFIRMED" if 1e-6 <= o37 / N31 <= 3e-5 else "REFUTED",
               48 / math.log(o37), "CONFIRMED" if 48 / math.log(o37) > 4 else "REFUTED"))
    # single-letter Phi vs ln occ across machines (R96's band)
    out("")
    out("SINGLE LETTERS: Phi(x) / ln occ(x) at every machine with a counted census")
    for y in sorted(res):
        occ = load_occ(y)
        if occ is None:
            continue
        r = res[y]
        for x in r["Lambda"]:
            if (x,) in occ:
                o, fs, gL, gR = occ[(x,)]
                out("   m%-3d x=%-3d %-6s occ %12d  Phi %3d  Phi/ln occ = %.2f"
                    % (y, x, "padded" if x % r["q1"] == 0 else "literal", o, fs,
                       fs / math.log(o) if o > 1 else float("nan")))
    with open(os.path.join(R30, "eps_chain_r30.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    out("")
    out("all assertions passed")


if __name__ == "__main__":
    main()
