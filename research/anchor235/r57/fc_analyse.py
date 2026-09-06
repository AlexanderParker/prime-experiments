"""fc_analyse.py -- the summary tables of branch 4.i.a, assembled from fc_frontier.json,
fc_top31.json and the m29 profile arrays.  No new sieving.

Tables:
  T1 per-rung summary (budget slack and its location, top slack, letters, legality)
  T2 the top law: Rest(F_old) against n1(F_old) and N(F_old), split by interior-legality
  T3 the collapse profile: Rest, Rest_2, Rest_3, occ-with-letter-neighbour, n1, n1_0 by a/F_old
  T4 the rarity test: has(a) against m(a) * pbar, and n1(a) against the rarity null
  T5 the shape of s: strict local minima, convexity, minimiser
  T6 the weakened strengthenings: c = max(Rest) - q', and the least a_0 with Rest <= q' above it
  T7 coverage of the three available bounds at each rung
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

F2 = {2: 3, 5: 8, 7: 11, 11: 16, 18: 25, 25: 31, 34: 39, 43: 55}   # F_2(M) by F(M), file 2g.i


def load():
    d = json.load(open(os.path.join(OUT, "fc_frontier.json")))
    rungs = {}
    for k, v in d.items():
        rungs[int(k)] = {kk: ({int(a): b for a, b in vv.items()} if isinstance(vv, dict) else vv)
                         for kk, vv in v.items()}
    t = json.load(open(os.path.join(OUT, "fc_top31.json")))
    spec29 = np.load(os.path.join(OUT, "spec29_fc.npy"))
    n1_29 = np.load(os.path.join(OUT, "n1_29.npy"))
    N_29 = np.load(os.path.join(OUT, "N_29.npy"))
    has29 = np.load(os.path.join(OUT, "has_29.npy"))
    R = {int(a): b for a, b in t["Rest"].items()}
    rungs[31] = dict(q=31, Fold=43, F=58, aL=10, bL=21,
                     Rest=R,
                     R2={int(a): b for a, b in t["R2"].items()},
                     R3={int(a): b for a, b in t["R3"].items()},
                     R4={int(a): b for a, b in t["R4"].items()},
                     s={int(a): 43 + 31 - int(a) - R[int(a)] for a in R},
                     m={a: int(spec29[a]) for a in R},
                     n1={a: int(n1_29[a]) for a in R},
                     N={a: int(N_29[a]) for a in R},
                     has={a: int(has29[a]) for a in R},
                     wit={int(a): b for a, b in t["wit"].items()},
                     null={})
    # rarity null for m29
    tail = np.cumsum(spec29[::-1])[::-1]
    Nt = int(spec29.sum())
    for a in rungs[31]["Rest"]:
        ok = np.flatnonzero(2 * spec29[a] * tail >= Nt)
        rungs[31]["null"][a] = int(ok.max()) if ok.size else 0
    return rungs


def legal(a, q, aL, bL):
    return (a % q in (0, aL, bL)) and a >= aL


def main():
    rungs = load()
    L, W = [], None
    W = L.append
    order = [7, 11, 13, 17, 19, 23, 29, 31]

    W("=== T1. Per-rung summary of the frontier and its slack ===")
    W("rung q' | F_old | F | F_old+q' | budget slack | a* (argmin s) | a*/F_old | "
      "record attained at a | s(F_old) | Rest(F_old) | letters | F_old mod q' | F_old legal?")
    for q in order:
        r = rungs[q]
        Fo, F = r["Fold"], r["F"]
        s = r["s"]
        realised = sorted(s)
        mins = min(s.values())
        amins = [a for a in realised if s[a] == mins]
        peaks = [a for a in realised if a + r["Rest"][a] == F]
        W(f"{q} | {Fo} | {F} | {Fo+q} | {mins} | {amins} | "
          f"{[round(a/Fo,3) for a in amins]} | {peaks} | {s[Fo]} | {r['Rest'][Fo]} | "
          f"({r['aL']}, {r['bL']}) | {Fo % q} | "
          f"{'YES' if legal(Fo, q, r['aL'], r['bL']) else 'no'}")

    W("")
    W("=== T2. The top law:  Rest(F_old) = N(F_old) if F_old is interior-legal, else n1(F_old) ===")
    W("rung q' | F_old | legal? | Rest(F_old) | n1(F_old) | N(F_old) | predicted | match?")
    ok = 0
    for q in order:
        r = rungs[q]
        Fo = r["Fold"]
        lg = legal(Fo, q, r["aL"], r["bL"])
        pred = r["N"][Fo] if lg else r["n1"][Fo]
        good = pred == r["Rest"][Fo]
        ok += good
        W(f"{q} | {Fo} | {'YES' if lg else 'no'} | {r['Rest'][Fo]} | {r['n1'][Fo]} | "
          f"{r['N'][Fo]} | {pred} | {'yes' if good else 'NO'}")
    W(f"  matches: {ok} of {len(order)}")

    W("")
    W("=== T3. The collapse profile at the three top rungs (a >= 0.4 F_old) ===")
    for q in (23, 29, 31):
        r = rungs[q]
        Fo = r["Fold"]
        W(f"-- rung {q} (F_old = {Fo}, letters {r['aL']}, {r['bL']}) --")
        W("  a | a/F_old | Rest | Rest_2 | Rest_3 | Rest_4+ | m(a) | n1(a) | N(a) | n1_0(a) | "
          "occ with letter nbr | a legal? | J of attaining word")
        for a in sorted(r["Rest"]):
            if a < 0.4 * Fo:
                continue
            r4 = r.get("R4", {}).get(a, -1)
            W(f"  {a} | {a/Fo:.3f} | {r['Rest'][a]} | {r['R2'].get(a,-1)} | {r['R3'].get(a,-1)} | "
              f"{r4} | {r['m'][a]} | {r['n1'][a]} | {r['N'][a]} | {r['null'][a]} | "
              f"{r['has'][a]} | {'YES' if legal(a,q,r['aL'],r['bL']) else 'no'} | "
              f"{len(r['wit'][a][1])}")
        z = [a for a in sorted(r["Rest"]) if r["has"][a] == 0]
        zz = [a for a in z if not legal(a, q, r["aL"], r["bL"])]
        bad = [a for a in zz if r["R3"].get(a, -1) >= 0 or r.get("R4", {}).get(a, -1) >= 0]
        W(f"  sizes with NO letter-sized neighbour anywhere: {z}")
        W(f"  of those, not themselves interior-legal: {zz}; of those, any J >= 3 fusion? "
          f"{bad if bad else 'none (0 exceptions)'}")

    W("")
    W("=== T4. Rarity against suppression ===")
    W("(a) letter-neighbour availability: has(a) against m(a) * pbar, pbar = sum has / sum m")
    for q in (23, 29, 31):
        r = rungs[q]
        Fo = r["Fold"]
        tm = sum(r["m"].values()); th = sum(r["has"].values())
        pbar = th / tm
        W(f"-- rung {q}: pbar = {pbar:.4f}; a | m(a) | has(a) | expected m(a)*pbar | ratio")
        for a in sorted(r["Rest"]):
            if a < 0.6 * Fo:
                continue
            e = r["m"][a] * pbar
            W(f"   {a} | {r['m'][a]} | {r['has'][a]} | {e:.2f} | "
              f"{(r['has'][a]/e if e else float('nan')):.3f}")
    W("(b) neighbour size: n1(a) against the rarity null n1_0(a), by a/F_old")
    W("rung | a/F_old band | cells | mean n1 | mean n1_0 | mean (n1_0 - n1) | max deficit")
    for q in (23, 29, 31):
        r = rungs[q]
        Fo = r["Fold"]
        for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
            cells = [a for a in sorted(r["Rest"]) if lo <= a / Fo < hi]
            if not cells:
                continue
            d = [r["null"][a] - r["n1"][a] for a in cells]
            W(f"{q} | [{lo}, {hi}) | {len(cells)} | "
              f"{np.mean([r['n1'][a] for a in cells]):.1f} | "
              f"{np.mean([r['null'][a] for a in cells]):.1f} | {np.mean(d):.1f} | {max(d)}")

    W("")
    W("=== T5. The shape of s(a) on a >= a_L ===")
    W("rung | cells | strict local minima (at a) | convex? | #negative 2nd differences | "
      "s at a_L | s min | s(F_old)")
    for q in order:
        r = rungs[q]
        Fo = r["Fold"]
        ra = [a for a in sorted(r["s"]) if a >= r["aL"]]
        sv = [r["s"][a] for a in ra]
        loc = [ra[i] for i in range(1, len(sv) - 1) if sv[i] < sv[i-1] and sv[i] < sv[i+1]]
        d2 = [sv[i+1] - 2*sv[i] + sv[i-1] for i in range(1, len(sv)-1)]
        W(f"{q} | {len(ra)} | {len(loc)} {loc} | {'yes' if all(x >= 0 for x in d2) else 'no'} | "
          f"{sum(1 for x in d2 if x < 0)} | {sv[0]} | {min(sv)} | {r['s'][Fo]}")

    W("")
    W("=== T6. The weakened strengthenings ===")
    W("rung q' | max Rest | at a | max Rest - q' | least a_0 with Rest(a) <= q' for all a >= a_0 | "
      "a_0/F_old | violating a")
    cs = []
    a0s = []
    for q in order:
        r = rungs[q]
        Fo = r["Fold"]
        mr = max(r["Rest"].values())
        am = [a for a in sorted(r["Rest"]) if r["Rest"][a] == mr]
        viol = [a for a in sorted(r["Rest"]) if r["Rest"][a] > q]
        a0 = (max(viol) + 1) if viol else 1
        cs.append(mr - q)
        a0s.append(a0 / Fo)
        W(f"{q} | {mr} | {am} | {mr - q} | {a0} | {a0/Fo:.3f} | {viol if viol else 'none'}")
    W(f"  c = max over rungs of (max Rest - q') = {max(cs)}; "
      f"least uniform a_0 / F_old = {max(a0s):.3f}")

    W("")
    W("=== T7. Coverage of the three available bounds, per rung ===")
    W("For each a the budget needs Rest(a) <= F_old + q' - a.  Bounds:")
    W("  (i)  deep-chain cap   Rest(a) <= (J_max - 1) a          [chain law, J_max = 1 + D]")
    W("  (ii) neighbour law    Rest(a) <= N(a) <= F_2(M)         [2g.i, J <= 3 only]")
    W("  (iii) top bound       Rest(a) <= q'                     [for a >= a_0]")
    W("rung | J_max | F_2 | a covered by (i) | by (ii, J<=3 only) | by (iii) | UNCOVERED band | "
      "record attained at a | inside the band?")
    Jmax = {7: 3, 11: 2, 13: 3, 17: 3, 19: 3, 23: 4, 29: 3, 31: 5}
    for q in order:
        r = rungs[q]
        Fo, F = r["Fold"], r["F"]
        f2 = F2[Fo]
        cov_i = [a for a in sorted(r["Rest"]) if (Jmax[q] - 1) * a <= Fo + q - a]
        cov_ii = [a for a in sorted(r["Rest"]) if a + f2 <= Fo + q]
        a0 = max([a for a in sorted(r["Rest"]) if r["Rest"][a] > q], default=0) + 1
        cov_iii = [a for a in sorted(r["Rest"]) if a >= a0]
        cov = set(cov_i) | set(cov_ii) | set(cov_iii)
        unc = [a for a in sorted(r["Rest"]) if a not in cov]
        peaks = [a for a in sorted(r["Rest"]) if a + r["Rest"][a] == F]
        W(f"{q} | {Jmax[q]} | {f2} | a <= {max(cov_i) if cov_i else '-'} | "
          f"a <= {max(cov_ii) if cov_ii else '-'} | a >= {a0} | "
          f"{unc if unc else 'none'} | {peaks} | "
          f"{[a in unc for a in peaks]}")

    W("")
    W("=== T8. Candidate sufficient statement: max over interior-legal a of (a + N(a)) ===")
    W("rung | max over legal a of (a + N(a)) | at a | F_old + q' | holds? | "
      "max over illegal a of (a + n1(a)) | at a | holds?")
    for q in order:
        r = rungs[q]
        Fo = r["Fold"]
        lg = [a for a in sorted(r["Rest"]) if legal(a, q, r["aL"], r["bL"])]
        il = [a for a in sorted(r["Rest"]) if not legal(a, q, r["aL"], r["bL"])]
        v1 = max([(a + r["N"][a], a) for a in lg], default=(0, 0))
        v2 = max([(a + min(r["n1"][a], a), a) for a in il], default=(0, 0))
        W(f"{q} | {v1[0]} | {v1[1]} | {Fo+q} | {'yes' if v1[0] <= Fo+q else 'NO'} | "
          f"{v2[0]} | {v2[1]} | {'yes' if v2[0] <= Fo+q else 'NO'}")

    W("")
    W("=== T9. The depth surplus: where Rest(a) exceeds the neighbour sum N(a) ===")
    W("(a J <= 3 fusion has rest <= N(a); a surplus means a J >= 4 fusion attains)")
    W("rung | #a with Rest > N | those a | max surplus | at a | J of that word | word")
    for q in order:
        r = rungs[q]
        ex = [a for a in sorted(r["Rest"]) if r["Rest"][a] > r["N"][a]]
        if ex:
            mx = max(ex, key=lambda a: r["Rest"][a] - r["N"][a])
            W(f"{q} | {len(ex)} | {ex} | {r['Rest'][mx] - r['N'][mx]} | {mx} | "
              f"{len(r['wit'][mx][1])} | {r['wit'][mx][1]}")
        else:
            W(f"{q} | 0 | none | - | - | - | -")

    W("")
    W("=== T10. The fusion-rate identity:  fused / occ = 4/q' generic, 3/q' at +-d, "
      "2/q' at 0 (mod q') ===")
    W("rung | sizes tested | matches | exceptions | interior/occ matches (0, 1/q', 2/q')")
    for q in order:
        r = rungs[q]
        if "occ" not in r:
            continue
        n = ok1 = ok2 = 0
        exc = []
        for a in sorted(r["occ"]):
            if r["occ"][a] == 0:
                continue
            n += 1
            cls = 0 if a % q == 0 else (1 if a % q in (r["aL"], r["bL"]) else 2)
            predf = {0: 2, 1: 3, 2: 4}[cls]
            predi = {0: 2, 1: 1, 2: 0}[cls]
            f = r["fus"][a] * q / r["occ"][a]
            i = r["inte"][a] * q / r["occ"][a]
            if abs(f - predf) < 1e-9:
                ok1 += 1
            else:
                exc.append((a, f, predf))
            if abs(i - predi) < 1e-9:
                ok2 += 1
        W(f"{q} | {n} | {ok1} | {exc if exc else 'none'} | {ok2} of {n}")

    W("")
    W("=== T11. The attaining fusion in the interior band a in [0.5, 0.8] F_old ===")
    W("rung | a | a/F_old | J | word | max at | interior pieces | all interior pieces letters?")
    for q in (17, 19, 23, 29, 31):
        r = rungs[q]
        Fo = r["Fold"]
        for a in sorted(r["Rest"]):
            if not (0.5 * Fo <= a <= 0.8 * Fo):
                continue
            w = r["wit"][a][1]
            i = int(np.argmax(w))
            inner = w[1:-1]
            allL = all((x % q in (0, r["aL"], r["bL"])) and x >= r["aL"] for x in inner) \
                if inner else None
            W(f"{q} | {a} | {a/Fo:.3f} | {len(w)} | {' '.join(map(str,w))} | {i} | "
              f"{inner if inner else '-'} | {allL}")

    txt = "\n".join(L)
    open(os.path.join(OUT, "fc_analyse.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
