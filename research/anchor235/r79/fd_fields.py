"""fd_fields.py -- the per-field census (fields.md sections 2, 3, 5) and the identity gates P2, P8.

For every section (finer p = 11..53; construction: base 3 links 1-3, base 5 links 1-2, base 7 links 1-2,
base 13 link 1) and every machine sight [1, q'^2), q = 7..53:
  per field j: members, left / right members, hit columns |H_j|, both-F_j columns, hit columns by side;
  column signatures (multiset of Omega of the composite members), exactly-one-field share and which field;
  gates: F_j n 5S = 5.F_{j-1} (T2); class rule right iff Omega_- even (T4); squares right-only (T4);
         mirror law (T5): member at (gm - k_g, side -eps_g) is g(6m - 1), at (gm + k_g, side eps_g) is g(6m + 1),
         with Omega one more than the cofactor's; twins from the fields (blind to every F_j) = twins direct.
Usage: uv run python fd_fields.py
"""
import os, json, time
import numpy as np
from collections import Counter
from fd_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
LOG = open(os.path.join(RES, "fields.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def census(sec, spf, gate_mirror=True):
    """the per-field census of a section dict that has omL/omR/ommL/ommR/lpfL/lpfR"""
    cols, L, R = sec["cols"], sec["L"], sec["R"]
    omL, omR, ommL, ommR = sec["omL"], sec["omR"], sec["ommL"], sec["ommR"]
    J = int(max(omL.max(), omR.max()))
    r = dict(name=sec["name"], columns=int(len(cols)), q=sec["q"], pn=sec["pn"], J=J)
    twin = (omL == 1) & (omR == 1)
    struck = ~twin
    r["twins"] = int(twin.sum())
    r["struck"] = int(struck.sum())
    r["both_struck"] = int(((omL >= 2) & (omR >= 2)).sum())
    fields = {}
    for j in range(1, J + 1):
        mL, mR = omL == j, omR == j
        fields[j] = dict(members=int(mL.sum() + mR.sum()), left=int(mL.sum()), right=int(mR.sum()),
                         hit_cols=int((mL | mR).sum()), hit_left_only=int((mL & ~mR).sum()),
                         hit_right_only=int((mR & ~mL).sum()), both=int((mL & mR).sum()))
    r["fields"] = fields
    # signatures of struck columns
    sig = Counter()
    one_field = Counter()
    for oL, oR in zip(omL[struck], omR[struck]):
        c = tuple(sorted(int(o) for o in (oL, oR) if o >= 2))
        sig[c] += 1
        if len(set(c)) == 1:
            one_field[c[0]] += 1
    r["signatures"] = {"+".join(map(str, k)): v for k, v in sorted(sig.items())}
    r["one_field"] = dict(one_field)
    n1 = sum(one_field.values())
    r["one_field_share"] = n1 / r["struck"] if r["struck"] else None
    r["one_field_F2_share"] = one_field[2] / n1 if n1 else None
    # gates
    g = {}
    # T4: class rule
    g["class_rule"] = int(((ommL % 2) != 1).sum() + ((ommR % 2) != 0).sum())  # mismatches
    # squares right-only, and which squares
    sqL = [int(x) for x in L[(omL == 2)] if int(np.sqrt(x)) ** 2 == x]
    sqR = [int(x) for x in R[(omR == 2)] if int(np.sqrt(x)) ** 2 == x and is_prime(int(np.sqrt(x)))]
    g["squares_on_left"] = len(sqL)
    r["square_field"] = sqR
    # T2: F_j n 5S = 5 F_{j-1}, the cofactor's Omega computed independently through the spf table
    lo, hi = int(L[0]), int(R[-1])
    top = hi // 5 + 10
    mm = np.arange(1, top + 1, dtype=np.int64)
    mm = mm[(mm % 6 == 1) | (mm % 6 == 5)]
    omc, _, _ = omega_arrays(mm, spf)
    omtab = np.full(top + 1, -1, dtype=np.int8)
    omtab[mm] = omc
    mism = 0
    for arr, om in ((L, omL), (R, omR)):
        d5 = arr % 5 == 0
        mism += int((omtab[arr[d5] // 5] + 1 != om[d5]).sum())
    g["T2_mismatch"] = mism
    # twins from the fields = twins direct
    blind_all = np.ones(len(cols), dtype=bool)
    for j in range(2, J + 1):
        blind_all &= ~((omL == j) | (omR == j))
    g["twins_from_fields_mismatch"] = int((blind_all != twin).sum())
    # T5 mirror law
    if gate_mirror:
        idx = {int(c): i for i, c in enumerate(cols)}
        mm_mis = 0
        mm_checked = 0
        for gg in sec["gears"]:
            kg, e = k_of(gg), eps_of(gg)
            m_lo = (lo + gg) // (6 * gg) - 1
            m_hi = hi // (6 * gg) + 2
            for m in range(max(1, m_lo), m_hi):
                c1, c2 = gg * m - kg, gg * m + kg
                if c1 in idx and c2 in idx:
                    i1, i2 = idx[c1], idx[c2]
                    # member at (c1, side -e) is g(6m-1); at (c2, side e) is g(6m+1)
                    a1 = (L if e == 1 else R)[i1]
                    a2 = (R if e == 1 else L)[i2]
                    o1 = (omL if e == 1 else omR)[i1]
                    o2 = (omR if e == 1 else omL)[i2]
                    if int(a1) != gg * (6 * m - 1) or int(a2) != gg * (6 * m + 1):
                        mm_mis += 1
                        continue
                    oc1 = omtab[6 * m - 1]
                    oc2 = omtab[6 * m + 1]
                    if int(o1) != int(oc1) + 1 or int(o2) != int(oc2) + 1:
                        mm_mis += 1
                    mm_checked += 1
        g["T5_checked"] = mm_checked
        g["T5_mismatch"] = mm_mis
    r["gates"] = g
    return r


def fmt_section(r):
    f = r["fields"]
    J = r["J"]
    cells = []
    for j in range(2, J + 1):
        d = f[j]
        cells.append(f"F{j}: {d['members']} ({d['left']}L/{d['right']}R), cols {d['hit_cols']}, both {d['both']}")
    return (f"| {r['name']} | {r['columns']} | {r['q']} | {r['twins']} | {r['struck']} | {r['both_struck']} | "
            + " ; ".join(cells) + f" | {r['one_field_share']:.3f} / {r['one_field_F2_share']:.3f} | "
            + f"T2 {r['gates']['T2_mismatch']}, class {r['gates']['class_rule']}, sqL {r['gates']['squares_on_left']}, "
            + f"T5 {r['gates'].get('T5_mismatch', '-')}/{r['gates'].get('T5_checked', '-')}, tw {r['gates']['twins_from_fields_mismatch']} |")


def main():
    t0 = time.time()
    results = {"sections": [], "machines": []}
    say("# fields census", time.ctime())
    say("| section | columns | q | twins | struck | both struck | fields (members (L/R), hit cols, both-F_j) | one-field share / F2 share | gates |")
    say("|---|---|---|---|---|---|---|---|---|")
    secs = [finer_section(p) for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]]
    secs += construction_sections(3, 2) + construction_sections(5, 2) + construction_sections(7, 2) + construction_sections(13, 1)
    hi = max(int(s["cols"][-1]) * 6 + 2 for s in secs)
    spf = spf_table(hi + 2)
    say(f"spf to {hi} in {time.time() - t0:.0f} s")
    for s in secs:
        section_census(s, spf)
        r = census(s, spf, gate_mirror=(len(s["cols"]) <= 200_000))
        results["sections"].append(r)
        say(fmt_section(r))
    # base 3 link 3 from the gate's saved census
    npz = os.path.join(RES, "base3_link3_omega.npz")
    if os.path.exists(npz):
        d = np.load(npz)
        s = construction_sections(3, 3)[2]
        for k in ("omL", "omR", "ommL", "ommR", "lpfL", "lpfR"):
            s[k] = d[k]
        s["L"], s["R"] = column_members(s["cols"])
        # T2 gate needs spf to hi/5 only for cofactors, but the members themselves are read from the npz;
        # use a spf table to hi/5 + slack for the cofactor omega, then the mirror gate off (size)
        spf3 = spf_table(s["hi"] // 5 + 10)
        r = census(s, spf3, gate_mirror=False)
        results["sections"].append(r)
        say(fmt_section(r))
    # machines: the sight [1, q'^2)
    say("\n## machine sights [1, q'^2), columns 1 .. b-1")
    say("| q | q' | columns | twins | struck | J | fields (members (L/R), hit cols, both) | one-field share / F2 share |")
    say("|---|---|---|---|---|---|---|---|")
    for q in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]:
        qn = nextprime(q)
        b = (qn * qn - 1) // 6
        s = dict(cols=np.arange(1, b, dtype=np.int64), gears=gears_of(q), q=q, pn=qn, name=f"sight q={q}")
        section_census(s, spf)
        r = census(s, spf, gate_mirror=False)
        results["machines"].append(r)
        f = r["fields"]
        cells = "; ".join(f"F{j}: {f[j]['members']} ({f[j]['left']}/{f[j]['right']}), {f[j]['hit_cols']}, {f[j]['both']}" for j in range(2, r["J"] + 1))
        say(f"| {q} | {qn} | {r['columns']} | {r['twins']} | {r['struck']} | {r['J']} | {cells} | {r['one_field_share']:.3f} / {r['one_field_F2_share']:.3f} |")
    json.dump(results, open(os.path.join(RES, "fields.json"), "w"), indent=1, default=int)
    say("done", f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
