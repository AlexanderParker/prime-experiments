"""fd_gate.py -- P1: the gates.  Every number here is on the record already.
(a) the eight twin gears of machine 2 in [9, 121) at columns 2, 3, 5, 7, 10, 12, 17, 18;
(b) base 3 section 4 [16129, 260467321): members with no prime factor <= 301 by Omega = 1,2,3,4:
    14218065 / 10991941 / 192920 / 0 (leftover_depth.md L' = 50); twins 1,027,948;
(c) depth histograms of origin_mechanic.md table A at p = 23 {2:33, 3:9, 4:1} and p = 53 {2:65, 3:27, 4:5, 5:1};
    twin counts 276, 29, 6224, 74, 48249, 455 (base 3 link 2, base 5 links 1-2, base 7 links 1-2, base 13 link 1).
Usage: uv run python fd_gate.py [--big]   (--big runs (b), which needs a 1 GB spf table)
"""
import sys, json, time, os
import numpy as np
from collections import Counter
from fd_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
out = {}

# (a)
sec = construction_sections(3, 1)[0]
spf = spf_table(200)
section_census(sec, spf)
twins = sec["cols"][(sec["omL"] == 1) & (sec["omR"] == 1)]
out["a_twins_9_121"] = twins.tolist()
print("(a) twin columns of [9,121):", twins.tolist(), "expected [2,3,5,7,10,12,17,18]")
assert twins.tolist() == [2, 3, 5, 7, 10, 12, 17, 18]

# (c)
spf = spf_table(30_000)
for p, exp in [(23, {2: 33, 3: 9, 4: 1}), (53, {2: 65, 3: 27, 4: 5, 5: 1})]:
    sec = section_census(finer_section(p), spf)
    # depth histogram of origin_mechanic table A: per struck MEMBER? Table A says "depth = Omega of the least-struck member"
    # and the histogram totals equal the number of struck members (43 at p=23: 33+9+1 = 43 = 51 columns - 8 open... no:
    # 51 columns, 8 open, 13 both struck -> 43 struck columns, 56 struck members).  So the histogram is per struck COLUMN,
    # with depth = min Omega over its composite members.
    # Table A: depth = Omega of the member carrying the column's least striker (the smaller lpf among composite members)
    d = []
    for oL, oR, lL, lR in zip(sec["omL"], sec["omR"], sec["lpfL"], sec["lpfR"]):
        c = [(int(l), int(o)) for o, l in ((oL, lL), (oR, lR)) if o >= 2]
        if c:
            d.append(min(c)[1])
    h = {int(k): v for k, v in Counter(d).items()}
    print(f"(c) p={p} depth histogram per struck column (Omega of the least-striker member):", h, "expected", exp)
    assert h == exp, (h, exp)
    out[f"c_hist_{p}"] = h

spf = spf_table(760_000)
for base, link, exp in [(3, 2, 276), (5, 1, 29), (5, 2, 6224), (7, 1, 74), (13, 1, 455)]:
    sec = construction_sections(base, link)[link - 1]
    section_census(sec, spf)
    t = int(((sec["omL"] == 1) & (sec["omR"] == 1)).sum())
    print(f"(c) {sec['name']}: twins {t} expected {exp}")
    assert t == exp
    out[f"c_twins_{base}_{link}"] = t

sec = construction_sections(7, 2)[1]
spf = spf_table(sec["hi"] + 2)
section_census(sec, spf)
t = int(((sec["omL"] == 1) & (sec["omR"] == 1)).sum())
print(f"(c) {sec['name']}: twins {t} expected 48249")
assert t == 48249
out["c_twins_7_2"] = t

if "--big" in sys.argv:
    t0 = time.time()
    sec = construction_sections(3, 3)[2]
    print("(b)", sec["name"], "columns", len(sec["cols"]))
    spf = spf_table(sec["hi"] + 2)
    print("   spf built", round(time.time() - t0), "s")
    section_census(sec, spf)
    print("   census built", round(time.time() - t0), "s")
    t = int(((sec["omL"] == 1) & (sec["omR"] == 1)).sum())
    print("   twins", t, "expected 1027948")
    assert t == 1027948
    # 301-rough members by Omega (both members of every column)
    om = np.concatenate([sec["omL"], sec["omR"]])
    lpf = np.concatenate([sec["lpfL"], sec["lpfR"]])
    rough = lpf > 301
    h = {j: int(((om == j) & rough).sum()) for j in (1, 2, 3, 4)}
    print("   301-rough by Omega", h, "expected {1: 14218065, 2: 10991941, 3: 192920, 4: 0}")
    assert h == {1: 14218065, 2: 10991941, 3: 192920, 4: 0}
    out["b_rough301"] = h
    out["b_twins"] = t
    np.savez_compressed(os.path.join(RES, "base3_link3_omega.npz"), cols=sec["cols"], omL=sec["omL"], omR=sec["omR"],
                        ommL=sec["ommL"], ommR=sec["ommR"], lpfL=sec["lpfL"], lpfR=sec["lpfR"])
    print("   saved", round(time.time() - t0), "s")

json.dump(out, open(os.path.join(RES, "gate.json"), "w"), indent=1)
print("ALL GATES PASSED")
