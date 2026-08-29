"""Round 26 (mechanic): verify a Q*_J witness AT THE TARGET MACHINE.

A lap-phase transfer run reports its witness in the OLD machine's coordinates
(a start opening x0, one phase c per new gear, and which interior openings are
the marks).  Phase c for new gear q means the window sits in lap j with
c = -j*P mod q, so CRT over the new gears gives j and hence the target-machine
address k = x0 + j*P.  This script translates and then checks EVERYTHING from
the definition at the target machine: the J+1 offsets are openings, every other
slot of the span is blocked, and the J-2 middle gaps form a LEGAL KILL WORD for
gear q'' (each in {0,+s,-s} mod q'', prefix sums of range <= 1).

usage: python research/qstar_witness_r26.py OLD q1,q2,... QPP x0 span c1,c2,... m1,m2,...
"""
import sys
from math import prod
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
from akill_verify_r25 import gears, is_open, crt

OLD = int(sys.argv[1]); NEW = [int(x) for x in sys.argv[2].split(',')]
QPP = int(sys.argv[3]); X0 = int(sys.argv[4]); SPAN = int(sys.argv[5])
PH = [int(x) for x in sys.argv[6].split(',')]
MK = [int(x) for x in sys.argv[7].split(',')]
TARGET = NEW[-1]

old = gears(OLD); P = prod(old)
assert is_open(X0, old) and is_open(X0 + SPAN, old), "endpoints not m%d openings" % OLD
interior = [x for x in range(X0 + 1, X0 + SPAN) if is_open(x, old)]
marks = [interior[t] - X0 for t in MK]
js = [(-c * pow(P % q, -1, q)) % q for q, c in zip(NEW, PH)]
j, _ = crt(js, NEW)
k = X0 + j * P
offs = [0] + marks + [SPAN]
gt = gears(TARGET); oset = set(offs)
for t in range(SPAN + 1):
    o = is_open(k + t, gt)
    assert o == (t in oset), f"machine-{TARGET} mismatch at offset {t}: open={o}"
gapw = [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]
mids = gapw[1:-1]
s = (2 * pow(6, -1, QPP)) % QPP
if '--nolegal' in sys.argv:          # floor-1 (unrestricted F_J) witnesses
    print(f"machine-{TARGET} address k = {k:,}")
    print(f"  {len(offs)} openings at k + {offs}; all "
          f"{SPAN + 1 - len(offs)} other slots of the span blocked")
    print(f"  gaps {gapw}  span {sum(gapw)} = {SPAN}  (no legality required: "
          f"this is an F_{len(gapw)} witness)")
    print("  ASSERTIONS PASSED")
    sys.exit()
p = lo = hi = 0
for v in mids:
    r = v % QPP
    assert r in (0, s, (-s) % QPP), f"middle gap {v} is not legal mod {QPP}"
    L = 0 if r == 0 else (1 if r == s else -1)
    p += L; lo, hi = min(lo, p), max(hi, p)
assert hi - lo <= 1, "letter word prefix-sum range > 1"
print(f"machine-{TARGET} address k = {k:,}")
print(f"  {len(offs)} openings at k + {offs}; all {SPAN + 1 - len(offs)} other "
      f"slots of the span blocked (checked slot by slot)")
print(f"  gaps {gapw}  span {sum(gapw)} = {SPAN}")
print(f"  middles {mids}: LEGAL kill word for gear {QPP} "
      f"(V = {{0,{s},{(-s) % QPP}}}, prefix range {hi - lo} <= 1)")
print("  ASSERTIONS PASSED")
