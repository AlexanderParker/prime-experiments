"""og_gate.py -- P1: validate the constructions against numbers already on the record.
  (a) the eight twin gears of machine 2 in [9, 121): columns 2, 3, 5, 7, 10, 12, 17, 18 open under {5, 7};
  (b) F(23) = 34;
  (c) the tooth-family killer (1, 1, 2, 6, 1) strikes all eleven columns 49..59 at p = 17;
  (d) the real engine {5..17} leaves exactly columns 52 and 58 open there.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from og_common import (finer_section, family_strikes_column, gears_of, real_teeth, record_F,
                       strikers_of_column, struck_segment)

out = []
# (a)
s = struck_segment(7, 2, 19)  # columns 2..20 (members 11..121); 121 = 11^2 is column 20, right member
open_cols = [2 + i for i in range(19) if not s[i]]
out.append(("twin gears of machine 2 in [9,121)", open_cols, [2, 3, 5, 7, 10, 12, 17, 18]))
# (b)
F23 = record_F(23)
out.append(("F(23)", F23, 34))
# (c)
a, b, cols, gears = finer_section(17)
teeth = [1, 1, 2, 6, 1]
killed = all(family_strikes_column(j, gears, teeth) for j in cols)
out.append(("killer (1,1,2,6,1) strikes every column 49..59", (cols[0], cols[-1], killed), (49, 59, True)))
# (d)
opn = [j for j in cols if not strikers_of_column(j, gears)]
out.append(("real {5..17} open columns in the section", opn, [52, 58]))
out.append(("real teeth of {5..17}", real_teeth(gears), [1, 1, 2, 2, 3]))

ok = True
for name, got, want in out:
    flag = "OK " if got == want else "BAD"
    ok = ok and got == want
    print(f"{flag} {name}: got {got}, expected {want}")
print("GATE", "PASSED" if ok else "FAILED")
