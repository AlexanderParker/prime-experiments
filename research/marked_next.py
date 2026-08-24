"""Round 22 (mechanic): does the marked qualifying spectrum CHAIN one more rung?

Q^[J](23) with q' = 29, q'' = 31 bounds Q_J(29; floor 10) and must sit under
the budget F(29) + 31 = 74 for the 29->31 rung to come from machine 23's
census instead of a machine-29 scan.
"""
import sys
sys.argv = ['x']
src = open('research/marked_qspec.py').read().split('STEPS = ')[0]
exec(src)

print("STEP 23 -> 29 (q'=29, floor 2u''(31)): does the ladder chain one more rung?")
best, bestw, a = marked_spectrum(23, 29, 31, Jmax=7, span_cap=220)
known = {2: 55, 3: 65, 4: 68, 5: 71, 6: 71, 7: 71}   # Q_j(29;10) exact (r17 row, 29->31)
budget = 43 + 31
print(f"  floor a = {a}, budget F(29)+31 = {budget}")
print("   J   Q_J(29) exact   Q^[J](23)   holds?   <= budget?")
mx = 0
for J in range(2, 8):
    kn, mk = known[J], best[J]
    mx = max(mx, mk)
    holds = "YES" if mk >= kn else "*** FAILS ***"
    print(f"  {J:2d}   {kn:8d}        {mk:8d}     {holds:14s} "
          f"{'yes' if mk <= budget else 'NO'}")
print(f"\n  max over J of Q^[J](23) = {mx}  vs budget {budget}  -> "
      f"{'RUNG SURVIVES' if mx <= budget else 'RUNG LOST'}")
