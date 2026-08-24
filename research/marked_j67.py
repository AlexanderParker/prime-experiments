import sys
sys.argv=['x']
exec(open('research/marked_qspec.py').read().split('STEPS = ')[0])
print("STEP 19 -> 23, DEEP: J = 2..7 (Q_6(23;10)=60 vs budget 63 is the crux)")
best, bestw, a = marked_spectrum(19, 23, 29, Jmax=7, span_cap=200)
known = {2:39, 3:43, 4:50, 5:55, 6:60, 7:0}
budget = 34+29
print(f"  floor a = {a}, budget F(23)+29 = {budget}")
print("   J   Q_J(23) exact   Q^[J](19)   holds?   <= budget?")
mx = 0
for J in range(2, 8):
    kn, mk = known[J], best[J]
    mx = max(mx, mk)
    print(f"  {J:2d}   {kn:8d}        {mk:8d}     "
          f"{'YES' if mk>=kn else '*** FAILS ***':14s} {'yes' if mk<=budget else 'NO'}")
print(f"\n  max over J of Q^[J](19) = {mx}   vs budget {budget}"
      f"  ->  {'RUNG SURVIVES' if mx<=budget else 'RUNG LOST'}")
